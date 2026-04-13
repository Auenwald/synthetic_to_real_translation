import torch
import argparse
from torch.utils.data import DataLoader, Dataset
from torch import nn
from datasets.dataset_cityscapes import *
from datasets.dataset_synthia import *
from datasets.dataset_synthia_branched import *
from datasets.dataset_synthia_style import *
from datasets.dataset_bdd import *
import numpy as np
import utils
from torch_ema import ExponentialMovingAverage
import pytorch_warmup as warmup
import model_utils
from segformer_crossattention_wrapperv2_late_fusion_branched import *
import os
import json
from losses import CombinedLoss
import math
import torch.nn.functional as F
from torchmetrics.functional import jaccard_index
import random


best_val_mean_IoU = 0
num_classes = 16


def set_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_parser(parser):
    parser.add_argument('--source_path', default='./synthia', required=True)
    parser.add_argument('--target_paths', type=str, nargs='+', default=['./cityscapes'])
    parser.add_argument('--model_name', type=str, default="segformer")
    parser.add_argument('--optimizer', '-o', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1.0e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--use_logging', type=lambda x: x == 'True', default=False)
    parser.add_argument('--log_file', type=str, default='./logs/log.txt')
    parser.add_argument('--weight_averaging', type=lambda x: x == 'True', default=False)
    parser.add_argument('--averaging_interval', type=int, default=20)
    parser.add_argument('--skip_val_source', type=lambda x: x == 'True', default=False)
    parser.add_argument('--decay_factor', type=float, default=0.999)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_synthia_shapes', type=lambda x: x == 'True', default=False)
    parser.add_argument('--train_print_steps', type=int, default=50)
    parser.add_argument('--mode', type=str, default="dct")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--source_dataset_name', type=str, default=None,
                    help='Überschreibt den aus source_path extrahierten Dataset-Namen')


def main():
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()

    BATCH_SIZE      = args.batch_size
    LR              = args.lr
    EPOCHS          = args.epochs
    WEIGHT_DECAY    = args.weight_decay
    DECAY_FACTOR    = args.decay_factor
    WEIGHT_AVERAGING = args.weight_averaging
    SKIP_VAL_SOURCE = args.skip_val_source
    AVERAGING_INTERVAL = args.averaging_interval
    SOURCE_PATH     = args.source_path
    TARGET_PATHS    = args.target_paths
    GPU             = args.gpu
    USE_SYNTHIA_SHAPES = args.use_synthia_shapes
    LOG_PATH        = args.log_file
    PRINT_INTERVAL  = args.train_print_steps
    SEED            = args.seed
    MODE            = args.mode
    SOURCE_DATASET_NAME = args.source_dataset_name.lower().strip() if args.source_dataset_name else SOURCE_PATH.split("/")[-1].lower().strip()
    epoch_modifier  = 0


    print(args)
    print("SOURCE_DATASET_NAME =", SOURCE_DATASET_NAME)
    print("TARGET_PATHS =", TARGET_PATHS)

    set_seed(SEED)

    DEVICE = f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu'
    print(f'Found the following device: {DEVICE}')

    BRANCHED = "branched" in SOURCE_DATASET_NAME

    global num_classes
    num_classes = 16 if "synthia" in SOURCE_PATH else 19

    # --- Dataloader ---
    source_train_data_loader = utils.get_dataloader_from_dataset(
        SOURCE_PATH, SOURCE_DATASET_NAME, 'train',
        batch_size=BATCH_SIZE, shuffle=True,
        use_synthia_shapes=USE_SYNTHIA_SHAPES,
        seed=SEED, num_classes=num_classes
    )
    source_val_data_loader = utils.get_dataloader_from_dataset(
        SOURCE_PATH, SOURCE_DATASET_NAME, 'val',
        batch_size=1, shuffle=False,
        seed=SEED, num_classes=num_classes
    )

    target_val_loaders = {}
    for target_path in TARGET_PATHS:
        target_name = target_path.split("/")[-1].lower().strip()
        target_val_loaders[target_name] = utils.get_dataloader_from_dataset(
            target_path, target_name, 'val',
            batch_size=1, shuffle=False,
            seed=SEED, num_classes=num_classes
        )

    # --- Model ---
    model = SegformerCrossAttentionWrapperV2Branched(
        num_classes=num_classes,
        segformer_name='nvidia/mit-b5',
        mode=MODE,
        fuse_stages=(2,3)
    )
    model = model.to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = None

    if WEIGHT_AVERAGING:
        ema = ExponentialMovingAverage(
            filter(lambda p: p.requires_grad, model.parameters()),
            decay=DECAY_FACTOR
        )
    else:
        ema = None

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    for epoch in range(1 + epoch_modifier, EPOCHS + 1 + epoch_modifier):
        train(
            source_train_data_loader, model, optim, loss_fn,
            DEVICE, ema, scheduler, PRINT_INTERVAL, AVERAGING_INTERVAL,
            SOURCE_DATASET_NAME, BRANCHED
        )

        # --- Validation ---
        if WEIGHT_AVERAGING and ema:
            with ema.average_parameters():
                if not SKIP_VAL_SOURCE:
                    validate(source_val_data_loader, model, DEVICE, LOG_PATH,
                             True, SOURCE_DATASET_NAME, epoch, EPOCHS)
                for target_name, loader in target_val_loaders.items():
                    validate(loader, model, DEVICE, LOG_PATH,
                             True, target_name, epoch, EPOCHS)

        if not SKIP_VAL_SOURCE:
            validate(source_val_data_loader, model, DEVICE, LOG_PATH,
                     False, SOURCE_DATASET_NAME, epoch, EPOCHS)
        for target_name, loader in target_val_loaders.items():
            validate(loader, model, DEVICE, LOG_PATH,
                     False, target_name, epoch, EPOCHS)


def train(train_loader, model, optim, loss_fn, DEVICE, ema, scheduler,
          PRINT_INTERVAL, AVERAGING_INTERVAL, SOURCE_DATASET_NAME, branched=False):
    model.train()

    for i, batch in enumerate(train_loader):

        if batch is None:
            continue

        # Branched: 3-Tuple (image_rgb, image_struct, mask)
        # Normal:   2-Tuple (image, mask)
        if branched:
            if len(batch) != 3:
                continue
            data, data_struct, targets = batch
            if data is None or targets is None:
                continue
            data        = data.to(DEVICE)
            data_struct = data_struct.to(DEVICE)
            targets     = targets.to(DEVICE).long()
        else:
            if len(batch) != 2:
                continue
            data, targets = batch
            if data is None or targets is None:
                continue
            data        = data.to(DEVICE)
            data_struct = None
            targets     = targets.to(DEVICE).long()

        logits = model_utils.get_logits(model, data, data_struct)

        h, w = data.shape[2], data.shape[3]
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)

        loss = loss_fn(logits, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()
        if scheduler:
            scheduler.step()

        if i > 0 and i % AVERAGING_INTERVAL == 0 and ema:
            ema.update()

        if i > 0 and i % PRINT_INTERVAL == 0:
            print(f"[train-{SOURCE_DATASET_NAME}] {i}/{len(train_loader)} "
                  f"loss: {loss.item():.6f} "
                  f"lr: {optim.param_groups[0]['lr']:.2e}")
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                miou = jaccard_index(
                    task='multiclass', ignore_index=255,
                    num_classes=num_classes, preds=preds, target=targets
                ) * 100
                print(f"[train-{SOURCE_DATASET_NAME}] {i}/{len(train_loader)} "
                      f"mIoU: {miou:.2f}")


def validate(val_loader, model, DEVICE, LOG_PATH, applied_ema,
             dataset_name, epoch, max_epochs):
    model.eval()
    suffix = "-ema" if applied_ema else ""
    dataset_key = dataset_name + suffix

    confusion_matrix = torch.zeros(num_classes, num_classes,
                                   dtype=torch.int64, device="cpu")

    for idx, batch in enumerate(val_loader):

        if batch is None:
            continue

        # Val-Loader gibt je nach Dataset 2- oder 3-Tuple zurück
        if len(batch) == 3:
            data, data_struct, targets = batch
        elif len(batch) == 2:
            data, targets = batch
            data_struct = None
        else:
            continue

        if data is None or targets is None:
            continue

        data = data.to(DEVICE)
        if data_struct is not None:
            data_struct = data_struct.to(DEVICE)
        targets = targets.to(DEVICE).long()

        with torch.no_grad():
            output = model_utils.get_logits(model, data, data_struct=data_struct)
            output = F.interpolate(
                output,
                size=utils.get_image_size(dataset_name),
                mode='bilinear',
                align_corners=False
            )

        preds = torch.argmax(output, dim=1)
        confusion_matrix += utils.torch_fast_hist(
            preds.cpu(), targets.cpu(), num_classes, device="cpu"
        )

        if idx % 10 == 0:
            print(f'[val-{dataset_name}{suffix}] Epoch: {epoch}/{max_epochs} '
                  f'Progress: {idx + 1}/{len(val_loader)}')

    miou, per_class_miou = utils.compute_mIoU_and_per_class_from_hist(confusion_matrix)

    scores = {}
    if LOG_PATH and os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0:
        try:
            with open(LOG_PATH, 'r') as f:
                scores = json.load(f)
        except json.JSONDecodeError:
            print(f"Warnung: {LOG_PATH} ungültig. Starte mit leerem Dict.")

    if dataset_key not in scores:
        scores[dataset_key] = {}

    scores[dataset_key][str(epoch)] = {
        "mean_iou": round(miou * 100, 3),
        "per_class_iou": {str(k): round(v * 100, 3) for k, v in per_class_miou.items()}
    }

    if LOG_PATH:
        with open(LOG_PATH, 'w') as f:
            json.dump(scores, f, indent=4)

    print(f'[val-{dataset_name}{suffix}] Epoch: {epoch}/{max_epochs} '
          f'mean-IoU: {miou * 100:.2f}')


if __name__ == '__main__':
    main()