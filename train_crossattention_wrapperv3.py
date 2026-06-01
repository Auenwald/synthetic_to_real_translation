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
import model_utils
from segformer_crossattention_wrapperv3 import SegformerCrossAttentionV3
import os
import json
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
    parser.add_argument('--source_path', required=True)
    parser.add_argument('--target_paths', type=str, nargs='+', default=['./cityscapes'])
    parser.add_argument('--model_name', type=str, default="segformer")
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
    parser.add_argument('--mode', type=str, default="edge")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--aux_loss_weight', type=float, default=0.4)
    parser.add_argument('--source_dataset_name', type=str, default=None)


def main():
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()

    BATCH_SIZE         = args.batch_size
    LR                 = args.lr
    EPOCHS             = args.epochs
    WEIGHT_DECAY       = args.weight_decay
    DECAY_FACTOR       = args.decay_factor
    WEIGHT_AVERAGING   = args.weight_averaging
    SKIP_VAL_SOURCE    = args.skip_val_source
    AVERAGING_INTERVAL = args.averaging_interval
    SOURCE_PATH        = args.source_path
    TARGET_PATHS       = args.target_paths
    GPU                = args.gpu
    USE_SYNTHIA_SHAPES = args.use_synthia_shapes
    LOG_PATH           = args.log_file
    PRINT_INTERVAL     = args.train_print_steps
    SEED               = args.seed
    MODE               = args.mode
    AUX_LOSS_WEIGHT    = args.aux_loss_weight
    SOURCE_DATASET_NAME = (
        args.source_dataset_name.lower().strip()
        if args.source_dataset_name
        else SOURCE_PATH.split("/")[-1].lower().strip()
    )

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
        seed=SEED, num_classes=num_classes, mode = MODE
    )
    source_val_data_loader = utils.get_dataloader_from_dataset(
        SOURCE_PATH, SOURCE_DATASET_NAME, 'val',
        batch_size=1, shuffle=False,
        seed=SEED, num_classes=num_classes, mode = MODE
    )

    target_val_loaders = {}
    for target_path in TARGET_PATHS:
        target_name = target_path.split("/")[-1].lower().strip()
        target_val_loaders[target_name] = utils.get_dataloader_from_dataset(
            target_path, target_name, 'val',
            batch_size=1, shuffle=False,
            seed=SEED, num_classes=num_classes
        )

    # --- Modell (ADE20K Pretraining via Default in SegformerCrossAttentionV3) ---
    model = SegformerCrossAttentionV3(
        num_classes=num_classes,
        mode=MODE,
        aux_loss_weight=AUX_LOSS_WEIGHT,
        bidir_from_layer=2,
    )
    model = model.to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if WEIGHT_AVERAGING:
        ema = ExponentialMovingAverage(
            filter(lambda p: p.requires_grad, model.parameters()),
            decay=DECAY_FACTOR
        )
    else:
        ema = None

    for epoch in range(1, EPOCHS + 1):
        train(
            source_train_data_loader, model, optim,
            DEVICE, ema, PRINT_INTERVAL, AVERAGING_INTERVAL,
            SOURCE_DATASET_NAME, BRANCHED, AUX_LOSS_WEIGHT
        )

        def run_val(use_ema: bool):
            if not SKIP_VAL_SOURCE:
                validate(source_val_data_loader, model, DEVICE, LOG_PATH,
                         use_ema, SOURCE_DATASET_NAME, epoch, EPOCHS)
            for target_name, loader in target_val_loaders.items():
                validate(loader, model, DEVICE, LOG_PATH,
                         use_ema, target_name, epoch, EPOCHS)

        if WEIGHT_AVERAGING and ema:
            with ema.average_parameters():
                run_val(use_ema=True)
        run_val(use_ema=False)


# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

def train(train_loader, model, optim, DEVICE, ema,
          PRINT_INTERVAL, AVERAGING_INTERVAL,
          SOURCE_DATASET_NAME, branched=False, aux_loss_weight=0.4):

    model.train()

    for i, batch in enumerate(train_loader):
        if batch is None:
            continue

        if branched:
            if len(batch) != 3:
                continue
            data, data_struct, targets = batch
        else:
            if len(batch) != 2:
                continue
            data, targets = batch
            data_struct = None

        if data is None or targets is None:
            continue

        data    = data.to(DEVICE)
        targets = targets.to(DEVICE).long()
        if data_struct is not None:
            data_struct = data_struct.to(DEVICE)

        output = model(data, image_struct=data_struct)

        if isinstance(output, tuple):
            logits, aux_logits = output
        else:
            logits, aux_logits = output, None

        h, w = data.shape[2], data.shape[3]
        logits_up = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)

        if aux_logits is not None and aux_loss_weight > 0:
            loss, main_loss, aux_loss = model.compute_loss(logits, aux_logits, targets)
        else:
            loss = F.cross_entropy(logits_up, targets, ignore_index=255)
            main_loss, aux_loss = loss, torch.tensor(0.0)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if i > 0 and i % AVERAGING_INTERVAL == 0 and ema:
            ema.update()

        if i > 0 and i % PRINT_INTERVAL == 0:
            with torch.no_grad():
                preds = torch.argmax(logits_up, dim=1)
                miou  = jaccard_index(
                    task='multiclass', ignore_index=255,
                    num_classes=num_classes, preds=preds, target=targets
                ) * 100

            print(
                f"[train-{SOURCE_DATASET_NAME}] {i}/{len(train_loader)} "
                f"loss: {loss.item():.4f}  "
                f"main: {main_loss.item():.4f}  "
                f"aux: {aux_loss.item():.4f}  "
                f"mIoU: {miou:.2f}  "
                f"lr: {optim.param_groups[0]['lr']:.2e}"
            )


# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------

def validate(val_loader, model, DEVICE, LOG_PATH, applied_ema,
             dataset_name, epoch, max_epochs):
    model.eval()
    suffix      = "-ema" if applied_ema else ""
    dataset_key = dataset_name + suffix

    confusion_matrix = torch.zeros(num_classes, num_classes,
                                   dtype=torch.int64, device="cpu")

    for idx, batch in enumerate(val_loader):
        if batch is None:
            continue

        if len(batch) == 3:
            data, data_struct, targets = batch
        elif len(batch) == 2:
            data, targets = batch
            data_struct = None
        else:
            continue

        if data is None or targets is None:
            continue

        data    = data.to(DEVICE)
        targets = targets.to(DEVICE).long()
        if data_struct is not None:
            data_struct = data_struct.to(DEVICE)

        with torch.no_grad():
            output = model(data, image_struct=data_struct, return_aux=False)
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
            print(f"Warnung: {LOG_PATH} ungültig. Starte neu.")

    scores.setdefault(dataset_key, {})[str(epoch)] = {
        "mean_iou":      round(miou * 100, 3),
        "per_class_iou": {str(k): round(v * 100, 3) for k, v in per_class_miou.items()},
    }

    if LOG_PATH:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, 'w') as f:
            json.dump(scores, f, indent=4)

    print(f'[val-{dataset_name}{suffix}] Epoch: {epoch}/{max_epochs} '
          f'mean-IoU: {miou * 100:.2f}')


if __name__ == '__main__':
    main()
