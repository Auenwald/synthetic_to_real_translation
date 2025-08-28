import torch
import argparse
from torch.utils.data import DataLoader, Dataset
from torch import nn
from datasets.dataset_cityscapes import *
from datasets.dataset_synthia import *
from datasets.dataset_synthia_style import *
from datasets.dataset_bdd import *
import numpy as np
import utils 
from torch_ema import ExponentialMovingAverage
import pytorch_warmup as  warmup
import model_utils
from segformer_crossattention_wrapper import *
import os
import json
from losses import CombinedLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import os
import math

# import torchmetrics
from torchmetrics.functional import jaccard_index
import random

# os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
SEED = 0

scores = {}
best_val_mean_IoU = 0
num_classes = 16

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def init_parser(parser):
    parser.add_argument('--source_path', default='./synthia', required=True, help='Path to the source dataset folder')
    parser.add_argument('--target_path', type=str, default='./cityscapes', help='path of the target data set')
    parser.add_argument('--model_name', type=str, default="segformer") # deeplab is also possible
    parser.add_argument('--optimizer', '-o', type=str, default='Adam', help ='Optimizer to use | SGD, Adam')
    parser.add_argument('--lr', type=float, default=1.0e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--use_logging', type=lambda x: x == 'True', default=False)
    parser.add_argument('--log_file', type=str, default='./logs/log.txt', help='path of the log file')
    parser.add_argument('--weight_averaging', type=lambda x: x == 'True', default=False)
    parser.add_argument('--averaging_interval', type=int, default=20, help="Specify the number of iterations for applying weight averaging")

    parser.add_argument('--skip_val_source', type=lambda x: x == 'True', default=False)
    parser.add_argument('--decay_factor', type=float, default=0.999, help='Specify the decay factor that is used in EMA')
    # parser.add_argument('--resume', type=bool, default=False, help='start from an existing checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help="Specify the gpu used for training")
    parser.add_argument('--use_synthia_shapes', type=lambda x: x == 'True', default=False)
    parser.add_argument('--train_print_steps', type=int, default=50, help="Specify the number of iterations between two mIoU prints during training")



def get_optimizer_and_scheduler(model, optimizer_name='adamw', lr=1e-5, total_steps=10000, warmup_steps=500, schedule=None, power=0.9, min_lr=1e-6):
    """
    optimizer_name: 'sgd', 'adam', 'adamw'
    total_steps: Gesamtzahl der Trainingsschritte
    warmup_steps: Anzahl der Schritte für linear warmup
    """
    optimizer_name = optimizer_name.lower()

    # adjust LRs depending on the optimizer
    if optimizer_name == 'sgd':
        base_lr = lr     # for pretrained RGB encoder
        hybrid_lr = lr * 10      # for Hybrid-Encoder + Cross-Attention + Decoder
        momentum = 0.9
        weight_decay = 5e-4
    elif optimizer_name in ['adam', 'adamw']:
        base_lr = lr
        hybrid_lr  = lr * 5
        momentum = None
        weight_decay = 1e-3
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")

    # params groups for different LRs
    param_groups = [
        {"params": model.encoder_rgb.parameters(), "lr": base_lr},
        {"params": model.encoder_feat_hybrid.parameters(), "lr": hybrid_lr },
        {"params": model.cross_attn_layers.parameters(), "lr": hybrid_lr },
        {"params": model.decoder.parameters(), "lr": hybrid_lr }
    ]

    # create optimizer
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay)
        print("Use SGD")
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
        print("Use Adam")
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        print("Use AdamW")

    # LR-Scheduler
    def lr_lambda(step):
        if schedule is None:
            return 1.0
        progress = float(step) / float(max(1, total_steps))
        if schedule == 'cosine':
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif schedule == 'poly':
            factor = (1 - progress) ** power
        else:
            factor = 1.0
        return max(factor, min_lr / lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()

    # set all Hyperparameters
    BATCH_SIZE, LR, EPOCHS, WEIGHT_DECAY, DECAY_FACTOR = args.batch_size, args.lr, args.epochs, args.weight_decay, args.decay_factor
    WEIGHT_AVERAGING, SKIP_VAL_SOURCE = args.weight_averaging, args.skip_val_source
    AVERAGING_INTERVAL = args.averaging_interval
    SOURCE_PATH, TARGET_PATH = args.source_path, args.target_path
    MODEL_NAME = args.model_name
    SOURCE_DATASET_NAME, TARGET_DATASET_NAME = SOURCE_PATH.split("/")[-1].lower().strip(), TARGET_PATH.split("/")[-1].lower().strip()
    GPU = args.gpu
    USE_SYNTHIA_SHAPES = args.use_synthia_shapes

    USE_LOGGING, LOG_PATH = args.use_logging, args.log_file
    PRINT_INTERVAL = args.train_print_steps
    epoch_modifier = 0

    DEVICE = f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu'
    print(f'Found the following device: {DEVICE}')

    # extend scores depending on source and target dataset
    scores[TARGET_DATASET_NAME], scores[TARGET_DATASET_NAME + "-ema"] = {}, {}
    scores[SOURCE_DATASET_NAME], scores[SOURCE_DATASET_NAME + "-ema"] = {}, {}

    # define the dataloader
    source_train_data_loader = utils.get_dataloader_from_dataset(SOURCE_PATH, SOURCE_DATASET_NAME, 'train', batch_size=BATCH_SIZE, shuffle=True, use_synthia_shapes=USE_SYNTHIA_SHAPES)
    source_val_data_loader = utils.get_dataloader_from_dataset(SOURCE_PATH, SOURCE_DATASET_NAME, 'val', batch_size=1, shuffle=False)

    target_val_data_loader = utils.get_dataloader_from_dataset(TARGET_PATH, TARGET_DATASET_NAME, 'val', batch_size=1, shuffle=False)

    global num_classes
    num_classes = 16 if "synthia" in SOURCE_PATH else 19

    # init model
    # model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512", ignore_mismatched_sizes=True, num_labels=num_classes)
    

    # model = model_utils.get_model_by_name(MODEL_NAME, num_classes)
    model = SegformerCrossAttentionWrapper(segformer_name='nvidia/mit-b5')

    model = model.to(DEVICE)
   
    optim, scheduler = get_optimizer_and_scheduler(model, optimizer_name=args.optimizer.lower(), lr=LR, total_steps=len(source_train_data_loader)*EPOCHS)


    if WEIGHT_AVERAGING:
        ema = ExponentialMovingAverage(filter(lambda p: p.requires_grad, model.parameters()), decay=DECAY_FACTOR)
    else:
        ema = None

    # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    loss_fn = CombinedLoss(ce_weight=0.5, dice_weight=0.5, ignore_index=255)

    for epoch in range(1 + epoch_modifier, EPOCHS + 1 + epoch_modifier):
         
        train(source_train_data_loader, model, optim, loss_fn, DEVICE, ema, scheduler, PRINT_INTERVAL, AVERAGING_INTERVAL, SOURCE_DATASET_NAME)

        # if epoch % 5 == 0 and epoch > 0:
        #    torch.save(model, f'./checkpoints/_decay_{DECAY_FACTOR}_wd_{WEIGHT_DECAY}_batch_{BATCH_SIZE}_lr_{LR}_{epoch}.pth')
         
        if WEIGHT_AVERAGING:
            with ema.average_parameters():
                if not SKIP_VAL_SOURCE:
                    validate(source_val_data_loader, model, DEVICE, LOG_PATH, True, f'{SOURCE_DATASET_NAME}', epoch, EPOCHS)
                validate(target_val_data_loader, model, DEVICE, LOG_PATH, True, f'{TARGET_DATASET_NAME}', epoch, EPOCHS)
        
        if not SKIP_VAL_SOURCE:
            validate(source_val_data_loader, model, DEVICE, LOG_PATH, False, f'{SOURCE_DATASET_NAME}', epoch, EPOCHS)
        validate(target_val_data_loader, model, DEVICE, LOG_PATH, False, f'{TARGET_DATASET_NAME}', epoch, EPOCHS)
        

        # write_scores_to_log_file(LOG_PATH, epoch, WEIGHT_AVERAGING, SOURCE_DATASET_NAME, TARGET_DATASET_NAME)
    
    # TODO: save model if necessary (meanIOU > bestMeanIOU)

def train(train_loader, model, optim, loss_fn, DEVICE, ema, scheduler, PRINT_INTERVAL, AVERAGING_INTERVAL, SOURCE_DATASET_NAME):
    model.train()
    for i, (data, targets) in enumerate(train_loader):
            
        if data is None or targets is None:
            continue
        
        data, targets = data.to(DEVICE), targets.to(DEVICE).long()
        # wrapper for handling deepLabv3 and SegFormer
        logits = model_utils.get_logits(model, data)

        if (data.shape[2] != targets.shape[1] or data.shape[3] != targets.shape[2]):
            print("SKIP" + str(data.shape) + str(targets.shape))
            with open("shape_log.txt", 'a') as f:
                f.write(str(data.shape) + ", " + str(targets.shape) + "\n")
            continue

        h, w = data.shape[2], data.shape[3]
        logits = torch.nn.functional.interpolate(logits, size=(h, w), mode='bilinear')
    
        loss = loss_fn(logits, targets)

        # optimizer area
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        if i > 0 and i % AVERAGING_INTERVAL == 0:
            if ema:
                # Update the moving average with the new parameters from the last optimizer step
                ema.update()

        if i > 0 and i % PRINT_INTERVAL == 0:
            preds = torch.argmax(logits, dim=1)
            mean_iou = jaccard_index(task='multiclass', ignore_index=255, num_classes=num_classes, preds=preds, target=targets) * 100
            print(f'[train-{SOURCE_DATASET_NAME}] Progress: {i}/{len(train_loader)}, mean-IoU: {mean_iou:.2f}, lr: {optim.param_groups[0]["lr"]}')
 
    
def validate(val_loader, model, DEVICE, LOG_PATH, applied_ema, dataset_name, epoch, max_epochs):
    model.eval()
    suffix = "-ema" if applied_ema else ""
    dataset_key = dataset_name + suffix

    all_preds = []
    all_targets = []
    

    for idx, (data, targets) in enumerate(val_loader):
    
        if data is None or targets is None:
            continue

        data, targets = data.to(DEVICE), targets.to(DEVICE).long()

        with torch.no_grad():
                output = model_utils.get_logits(model, data)
                output = torch.nn.functional.interpolate(output, size=utils.get_image_size(dataset_name), mode='bilinear', align_corners=False)

        preds = torch.argmax(output, dim=1)     
        all_preds.append(preds)
        all_targets.append(targets)    
        

        if idx % 10 == 0:
            print(f'[val-{dataset_name}{suffix}] - Epoch: {epoch}/{max_epochs} '
                f'Progress: {idx + 1}/{len(val_loader)}')
            
        
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)   

    miou, per_class_miou = utils.compute_mIoU_and_per_class(all_preds, all_targets, num_classes=num_classes) 
    


    LOG_JSON_PATH = LOG_PATH
    scores = {}

    if LOG_JSON_PATH and os.path.exists(LOG_JSON_PATH) and os.path.getsize(LOG_JSON_PATH) > 0:
        try:
            with open(LOG_JSON_PATH, 'r') as f:
                    scores = json.load(f)
        except json.JSONDecodeError:
            print(f"Warnung: {LOG_JSON_PATH} ist leer oder ungültig. Starte mit leerem Dict.")

    if dataset_key not in scores:
        scores[dataset_key] = {}

    # Epoch-Daten immer setzen
    scores[dataset_key][str(epoch)] = {
        "mean_iou": round(miou * 100, 3),
        "per_class_iou": {str(k): round(v*100, 3) for k, v in per_class_miou.items()}
    }

     # JSON zurückschreiben
    if LOG_JSON_PATH:
        with open(LOG_JSON_PATH, 'w') as f:
            json.dump(scores, f, indent=4)

    print(f'[val-{dataset_name}{suffix}] - Epoch: {epoch}/{max_epochs} - mean-IoU: {miou*100:.2f}')


if __name__ == '__main__':
    main()