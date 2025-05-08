import torch
import torch.optim
import os
import argparse
from torch.utils.data import DataLoader, Dataset
from dataset_cityscapes import *
from dataset_synthia import *
from dataset_bdd import *
import segmentation_models_pytorch as smp
import numpy as np
import utils 
from torch_ema import ExponentialMovingAverage
import pytorch_warmup as  warmup
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
# import torchmetrics
from torchmetrics.functional import jaccard_index
from segformer_pytorch import Segformer
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
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
    
    parser.add_argument('--train_print_steps', type=int, default=50, help="Specify the number of iterations between two mIoU prints during training")
    

def main():
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()

    # set all Hyperparameters
    BATCH_SIZE, LR, EPOCHS, WEIGHT_DECAY, DECAY_FACTOR = args.batch_size, args.lr, args.epochs, args.weight_decay, args.decay_factor
    WEIGHT_AVERAGING, SKIP_VAL_SOURCE = args.weight_averaging, args.skip_val_source
    AVERAGING_INTERVAL = args.averaging_interval
    SOURCE_PATH, TARGET_PATH = args.source_path, args.target_path
    SOURCE_DATASET_NAME, TARGET_DATASET_NAME = SOURCE_PATH.split("/")[-1].lower().strip(), TARGET_PATH.split("/")[-1].lower().strip()
    
    USE_LOGGING, LOG_PATH = args.use_logging, args.log_file
    PRINT_INTERVAL = args.train_print_steps
    epoch_modifier = 0

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Found the following device: {DEVICE}')

    # extend scores depending on source and target dataset
    scores[TARGET_DATASET_NAME], scores[TARGET_DATASET_NAME + "-ema"] = {}, {}
    scores[SOURCE_DATASET_NAME], scores[SOURCE_DATASET_NAME + "-ema"] = {}, {}

    # define the dataloader
    source_train_data_loader = utils.get_dataloader_from_dataset(SOURCE_PATH, SOURCE_DATASET_NAME, 'train', batch_size=BATCH_SIZE, shuffle=True)
    source_val_data_loader = utils.get_dataloader_from_dataset(SOURCE_PATH, SOURCE_DATASET_NAME, 'val', batch_size=1, shuffle=False)

    target_val_data_loader = utils.get_dataloader_from_dataset(TARGET_PATH, TARGET_DATASET_NAME, 'val', batch_size=1, shuffle=False)

    global num_classes
    num_classes = 16 if "synthia" in SOURCE_PATH else 19

    # init model
    # model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512", ignore_mismatched_sizes=True, num_labels=num_classes)
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640", ignore_mismatched_sizes=True, num_labels=num_classes)


    model = model.to(DEVICE)
    if WEIGHT_AVERAGING:
        ema = ExponentialMovingAverage(filter(lambda p: p.requires_grad, model.parameters()), decay=DECAY_FACTOR)
    else:
        ema = None

    if args.optimizer == 'SGD': 
        optim = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    elif args.optimizer == 'Adam':
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    elif args.optimizer == 'AdamW':
        optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)    


    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    for epoch in range(1 + epoch_modifier, EPOCHS + 1 + epoch_modifier):
         
        scores[TARGET_DATASET_NAME][epoch] = []
        scores[TARGET_DATASET_NAME + "-ema"][epoch] = []
        scores[SOURCE_DATASET_NAME][epoch] = []
        scores[SOURCE_DATASET_NAME + "-ema"][epoch] = []
         
        train(source_train_data_loader, model, optim, loss_fn, DEVICE, ema, PRINT_INTERVAL, AVERAGING_INTERVAL, SOURCE_DATASET_NAME)

        # if epoch % 5 == 0 and epoch > 0:
        #    torch.save(model, f'./checkpoints/_decay_{DECAY_FACTOR}_wd_{WEIGHT_DECAY}_batch_{BATCH_SIZE}_lr_{LR}_{epoch}.pth')
         
        if WEIGHT_AVERAGING:
            with ema.average_parameters():
                if not SKIP_VAL_SOURCE:
                    validate(source_val_data_loader, model, DEVICE, True, f'{SOURCE_DATASET_NAME}', epoch, EPOCHS)
                validate(target_val_data_loader, model, DEVICE, True, f'{TARGET_DATASET_NAME}', epoch, EPOCHS)
        
        if not SKIP_VAL_SOURCE:
            validate(source_val_data_loader, model, DEVICE, False, f'{SOURCE_DATASET_NAME}', epoch, EPOCHS)
        validate(target_val_data_loader, model, DEVICE, False, f'{TARGET_DATASET_NAME}', epoch, EPOCHS)
        

        write_scores_to_log_file(LOG_PATH, epoch, WEIGHT_AVERAGING, SOURCE_DATASET_NAME, TARGET_DATASET_NAME)
    
    # TODO: save model if necessary (meanIOU > bestMeanIOU)

def train(train_loader, model, optim, loss_fn, DEVICE, ema, PRINT_INTERVAL, AVERAGING_INTERVAL, SOURCE_DATASET_NAME):
    model.train()
    for i, (data, targets) in enumerate(train_loader):
            
        if data is None or targets is None:
            continue
        
        data, targets = data.to(DEVICE), targets.to(DEVICE).long()
        logits = model(data).logits

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

        if i > 0 and i % AVERAGING_INTERVAL == 0:
            if ema:
                # Update the moving average with the new parameters from the last optimizer step
                ema.update()

        if i > 0 and i % PRINT_INTERVAL == 0:
            preds = torch.argmax(logits, dim=1)
            mean_iou = jaccard_index(task='multiclass', ignore_index=255, num_classes=num_classes, preds=preds, target=targets) * 100
            print(f'[train-{SOURCE_DATASET_NAME}] Progress: {i}/{len(train_loader)}, mean-IoU: {mean_iou:.2f}, lr: {optim.param_groups[0]["lr"]}')
 
    
def validate(val_loader, model, DEVICE, applied_ema, dataset_name, epoch, max_epochs):
     model.eval()

     for idx, (data, targets) in enumerate(val_loader):
        
        if data is None or targets is None:
            continue

        data, targets = data.to(DEVICE), targets.to(DEVICE).long()

        with torch.no_grad():
             output = model(data).logits
             output = torch.nn.functional.interpolate(output, size=utils.get_image_size(dataset_name), mode='bilinear')
        
        del data

        preds = torch.argmax(output, dim=1)          
        mean_iou = jaccard_index(task='multiclass', ignore_index=255, num_classes=num_classes, preds=preds, target=targets) * 100


        # TODO: not fancy
        if applied_ema:
            logging_text = f'[val-{dataset_name}-ema] - Epoch: {epoch}/{max_epochs}'
            print(f'{logging_text} Progress: {idx}/{len(val_loader)}, mean-IoU: {mean_iou:.2f}')
            scores[dataset_name + "-ema"][epoch].append(round(mean_iou.item(), 3))
        else:
            logging_text = f'[val-{dataset_name}] - Epoch: {epoch}/{max_epochs}'
            print(f'{logging_text} Progress: {idx}/{len(val_loader)}, mean-IoU: {mean_iou:.2f}')
            scores[dataset_name][epoch].append(round(mean_iou.item(), 3))

        # logging_text = f'[val-{dataset_name}{"-ema" if applied_ema else ""}] - Epoch: {epoch}/{max_epochs}'
        # print(f'{logging_text} Progress: {idx}/{len(val_loader)}, mean-IoU: {mean_iou:.2f}')
        # scores[f'{dataset_name}{"-ema" if applied_ema else ""}'][epoch].append(round(mean_iou.item(), 3))



def write_scores_to_log_file(LOG_PATH, epoch, APPLY_AVERAGING, SOURCE_DATASET_NAME, TARGET_DATASET_NAME):

    with open(LOG_PATH, 'a') as f:
        f.write(f'{TARGET_DATASET_NAME} ' + str(epoch) + " ")
        for mean_IoU in scores[f'{TARGET_DATASET_NAME}'][epoch]:
            f.write(str(mean_IoU) + " ")
        f.write('\n')

        f.write(f'{SOURCE_DATASET_NAME} ' + str(epoch) + " ")
        for mean_IoU in scores[SOURCE_DATASET_NAME][epoch]:
            f.write(str(mean_IoU) + " ")
        f.write('\n')

        if APPLY_AVERAGING:
            f.write(f'{TARGET_DATASET_NAME}-ema ' + str(epoch) + " ")
        for mean_IoU in scores[f'{TARGET_DATASET_NAME}-ema'][epoch]:
            f.write(str(mean_IoU) + " ")
        f.write('\n')

        f.write(f'{SOURCE_DATASET_NAME}-ema ' + str(epoch) + " ")
        for mean_IoU in scores[f'{SOURCE_DATASET_NAME}-ema'][epoch]:
            f.write(str(mean_IoU) + " ")
        f.write('\n')

    
if __name__ == '__main__':
    main()