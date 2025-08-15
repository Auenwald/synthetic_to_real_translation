import torch
import torch.optim
import os
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
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models import resnet101, ResNet101_Weights

# import torchmetrics
from torchmetrics.functional import jaccard_index
from segformer_pytorch import Segformer
from transformers import SegformerModel, SegformerConfig, SegformerForSemanticSegmentation
import random
from torchvision import models
from torch.cuda.amp import autocast

from transformers.modeling_outputs import SemanticSegmenterOutput
# os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import torch.nn.functional as F

def sobel_edges(images):
    # images: [B, 3, H, W] -> Graustufen + Sobel
    gray = images.mean(dim=1, keepdim=True)  # [B,1,H,W]
    kernel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=images.device).view(1,1,3,3)
    kernel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=images.device).view(1,1,3,3)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    edges = torch.sqrt(grad_x**2 + grad_y**2)
    edges = edges / (edges.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-6)  # normalize per image
    return edges

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, in_dim, attn_dim=128, num_heads=4):
        super().__init__()
        assert attn_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads

        self.query = nn.Linear(in_dim, attn_dim)
        self.key = nn.Linear(in_dim, attn_dim)
        self.value = nn.Linear(in_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, in_dim)

    def forward(self, x_q, x_kv):
        B, N_q, _ = x_q.shape
        B, N_kv, _ = x_kv.shape

        Q = self.query(x_q).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x_kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x_kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = attn @ V
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.num_heads * self.head_dim)
        return self.out_proj(context)


class SegformerCrossAttentionWrapper(nn.Module):
    def __init__(self, segformer_name='nvidia/mit-b5', cross_attn_dims=[64, 128, 256, 384], downsample_factor=0.5, num_classes=16):
        super().__init__()

        # RGB-Branch
        base_model_rgb = SegformerForSemanticSegmentation.from_pretrained(segformer_name, num_labels=num_classes)
        self.encoder_rgb = base_model_rgb.segformer.encoder
        self.decoder = base_model_rgb.decode_head

        config = self.encoder_rgb.config
        self.encoder_edge = SegformerModel(config).encoder

        # Patch embedding auf 1 Channel anpassen
        self.encoder_edge.patch_embeddings[0].proj = nn.Conv2d(
            in_channels=1,
            out_channels=self.encoder_edge.config.hidden_sizes[0],
            kernel_size=7,
            stride=4,
            padding=3
        )

        self.cross_attn_layers = nn.ModuleList([
            MultiHeadCrossAttention(in_dim=c, attn_dim=d) 
            for c, d in zip(config.hidden_sizes, cross_attn_dims)
        ])

        self.downsample_factor = downsample_factor

    def forward(self, image_rgb, labels=None):

        edge_map = sobel_edges(image_rgb)

        # RGB hidden states
        rgb_outputs = self.encoder_rgb(image_rgb, output_hidden_states=True)
        rgb_hidden_states = rgb_outputs.hidden_states

        # Edge hidden states
        edge_outputs = self.encoder_edge(edge_map, output_hidden_states=True)
        edge_hidden_states = edge_outputs.hidden_states

        cross_features = []
        for i in range(4):
            B, C, H, W = rgb_hidden_states[i].shape

            # Downsample für Cross-Attention
            rgb_small = F.interpolate(rgb_hidden_states[i], scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)
            edge_small = F.interpolate(edge_hidden_states[i], scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)

            # Flatten für Attention
            rgb_flat = rgb_small.flatten(2).transpose(1, 2)  # B, N, C
            edge_flat = edge_small.flatten(2).transpose(1, 2)

            # Cross-Attention
            fused = rgb_flat + self.cross_attn_layers[i](rgb_flat, edge_flat)

            # Zurück auf Originalgröße
            fused = fused.transpose(1, 2).view(B, C, int(H*self.downsample_factor), int(W*self.downsample_factor))
            fused = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)
            cross_features.append(fused)

        # Decoder
        logits = self.decoder(cross_features)

        return logits

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


def get_model_by_name(name):
    if "segformer" in name.lower():
        print("Using SegFormer B5")

        backbone = SegformerModel.from_pretrained('nvidia/mit-b5')
        config = SegformerConfig.from_pretrained('nvidia/mit-b5', num_labels=num_classes)
        
        model = SegformerForSemanticSegmentation(config)
        model.segformer.load_state_dict(backbone.state_dict(), strict=False)


        # pretrained on Ade20k: SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640", ignore_mismatched_sizes=True, num_labels=num_classes)

        return model
    elif "deeplab" in name.lower():
        print("Using DeeplabV3")
        backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        model = models.segmentation.deeplabv3_resnet101(weights=None, backbone=backbone) 
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

        if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
            model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        

        return model
    else:
        raise ValueError("Unknown model name!")

def get_logits(model, data):

    output = model(data)

    # HuggingFace SegFormer
    if isinstance(output, SemanticSegmenterOutput):
        return output.logits

    # torchvision DeepLabV3
    if isinstance(output, dict) and 'out' in output:
        return output['out']

    # Nur Tensor zurückgegeben?
    if isinstance(output, torch.Tensor):
        return output

    raise ValueError(f"Unbekanntes Modell-Rückgabeformat: {type(output)}")


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
    

    # model = get_model_by_name(MODEL_NAME)
    model = SegformerCrossAttentionWrapper(segformer_name='nvidia/mit-b5')


    model = model.to(DEVICE)
    if WEIGHT_AVERAGING:
        ema = ExponentialMovingAverage(filter(lambda p: p.requires_grad, model.parameters()), decay=DECAY_FACTOR)
    else:
        ema = None

    if args.optimizer.lower() == 'sgd': 
        optim = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    elif args.optimizer.lower() == 'adam':
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    elif args.optimizer.lower() == 'adamw':
        optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)    

    if WEIGHT_AVERAGING:
        ema = ExponentialMovingAverage(filter(lambda p: p.requires_grad, model.parameters()), decay=DECAY_FACTOR)
    else:
        ema = None


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
        # wrapper for handling deepLabv3 and SegFormer
        logits = get_logits(model, data)

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
             output = get_logits(model, data)
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


def write_scores_to_log_file(LOG_PATH, epoch, APPLY_AVERAGING, SOURCE_DATASET_NAME, TARGET_DATASET_NAME):

    datasets_to_log = [TARGET_DATASET_NAME, SOURCE_DATASET_NAME]

    if APPLY_AVERAGING:
        datasets_to_log += [f'{TARGET_DATASET_NAME}-ema', f'{SOURCE_DATASET_NAME}-ema']

    with open(LOG_PATH, 'a') as f:
        for dataset_key in datasets_to_log:
            scores_for_epoch = scores[dataset_key][epoch]
            line = f"{dataset_key} {epoch} " + " ".join(str(iou) for iou in scores_for_epoch) + "\n"
            f.write(line)
    
if __name__ == '__main__':
    main()