import torch
from transformers import SegformerModel, SegformerConfig, SegformerForSemanticSegmentation
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models import resnet101, ResNet101_Weights
from torch import nn
num_classes = 16
from transformers.modeling_outputs import SemanticSegmenterOutput
# model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640", ignore_mismatched_sizes=True, num_labels=num_classes)

# print(model)


backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
model = torchvision.models.segmentation.deeplabv3_resnet101(weights=None, backbone=backbone) 
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

print(model)