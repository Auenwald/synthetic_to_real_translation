from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models import resnet101, ResNet101_Weights
from segformer_pytorch import Segformer
from transformers import SegformerModel, SegformerConfig, SegformerForSemanticSegmentation
import torch
import torch.nn as nn
from transformers.modeling_outputs import SemanticSegmenterOutput



def get_model_by_name(name, num_classes):
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

def get_logits(model, data, data_struct=None):
    if data_struct is not None:
        output = model(data, image_struct=data_struct)
    else:
        output = model(data)

    if isinstance(output, SemanticSegmenterOutput):
        return output.logits

    if isinstance(output, dict) and 'out' in output:
        return output['out']

    if isinstance(output, torch.Tensor):
        return output

    raise ValueError(f"Unbekanntes Modell-Rückgabeformat: {type(output)}")