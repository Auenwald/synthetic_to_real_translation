import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # [B, C, H, W]
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]

        # Gültige Maske erstellen
        mask = None
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1)  # [B,1,H,W]

        # One-Hot Kodierung, out-of-bounds durch 0 maskiert
        targets_onehot = F.one_hot(
            torch.clamp(targets, 0, num_classes-1), num_classes=num_classes
        ).permute(0,3,1,2).float()  # [B,C,H,W]

        if mask is not None:
            probs = probs * mask
            targets_onehot = targets_onehot * mask

        intersection = torch.sum(probs * targets_onehot, dim=(0,2,3))
        union = torch.sum(probs + targets_onehot, dim=(0,2,3))

        dice = (2*intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, ignore_index=255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
