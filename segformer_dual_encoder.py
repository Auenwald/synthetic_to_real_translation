import torch.nn as nn
from transformers import SegformerModel, SegformerForSemanticSegmentation
import utils
import torch
import torch.nn.functional as F
import kornia


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

    def forward(self, x_q, x_kv, return_attn=False):
        B, N_q, _ = x_q.shape
        B, N_kv, _ = x_kv.shape

        Q = self.query(x_q).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x_kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x_kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = attn @ V
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.num_heads * self.head_dim)
        
        if return_attn:
            return self.out_proj(context), attn
        else:
            return self.out_proj(context)

class SegformerDualEncoderWrapper(nn.Module):
    def __init__(self, segformer_name='nvidia/mit-b5', 
                 cross_attn_dims=[64, 128, 256, 384], 
                 downsample_factor=0.5,
                 num_classes=16,
                 num_heads=4,
                 use_attention=False):
        super().__init__()

        

        self.use_attention = use_attention
        
        self.num_heads = num_heads
        self.downsample_factor = downsample_factor

        # --- RGB Frozen Branch ---
        base_model_rgb = SegformerForSemanticSegmentation.from_pretrained(segformer_name, num_labels=num_classes)
        self.encoder_rgb_frozen = base_model_rgb.segformer.encoder
        self.decoder = base_model_rgb.decode_head


        for param in self.encoder_rgb_frozen.parameters():
            param.requires_grad = False

        self.encoder_rgb_frozen.eval()

        config = self.encoder_rgb_frozen.config

        # --- Synth Encoder ---
        self.encoder_rgb = SegformerModel(config).encoder


        # --- Cross-Attention Layer ---
        self.cross_attn_layers = nn.ModuleList([
            MultiHeadCrossAttention(in_dim=c, attn_dim=d, num_heads=self.num_heads)
            for c, d in zip(config.hidden_sizes, cross_attn_dims)
        ])

        # --- 1x1 conv after concat fusion ---
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(c*2, c, kernel_size=1) for c in config.hidden_sizes
        ])



    def forward(self, image_rgb, labels=None, return_attention=False):
        

        # --- Encoder Outputs ---
        with torch.no_grad():
            rgb_frozen_hidden_states = list(
            self.encoder_rgb_frozen(image_rgb, output_hidden_states=True, return_dict=True).hidden_states
        )
            
        rgb_hidden_states = list(self.encoder_rgb(image_rgb, output_hidden_states=True).hidden_states)

        cross_features = []
        attentions_per_layer = []


        if self.use_attention:

            for i in range(4):
                B, C, H, W = rgb_hidden_states[i].shape

                # Downsampling
                rgb_small = F.interpolate(rgb_frozen_hidden_states[i], scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)
                feat_small = F.interpolate(rgb_hidden_states[i], scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)

                # flattening and reordering of the dimensions
                rgb_flat = rgb_small.flatten(2).transpose(1, 2)
                feat_flat = feat_small.flatten(2).transpose(1, 2)

                # Cross-Attention
                fused = self.cross_attn_layers[i](rgb_flat, feat_flat)

                # reshaping (back to HxW)
                fused = fused.transpose(1, 2).view(B, C, int(H*self.downsample_factor), int(W*self.downsample_factor))
                fused = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)

                # Concat + 1x1 Conv Fusion
                concat_feat = torch.cat([rgb_hidden_states[i], fused], dim=1)
                fused_conv = self.fusion_convs[i](concat_feat)

                cross_features.append(fused_conv)

        else:
            for i in range(4):

                concat_feat = torch.cat([rgb_hidden_states[i], rgb_frozen_hidden_states[i]], dim=1) 
                fused_conv = self.fusion_convs[i](concat_feat)
                cross_features.append(fused_conv)


        logits = self.decoder(cross_features)

        return logits, rgb_hidden_states, rgb_frozen_hidden_states
        
