import torch.nn as nn
from transformers import SegformerModel, SegformerForSemanticSegmentation
import utils
import torch
import torch.nn.functional as F
import kornia
from torch_dct import dct_2d
from pytorch_wavelets import DWTForward 
import math

class WaveletExtractor(nn.Module):
    """
    Extrahiert Highpass-Wavelet-Bänder als zusätzliche Modalität
    """
    def __init__(self, wave='haar', level=1, in_ch=3, normalize=True):
        super().__init__()
        self.dwt = DWTForward(J=level, wave=wave)
        self.level = level
        self.in_ch = in_ch
        self.normalize = normalize

    def forward(self, x):
        # x: [B, C, H, W] = [2, 3, 760, 1280]
        yl, yh = self.dwt(x)   # yl: [2, 3, 380, 640], yh: Liste mit [2, 3, level, H', W']
        
        highs = []
        for level_data in yh:  # level_data: [B, C, level, H', W']
            B, C, L, H_half, W_half = level_data.shape
            level_flat = level_data.reshape(B, C * L, H_half, W_half)
            highs.append(level_flat)
        
        # Alle Level zusammenführen
        high = torch.cat(highs, dim=1)  # [B, C*L*len(yh), H', W']
        
        # Hochskalieren auf Originalgröße
        high_up = F.interpolate(high, size=(x.shape[2], x.shape[3]),
                                mode='bilinear', align_corners=False)
        
        # Normalisierung
        if self.normalize:
            mean = high_up.mean(dim=[2, 3], keepdim=True)
            std  = high_up.std(dim=[2, 3], keepdim=True) + 1e-6
            high_up = (high_up - mean) / std
        
        return high_up


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, in_dim, attn_dim=128, num_heads=4):
        super().__init__()
        assert attn_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads

        self.norm_q  = nn.LayerNorm(in_dim)
        self.norm_kv = nn.LayerNorm(in_dim)

        self.query   = nn.Linear(in_dim, attn_dim)
        self.key     = nn.Linear(in_dim, attn_dim)
        self.value   = nn.Linear(in_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, in_dim)

    def forward(self, x_q, x_kv, return_attn=False):
        x_q  = self.norm_q(x_q)
        x_kv = self.norm_kv(x_kv)
        
        B, N_q, _  = x_q.shape
        B, N_kv, _ = x_kv.shape

        Q = self.query(x_q).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x_kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x_kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        scores  = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn    = F.softmax(scores, dim=-1)
        context = attn @ V
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.num_heads * self.head_dim)

        if return_attn:
            return self.out_proj(context), attn
        else:
            return self.out_proj(context)



import math
import torch
import torch.nn as nn

def adapt_input_conv(pretrained_conv: nn.Conv2d, new_in_channels: int) -> nn.Conv2d:
    old_weight = pretrained_conv.weight.detach()
    old_out_channels, old_in_channels, kH, kW = old_weight.shape

    new_conv = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=pretrained_conv.out_channels,
        kernel_size=pretrained_conv.kernel_size,
        stride=pretrained_conv.stride,
        padding=pretrained_conv.padding,
        bias=pretrained_conv.bias is not None,
    )

    with torch.no_grad():
        if new_in_channels == old_in_channels:
            new_weight = old_weight.clone()

        elif new_in_channels == 1:
            # RGB -> 1 Kanal
            new_weight = old_weight.mean(dim=1, keepdim=True)

        elif new_in_channels == 2 and old_in_channels >= 2:
            # nimm zwei pretrained Kanäle statt zweimal Mittelwert
            new_weight = old_weight[:, :2, :, :].clone()
            new_weight *= (old_in_channels / new_in_channels)

        elif new_in_channels < old_in_channels:
            # allgemeiner Fallback
            new_weight = old_weight[:, :new_in_channels, :, :].clone()
            new_weight *= (old_in_channels / new_in_channels)

        else:
            # mehr Kanäle als vorher -> wiederholen + skalieren
            repeat_factor = math.ceil(new_in_channels / old_in_channels)
            new_weight = old_weight.repeat(1, repeat_factor, 1, 1)[:, :new_in_channels, :, :]
            new_weight *= (old_in_channels / new_in_channels)

        new_conv.weight.copy_(new_weight)

        if pretrained_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(pretrained_conv.bias.detach())

    return new_conv


class SegformerCrossAttentionWrapperV2Branched(nn.Module):
    def __init__(self, segformer_name='nvidia/mit-b5', 
                 cross_attn_dims=[64, 128, 256, 384], 
                 downsample_factor=0.5,
                 num_classes=16,
                 num_heads=4,
                 mode="edge",
                 fuse_stages=(0, 1, 2, 3)):
        super().__init__()

        # --- Hybrid Feature Kanalanzahl bestimmen ---
        if mode == "edge" or mode == "fft" or mode == "dct":
            self.hybrid_ch_in = 1
        elif mode == "lab" or mode == "fft_dct_edge" or mode == "hsv" or mode == 'multiscale_fft':
            self.hybrid_ch_in = 3
        elif mode == "fft_dct":
            self.hybrid_ch_in = 2
        elif mode == "wavelet":
            self.wavelet_level = 1
            self.hybrid_ch_in = 9
            self.wavelet_extractor = WaveletExtractor(
                wave='haar', level=self.wavelet_level, in_ch=3
            )
        else:
            self.hybrid_ch_in = 1
            print("Mode unknown:", mode)

        self.mode = mode
        self.num_heads = num_heads
        self.downsample_factor = downsample_factor
        self.fuse_stages = set(fuse_stages)

        # --- RGB Branch ---
        base_model_rgb = SegformerForSemanticSegmentation.from_pretrained(segformer_name, num_labels=num_classes)
        self.encoder_rgb = base_model_rgb.segformer.encoder
        self.decoder = base_model_rgb.decode_head

        config = self.encoder_rgb.config

        # --- Hybrid Branch ---
        # --- Hybrid Branch ---
        base_model_aux = SegformerForSemanticSegmentation.from_pretrained(
            segformer_name, num_labels=num_classes
        )
        self.encoder_feat_hybrid = base_model_aux.segformer.encoder

        old_proj = self.encoder_feat_hybrid.patch_embeddings[0].proj
        self.encoder_feat_hybrid.patch_embeddings[0].proj = adapt_input_conv(
            old_proj, self.hybrid_ch_in
        )     

        # --- Cross-Attention Layer ---
        self.cross_attn_layers = nn.ModuleList([
            MultiHeadCrossAttention(in_dim=c, attn_dim=d, num_heads=self.num_heads)
            for c, d in zip(config.hidden_sizes, cross_attn_dims)
        ])

        # --- 1x1 conv after concat fusion ---
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(c*2, c, kernel_size=1) for c in config.hidden_sizes
        ])

        # --- Stabiles Gating pro Layer ---
        # --- Stabiles Gating pro Layer ---
        self.gating_weights = nn.ParameterList([
            nn.Parameter(torch.full((c,), -2.0))
            for c in config.hidden_sizes
        ])

        # Feature Dropout optional
        # self.feature_dropout = FeatureDropout(drop_prob=0.3, mode="branch")


    def forward(self, image_rgb, image_struct=None, labels=None):
        image_rgb = image_rgb.float()
    
        # Quelle für Hybrid-Features bestimmen
        source_for_hybrid = image_struct.float() if image_struct is not None else image_rgb
        
        with torch.no_grad():
            if self.mode == "fft":
                feat_hybrid = self.fft_magnitude_1ch(source_for_hybrid)
            elif self.mode == "multiscale_fft":
                feat_hybrid = self.normalized_multiscale_fft(source_for_hybrid)
            elif self.mode == "dct":
                feat_hybrid = self.dct_map_1ch(source_for_hybrid)
            elif self.mode == "lab":
                feat_hybrid = kornia.color.rgb_to_lab(source_for_hybrid)
            elif self.mode == "hsv":
                feat_hybrid = kornia.color.rgb_to_hsv(source_for_hybrid)
            elif self.mode == "fft_dct":
                feat_hybrid = self.fft_dct_stack(source_for_hybrid)
            elif self.mode == "fft_dct_edge":
                feat_hybrid = self.fft_dct_edge_stack(source_for_hybrid)
            elif self.mode == "wavelet":
                feat_hybrid = self.wavelet_extractor(source_for_hybrid)
            else:
                feat_hybrid = utils.multiscale_scharr_edges(source_for_hybrid)

        # --- Encoder Outputs ---
        rgb_hidden_states = list(self.encoder_rgb(image_rgb, output_hidden_states=True).hidden_states)
        feat_hybrid_hidden_states = list(self.encoder_feat_hybrid(feat_hybrid, output_hidden_states=True).hidden_states)

        cross_features = []

        for i in range(4):
            B, C, H, W = rgb_hidden_states[i].shape

            if i not in self.fuse_stages:
                cross_features.append(rgb_hidden_states[i])
                continue

            # Downsampling
            rgb_small = F.interpolate(
                rgb_hidden_states[i],
                scale_factor=self.downsample_factor,
                mode='bilinear',
                align_corners=False
            )
            feat_small = F.interpolate(
                feat_hybrid_hidden_states[i],
                scale_factor=self.downsample_factor,
                mode='bilinear',
                align_corners=False
            )

            # flattening and reordering of the dimensions
            rgb_flat = rgb_small.flatten(2).transpose(1, 2)
            feat_flat = feat_small.flatten(2).transpose(1, 2)

            # Cross-Attention
            attn_out = self.cross_attn_layers[i](rgb_flat, feat_flat)

            # gating mechanism
            beta = torch.sigmoid(self.gating_weights[i]).view(1, 1, C)
            fused = rgb_flat + beta * attn_out

            # reshaping (back to HxW)
            Hs, Ws = rgb_small.shape[-2:]
            fused = fused.transpose(1, 2).reshape(B, C, Hs, Ws)
            fused = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)

            # Concat + 1x1 Conv Fusion
            concat_feat = torch.cat([rgb_hidden_states[i], fused], dim=1)
            fused_conv = self.fusion_convs[i](concat_feat)

            cross_features.append(fused_conv)
    
        logits = self.decoder(cross_features)
        return logits

    def fft_magnitude(self, image_rgb):
        # image_rgb: [B, C, H, W]
        fft = torch.fft.fft2(image_rgb)                # komplexes Spektrum
        fft_shift = torch.fft.fftshift(fft, dim=(-2,-1))  # Zero-Freq ins Zentrum
        magnitude = torch.abs(fft_shift) 
        log_mag = torch.log1p(magnitude)              
        # Normierung für Encoder
        log_mag = (log_mag - log_mag.mean()) / (log_mag.std() + 1e-6)
        return log_mag
    

    def fft_magnitude_1ch(self, image_rgb):
        """
        image_rgb: [B, C, H, W]
        return: [B, 1, H, W] log-Magnitude
        """

        # fft per channel
        fft = torch.fft.fft2(image_rgb)                   
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.abs(fft_shift)
        
        # mean over rgb channel -> 1 channel
        magnitude_1ch = magnitude.mean(dim=1, keepdim=True)
        
        # log-Skalierung
        log_mag = torch.log1p(magnitude_1ch)
        
        # normalization
        log_mag = (log_mag - log_mag.mean(dim=[2,3], keepdim=True)) / (log_mag.std(dim=[2,3], keepdim=True) + 1e-6)
    
        return log_mag  # [B, 1, H, W]
    

    def normalized_multiscale_fft(self, image_rgb, scales=[1.0, 0.5, 0.25]):
        """
        Normalized multi-scale FFT with better numerical stability
        Returns: [B, len(scales), H, W] normalized magnitudes
        """
        batch_size, channels, H, W = image_rgb.shape
        multi_scale_magnitudes = []
        
        for scale in scales:
            # Skalierung
            if scale != 1.0:
                new_H, new_W = int(H * scale), int(W * scale)
                scaled_img = F.interpolate(image_rgb, size=(new_H, new_W), 
                                        mode='bilinear', align_corners=False)
            else:
                scaled_img = image_rgb
            
            # FFT Berechnung mit stabiler Log-Transform
            fft = torch.fft.fft2(scaled_img)
            fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
            magnitude = torch.abs(fft_shift)
            magnitude_1ch = magnitude.mean(dim=1, keepdim=True)
            
            # Stabiler Logarithmus mit epsilon
            log_mag = torch.log(magnitude_1ch + 1e-10)  # Besser als log1p für FFT
            
            # Zurück auf Originalgröße
            if scale != 1.0:
                log_mag = F.interpolate(log_mag, size=(H, W), 
                                    mode='bilinear', align_corners=False)
            
            multi_scale_magnitudes.append(log_mag)
        
        # Concatenate all scales
        multiscale_result = torch.cat(multi_scale_magnitudes, dim=1)  # [B, S, H, W]
        
        # Instance Normalization pro Skala
        b, s, h, w = multiscale_result.shape
        normalized_result = torch.zeros_like(multiscale_result)
        
        for i in range(s):
            channel = multiscale_result[:, i:i+1, :, :]
            # Robust normalization with epsilon
            mean = channel.mean(dim=[2, 3], keepdim=True)
            std = channel.std(dim=[2, 3], keepdim=True) + 1e-8
            normalized_result[:, i:i+1, :, :] = (channel - mean) / std
    
        return normalized_result

    def dct_map_1ch(self, image_rgb):
        """
        image_rgb: [B, C, H, W]
        return: [B, 1, H, W] DCT-Magnitude
        """
        # dct per channel
        dct_out = dct_2d(image_rgb)

        # mean over channels → 1 channel
        dct_1ch = dct_out.mean(dim=1, keepdim=True)

        # normalization step
        dct_1ch = (dct_1ch - dct_1ch.mean(dim=[2,3], keepdim=True)) / (dct_1ch.std(dim=[2,3], keepdim=True) + 1e-6)
    
        return dct_1ch  # [B, 1, H, W]


    def fft_dct_stack(self, image_rgb):
        fft_1ch = self.fft_magnitude_1ch(image_rgb)  # [B, 1, H, W]
        dct_1ch = self.dct_map_1ch(image_rgb)        # [B, 1, H, W]
        
        # stack → 2 channels
        combined = torch.cat([fft_1ch, dct_1ch], dim=1)  # [B, 2, H, W]
        return combined

    def fft_dct_edge_stack(self, image_rgb):
        fft_1ch = self.fft_magnitude_1ch(image_rgb)  # [B, 1, H, W]
        dct_1ch = self.dct_map_1ch(image_rgb)        # [B, 1, H, W]
        edge_1ch = utils.multiscale_scharr_edges(image_rgb)

        # stack → 3 channels
        combined = torch.cat([fft_1ch, dct_1ch, edge_1ch], dim=1)  # [B, 3, H, W]
        return combined