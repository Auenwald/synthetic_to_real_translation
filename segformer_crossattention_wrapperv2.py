import torch.nn as nn
from transformers import SegformerModel, SegformerForSemanticSegmentation
import utils
import torch
import torch.nn.functional as F
import kornia
from torch_dct import dct_2d


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

class SegformerCrossAttentionWrapperV2(nn.Module):
    def __init__(self, segformer_name='nvidia/mit-b5', 
                 cross_attn_dims=[64, 128, 256, 384], 
                 downsample_factor=0.5,
                 num_classes=16,
                 num_heads=4,
                 mode="edge"):
        super().__init__()

        # --- Hybrid Feature Kanalanzahl bestimmen ---
        if mode in ["edge", "fft", "dct"]:
            self.hybrid_out_ch = 1
        elif mode in ["lab", "fft_dct_edge", "hsv", "multiscale_fft"]:
            self.hybrid_out_ch = 3
        elif mode == "fft_dct":
            self.hybrid_out_ch = 2
        else:
            self.hybrid_out_ch = 1
            print("Mode unknown:", mode)

        self.mode = mode
        self.num_heads = num_heads
        self.downsample_factor = downsample_factor

        # --- RGB Branch ---
        base_model_rgb = SegformerForSemanticSegmentation.from_pretrained(segformer_name, num_labels=num_classes)
        self.encoder_rgb = base_model_rgb.segformer.encoder
        self.decoder = base_model_rgb.decode_head

        config = self.encoder_rgb.config

        # --- Hybrid Branch ---
        self.encoder_feat_hybrid = SegformerModel(config).encoder
        self.encoder_feat_hybrid.patch_embeddings[0].proj = nn.Conv2d(
            in_channels=self.hybrid_out_ch,
            out_channels=self.encoder_feat_hybrid.config.hidden_sizes[0],
            kernel_size=7,
            stride=4,
            padding=3
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
        self.gating_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(1))  # Startwert 0 → Sigmoid ≈ 0.5
            for _ in range(len(config.hidden_sizes))
        ])

        # Feature Dropout optional
        # self.feature_dropout = FeatureDropout(drop_prob=0.3, mode="branch")


    def forward(self, image_rgb, labels=None, return_attention=False):
        image_rgb = image_rgb.float()

        # --- Hybrid Features ---
        with torch.no_grad():
            if self.mode == "fft":
                feat_hybrid = self.fft_magnitude_1ch(image_rgb)
            elif self.mode == "multiscale_fft":
                feat_hybrid = self.normalized_multiscale_fft(image_rgb)
            elif self.mode == "dct":
                feat_hybrid = self.dct_map_1ch(image_rgb)
            elif self.mode == "lab":
                feat_hybrid = kornia.color.rgb_to_lab(image_rgb)
            elif self.mode == "hsv":
                feat_hybrid = kornia.color.rgb_to_hsv(image_rgb)
            elif self.mode == "fft_dct":
                feat_hybrid = self.fft_dct_stack(image_rgb)
            elif self.mode == "fft_dct_edge":
                feat_hybrid = self.fft_dct_edge_stack(image_rgb)
            else:
                feat_hybrid = utils.multiscale_scharr_edges(image_rgb)

        feat_hybrid = feat_hybrid.detach()

        # --- Encoder Outputs ---
        rgb_hidden_states = list(self.encoder_rgb(image_rgb, output_hidden_states=True).hidden_states)
        feat_hybrid_hidden_states = list(self.encoder_feat_hybrid(feat_hybrid, output_hidden_states=True).hidden_states)

        cross_features = []
        attentions_per_layer = []

        for i in range(4):
            B, C, H, W = rgb_hidden_states[i].shape

            # Downsampling
            rgb_small = F.interpolate(rgb_hidden_states[i], scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)
            feat_small = F.interpolate(feat_hybrid_hidden_states[i], scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)

            # Flatten für Cross-Attention
            rgb_flat = rgb_small.flatten(2).transpose(1, 2)
            feat_flat = feat_small.flatten(2).transpose(1, 2)

            # Cross-Attention
            if return_attention:
                attn_out, attn = self.cross_attn_layers[i](rgb_flat, feat_flat, return_attn=True)
                attentions_per_layer.append(attn)
            else:
                attn_out = self.cross_attn_layers[i](rgb_flat, feat_flat)

            fused = rgb_flat + attn_out

            # Gating
            alpha = torch.sigmoid(self.gating_weights[i])
            fused = alpha * rgb_flat + (1 - alpha) * fused

            # Reshape zurück auf HxW
            fused = fused.transpose(1, 2).view(B, C, int(H*self.downsample_factor), int(W*self.downsample_factor))
            fused = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)

            # Concat + 1x1 Conv Fusion
            concat_feat = torch.cat([rgb_hidden_states[i], fused], dim=1)
            fused_conv = self.fusion_convs[i](concat_feat)

            cross_features.append(fused_conv)

        logits = self.decoder(cross_features)

        if return_attention:
            return logits, attentions_per_layer
        else:
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