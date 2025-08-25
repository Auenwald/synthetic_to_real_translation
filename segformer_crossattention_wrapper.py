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


class HybridLayer(nn.Module):
    def __init__(self, in_ch, out_ch=4, non_linear=True):
        super().__init__()
        if non_linear:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=in_ch*2, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_ch*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_ch*2, out_channels=out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.net = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        return self.net(x)
        


class SegformerCrossAttentionWrapper(nn.Module):
    def __init__(self, segformer_name='nvidia/mit-b5', 
                 cross_attn_dims=[64, 128, 256, 384], 
                 downsample_factor=0.5,
                   num_classes=16,
                   hybrid_out_ch=4):
        super().__init__()

        # RGB-Branch
        base_model_rgb = SegformerForSemanticSegmentation.from_pretrained(segformer_name, num_labels=num_classes)
        self.encoder_rgb = base_model_rgb.segformer.encoder
        self.decoder = base_model_rgb.decode_head

        config = self.encoder_rgb.config
        self.encoder_edge = SegformerModel(config).encoder

        # add hybrid layer
        self.hybrid_layer = HybridLayer(in_ch=3, out_ch=hybrid_out_ch, non_linear=True)

        # Patch embedding auf hybrid_out_ch Channel anpassen
        self.encoder_edge.patch_embeddings[0].proj = nn.Conv2d(
            in_channels=hybrid_out_ch,
            out_channels=self.encoder_edge.config.hidden_sizes[0],
            kernel_size=7,
            stride=4,
            padding=3
        )

        

        self.cross_attn_layers = nn.ModuleList([
            MultiHeadCrossAttention(in_dim=c, attn_dim=d) 
            for c, d in zip(config.hidden_sizes, cross_attn_dims)
        ])

        self.alpha_logits = nn.ParameterList([
            nn.Parameter(torch.tensor(-0.2, dtype=torch.float32))
            for _ in range(len(config.hidden_sizes))
        ])

        self.downsample_factor = downsample_factor


    def fft_magnitude(self, image_rgb):
        # image_rgb: [B, C, H, W]
        fft = torch.fft.fft2(image_rgb)                # komplexes Spektrum
        fft_shift = torch.fft.fftshift(fft, dim=(-2,-1))  # Zero-Freq ins Zentrum
        magnitude = torch.abs(fft_shift)               # Betrag
        log_mag = torch.log1p(magnitude)               # log-Skalierung
        # Normierung für Encoder
        log_mag = (log_mag - log_mag.mean()) / (log_mag.std() + 1e-6)
        return log_mag
    

    def fft_magnitude_1ch(self, image_rgb):
        """
        image_rgb: [B, C, H, W]
        return: [B, 1, H, W] log-Magnitude
        """

        # FFT pro Kanal
        fft = torch.fft.fft2(image_rgb)                   
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.abs(fft_shift)
        
        # Mittelwert über RGB-Kanäle → 1 Channel
        magnitude_1ch = magnitude.mean(dim=1, keepdim=True)
        
        # log-Skalierung
        log_mag = torch.log1p(magnitude_1ch)
        
        # Normalisierung
        log_mag = (log_mag - log_mag.mean(dim=[2,3], keepdim=True)) / (log_mag.std(dim=[2,3], keepdim=True) + 1e-6)
    
        return log_mag  # [B, 1, H, W]

    def dct_map_1ch(self, image_rgb):
        """
        image_rgb: [B, C, H, W]
        return: [B, 1, H, W] DCT-Magnitude
        """
        # DCT pro Kanal
        dct_out = dct_2d(image_rgb)

        # Mittelwert über Kanäle → 1 Channel
        dct_1ch = dct_out.mean(dim=1, keepdim=True)

        # Normalisierung
        dct_1ch = (dct_1ch - dct_1ch.mean(dim=[2,3], keepdim=True)) / (dct_1ch.std(dim=[2,3], keepdim=True) + 1e-6)
    
        return dct_1ch  # [B, 1, H, W]


    def fft_dct_stack(self, image_rgb):
        fft_1ch = self.fft_magnitude_1ch(image_rgb)  # [B, 1, H, W]
        dct_1ch = self.dct_map_1ch(image_rgb)        # [B, 1, H, W]
        
        # Stack → 2 Kanäle
        combined = torch.cat([fft_1ch, dct_1ch], dim=1)  # [B, 2, H, W]
        return combined

    def fft_dct_edge_stack(self, image_rgb):
        fft_1ch = self.fft_magnitude_1ch(image_rgb)  # [B, 1, H, W]
        dct_1ch = self.dct_map_1ch(image_rgb)        # [B, 1, H, W]
        edge_1ch = utils.multiscale_scharr_edges(image_rgb)

        # Stack → 3 Kanäle
        combined = torch.cat([fft_1ch, dct_1ch, edge_1ch], dim=1)  # [B, 2, H, W]
        return combined


    def forward(self, image_rgb, labels=None, mode="fft_dct_edge_hybrid"):

        # image_lab = kornia.color.rgb_to_lab(image_rgb)
        # edge_map = utils.multiscale_scharr_edges(image_rgb)

        if mode == "fft":
            edge_map = self.fft_magnitude_1ch(image_rgb)
        elif mode == "dct":
            edge_map = self.dct_map_1ch(image_rgb)
        elif mode == "fft_dct":
            edge_map = self.fft_dct_stack(image_rgb)
        elif mode == "fft_dct_edge":
            edge_map = self.fft_dct_edge_stack(image_rgb)
        elif mode == "fft_dct_edge_hybrid":
            combined = self.fft_dct_edge_stack(image_rgb)
            edge_map = self.hybrid_layer(combined)
        else:
            edge_map = utils.multiscale_scharr_edges(image_rgb)

        # rGB hidden states
        rgb_outputs = self.encoder_rgb(image_rgb, output_hidden_states=True)
        rgb_hidden_states = rgb_outputs.hidden_states

        # edge hidden states
        edge_outputs = self.encoder_edge(edge_map, output_hidden_states=True)
        edge_hidden_states = edge_outputs.hidden_states

        cross_features = []
        for i in range(4):
            B, C, H, W = rgb_hidden_states[i].shape

            # # downsampling
            rgb_small = F.interpolate(rgb_hidden_states[i], scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)
            edge_small = F.interpolate(edge_hidden_states[i], scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)

            # flattening for attention
            rgb_flat = rgb_small.flatten(2).transpose(1, 2)  # B, N, C
            edge_flat = edge_small.flatten(2).transpose(1, 2)

            # applying cross.attention
            attn_out = self.cross_attn_layers[i](rgb_flat, edge_flat)
            alpha = torch.sigmoid(self.alpha_logits[i])
            fused = rgb_flat + alpha * attn_out

            # upscaling via interpolation 
            fused = fused.transpose(1, 2).view(B, C, int(H*self.downsample_factor), int(W*self.downsample_factor))
            fused = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)
            cross_features.append(fused)

        # Decoder
        logits = self.decoder(cross_features)

        return logits
