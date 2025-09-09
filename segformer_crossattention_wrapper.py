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


class FeatureDropout(nn.Module):
    def __init__(self, drop_prob=0.3, mode="branch"):
        """
        drop_prob: probability for dropping one of the branches
        mode: "branch" -> drop rgb or hybrid branch entirely
              "channel" -> mask channels within the hybrid branch
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.mode = mode

    def forward(self, feat_rgb, feat_hybrid):
        if not self.training or self.drop_prob == 0:
            return feat_rgb, feat_hybrid

        if self.mode == "branch":
            if torch.rand(1).item() < self.drop_prob:
                # drop one branch
                if torch.rand(1).item() < 0.5:
                    feat_rgb = torch.zeros_like(feat_rgb)
                else:
                    feat_hybrid = torch.zeros_like(feat_hybrid)

        elif self.mode == "channel":
            mask = (torch.rand(feat_hybrid.size(1), device=feat_hybrid.device) > self.drop_prob).float()
            feat_hybrid = feat_hybrid * mask.view(1, -1, 1, 1)

        return feat_rgb, feat_hybrid


class SegformerCrossAttentionWrapper(nn.Module):
    def __init__(self, segformer_name='nvidia/mit-b5', 
                 cross_attn_dims=[64, 128, 256, 384], 
                 downsample_factor=0.5,
                   num_classes=16,
                   mode="edge"):
        super().__init__()

        if mode == "edge" or mode == "fft" or mode == "dct":
            self.hybrid_out_ch = 1
        elif mode == "lab" or mode == "fft_dct_edge" or mode == "hsv":
            self.hybrid_out_ch = 3
        elif mode == "fft_dct":
            self.hybrid_out_ch = 2
        else:
            self.hybrid_out_ch = 1
            print("Mode unknown:", mode)

        self.mode = mode

        # rgb branch
        base_model_rgb = SegformerForSemanticSegmentation.from_pretrained(segformer_name, num_labels=num_classes)
        self.encoder_rgb = base_model_rgb.segformer.encoder
        self.decoder = base_model_rgb.decode_head

        config = self.encoder_rgb.config
        self.encoder_feat_hybrid = SegformerModel(config).encoder

        # adjust input embedding dim to number of modalities + channels
        self.encoder_feat_hybrid.patch_embeddings[0].proj = nn.Conv2d(
            in_channels=self.hybrid_out_ch,
            out_channels=self.encoder_feat_hybrid.config.hidden_sizes[0],
            kernel_size=7,
            stride=4,
            padding=3
        )


        # self.feature_dropout = FeatureDropout(drop_prob=0.3, mode="branch")

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
    


    def forward(self, image_rgb, labels=None):
        image_rgb = image_rgb.float()
        with torch.no_grad():
            if self.mode == "fft":
                feat_hybrid = self.fft_magnitude_1ch(image_rgb)
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

        # rgb hidden states
        rgb_outputs = self.encoder_rgb(image_rgb, output_hidden_states=True)
        rgb_hidden_states = rgb_outputs.hidden_states

        # edge hidden states
        feat_hybrid_outputs = self.encoder_feat_hybrid(feat_hybrid, output_hidden_states=True)
        feat_hybrid_hidden_states = feat_hybrid_outputs.hidden_states

        
        cross_features = []

        # # wrap into a list - necessary for feature dropout
        # rgb_hidden_states = list(rgb_hidden_states)
        # feat_hybrid_hidden_states = list(feat_hybrid_hidden_states)[1:]


        # for i in range(4):
        #     B, C, H, W = rgb_hidden_states[i].shape



        #     if i == 0:
        #         cross_features.append(rgb_hidden_states[i])
        #     else:

        #         # feature dropout
        #         # rgb_hidden_states[i], feat_hybrid_hidden_states[i] = self.feature_dropout(rgb_hidden_states[i], feat_hybrid_hidden_states[i])

        #         # # downsampling
        #         rgb_small = F.interpolate(rgb_hidden_states[i], scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)
        #         feat_hybrid_small = F.interpolate(feat_hybrid_hidden_states[i-1], scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)

        #         # flattening for attention
        #         rgb_flat = rgb_small.flatten(2).transpose(1, 2)  # B, N, C
        #         feat_hybrid_flat = feat_hybrid_small.flatten(2).transpose(1, 2) 

        #         # --- Aux Dropout (Feature-Dropout auf Aux-Branch) ---
        #         # if self.training:
        #         #     aux_mask = (torch.rand(B, 1, 1, device=feat_hybrid_flat.device) > 0.2).float()
        #         #     feat_hybrid_flat = feat_hybrid_flat * aux_mask  # shape: (B, N, C)

        #         # applying cross.attention
        #         attn_out = self.cross_attn_layers[i](rgb_flat, feat_hybrid_flat)

        #         fused = rgb_flat + attn_out  

        #         # upscaling via interpolation 
        #         fused = fused.transpose(1, 2).view(B, C, int(H*self.downsample_factor), int(W*self.downsample_factor))
        #         fused = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)
        #         cross_features.append(fused)


        # # decoder
        # logits = self.decoder(cross_features)


         # wrap into a list - necessary for feature dropout
        rgb_hidden_states = list(rgb_hidden_states)
        feat_hybrid_hidden_states = list(feat_hybrid_hidden_states)


        for i in range(4):
            B, C, H, W = rgb_hidden_states[i].shape

            # # downsampling
            rgb_small = F.interpolate(rgb_hidden_states[i], scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)
            feat_hybrid_small = F.interpolate(feat_hybrid_hidden_states[i], scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)

            # flattening for attention
            rgb_flat = rgb_small.flatten(2).transpose(1, 2)  # B, N, C
            feat_hybrid_flat = feat_hybrid_small.flatten(2).transpose(1, 2) 

            # --- Aux Dropout (Feature-Dropout auf Aux-Branch) ---
            # if self.training:
            #     aux_mask = (torch.rand(B, 1, 1, device=feat_hybrid_flat.device) > 0.2).float()
            #     feat_hybrid_flat = feat_hybrid_flat * aux_mask  # shape: (B, N, C)

            # applying cross.attention
            attn_out = self.cross_attn_layers[i](rgb_flat, feat_hybrid_flat)

            fused = rgb_flat + attn_out  

            # upscaling via interpolation 
            fused = fused.transpose(1, 2).view(B, C, int(H*self.downsample_factor), int(W*self.downsample_factor))
            fused = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)
            cross_features.append(fused)


        # decoder
        logits = self.decoder(cross_features)

        return logits




    

    def style_mix(self, f):
        """
        f: (B, C, H, W) Feature Map
        mischt mean/std über Batch-Dimension
        """
        B, C, H, W = f.shape
        mean = f.mean(dim=[2,3], keepdim=True)
        std  = f.std(dim=[2,3], keepdim=True)

        normed = (f - mean) / (std + 1e-6)

        # Batch-Shuffle
        perm = torch.randperm(B)
        mean2, std2 = mean[perm], std[perm]

        # Mischen mit Lam (pro Sample unterschiedlich)
        lam = torch.rand(B, 1, 1, 1, device=f.device)
        mixed_mean = lam * mean + (1 - lam) * mean2
        mixed_std  = lam * std + (1 - lam) * std2

        return normed * mixed_std + mixed_mean
