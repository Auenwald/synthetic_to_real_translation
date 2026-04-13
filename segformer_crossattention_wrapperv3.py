import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import kornia
from torch_dct import dct_2d
from pytorch_wavelets import DWTForward
from transformers import SegformerForSemanticSegmentation

import utils


# ==============================================================================
# Hilfsfunktionen
# ==============================================================================

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
            new_weight = old_weight.mean(dim=1, keepdim=True)
        elif new_in_channels < old_in_channels:
            new_weight = old_weight[:, :new_in_channels, :, :].clone()
            new_weight *= old_in_channels / new_in_channels
        else:
            repeat_factor = math.ceil(new_in_channels / old_in_channels)
            new_weight = old_weight.repeat(1, repeat_factor, 1, 1)[:, :new_in_channels, :, :]
            new_weight *= old_in_channels / new_in_channels

        new_conv.weight.copy_(new_weight)
        if pretrained_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(pretrained_conv.bias.detach())

    return new_conv


# ==============================================================================
# WaveletExtractor (unverändert, funktioniert gut)
# ==============================================================================

class WaveletExtractor(nn.Module):
    def __init__(self, wave='haar', level=1, in_ch=3, normalize=True):
        super().__init__()
        self.dwt = DWTForward(J=level, wave=wave)
        self.normalize = normalize

    def forward(self, x):
        yl, yh = self.dwt(x)
        highs = []
        for level_data in yh:
            B, C, L, H_half, W_half = level_data.shape
            highs.append(level_data.reshape(B, C * L, H_half, W_half))
        high = torch.cat(highs, dim=1)
        high_up = F.interpolate(high, size=(x.shape[2], x.shape[3]),
                                mode='bilinear', align_corners=False)
        if self.normalize:
            mean = high_up.mean(dim=[2, 3], keepdim=True)
            std  = high_up.std(dim=[2, 3], keepdim=True) + 1e-6
            high_up = (high_up - mean) / std
        return high_up


# ==============================================================================
# NEU: AdaptiveLayerNorm
# Lernt branch-spezifische Skalierung auf top of LayerNorm.
# Gleicht unterschiedliche Feature-Statistiken zwischen RGB- und
# Hybrid-Branch aus, ohne die Vortraining-Gewichte zu zerstören.
# ==============================================================================

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        # Gamma/Beta als lernbare Residual-Skalierung (Init: Identity)
        self.scale = nn.Parameter(torch.ones(channels))
        self.bias  = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # x: [B, N, C]
        return self.norm(x) * self.scale + self.bias


# ==============================================================================
# NEU: DynamicGate
# Ersetzt den statischen gating_weight-Skalar.
# Das Gate konditioniert sich auf den aktuellen Input beider Branches,
# sodass das Modell per Sample und Layer entscheiden kann, wie stark
# der Hybrid-Branch beitragen soll.
# ==============================================================================

class DynamicGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, rgb_feat: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        """
        rgb_feat, attn_out: [B, N, C]
        Globale Poling über Sequenz-Dim → Gate-Vektor [B, C] → broadcast zurück.
        """
        gate_input = torch.cat([
            rgb_feat.mean(dim=1),   # [B, C] — globaler RGB-Kontext
            attn_out.mean(dim=1),   # [B, C] — globaler Hybrid-Kontext
        ], dim=-1)                  # [B, 2C]
        beta = self.gate_net(gate_input).unsqueeze(1)  # [B, 1, C]
        return rgb_feat + beta * attn_out


# ==============================================================================
# NEU: MultiHeadCrossAttention mit integrierter AdaptiveLayerNorm
# Kompaktere Version, norm_q und norm_kv sind jetzt AdaptiveLayerNorm
# damit branch-spezifische Statistiken korrekt normalisiert werden.
# ==============================================================================

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int = 128, num_heads: int = 4):
        super().__init__()
        assert attn_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = attn_dim // num_heads

        # AdaptiveLayerNorm statt LayerNorm: lernt branch-spez. Offset
        self.norm_q  = AdaptiveLayerNorm(in_dim)
        self.norm_kv = AdaptiveLayerNorm(in_dim)

        self.query    = nn.Linear(in_dim, attn_dim)
        self.key      = nn.Linear(in_dim, attn_dim)
        self.value    = nn.Linear(in_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, in_dim)

    def forward(self, x_q, x_kv, return_attn=False):
        x_q  = self.norm_q(x_q)
        x_kv = self.norm_kv(x_kv)

        B, N_q,  _ = x_q.shape
        B, N_kv, _ = x_kv.shape

        Q = self.query(x_q).view(B, N_q,  self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x_kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x_kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        scores  = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn    = F.softmax(scores, dim=-1)
        context = (attn @ V).transpose(1, 2).contiguous().view(B, N_q, -1)

        out = self.out_proj(context)
        if return_attn:
            return out, attn
        return out


# ==============================================================================
# Haupt-Modell: SegformerCrossAttentionV3
#
# Änderungen gegenüber V2:
#   1. AdaptiveLayerNorm in Cross-Attention (Feature-Alignment)
#   2. DynamicGate statt statischer gating_weights (input-konditioniert)
#   3. Bidirektionale Cross-Attention für Layer 2 & 3 (semantic layers)
#   4. Aux-Head auf dem tiefsten Hybrid-Feature (erzwingt semantische Repr.)
#   5. Concat+Conv-Fusion bleibt, aber nach dynamischer Fusion
# ==============================================================================

class SegformerCrossAttentionV3(nn.Module):
    def __init__(
        self,
        segformer_name: str = 'nvidia/segformer-b5-finetuned-ade-640-640',
        cross_attn_dims: list = [64, 128, 320, 512],
        downsample_factor: float = 0.5,
        num_classes: int = 16,
        num_heads: int = 4,
        mode: str = "edge",
        aux_loss_weight: float = 0.4,
        # Bidirektionale Attention nur für Layer >= bidir_from_layer
        bidir_from_layer: int = 2,
    ):
        super().__init__()

        # --- Hybrid-Kanal-Anzahl je nach Modus ---
        mode_channels = {
            "edge": 1, "fft": 1, "dct": 1,
            "lab": 3, "hsv": 3, "fft_dct_edge": 3, "multiscale_fft": 3,
            "fft_dct": 2,
            "wavelet": 9,
        }
        self.hybrid_ch_in = mode_channels.get(mode, 1)
        if mode not in mode_channels:
            print(f"[Warnung] Unbekannter Modus: {mode}, verwende 1 Kanal.")

        if mode == "wavelet":
            self.wavelet_extractor = WaveletExtractor(wave='haar', level=1, in_ch=3)

        self.mode              = mode
        self.num_heads         = num_heads
        self.downsample_factor = downsample_factor
        self.aux_loss_weight   = aux_loss_weight
        self.bidir_from_layer  = bidir_from_layer

        # --- RGB Encoder + Decoder (aus Pretrained) ---
        # ignore_mismatched_sizes=True: ADE20K hat 150 Klassen,
        # Decode-Head wird für num_classes neu initialisiert.
        base_rgb = SegformerForSemanticSegmentation.from_pretrained(
            segformer_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        self.encoder_rgb = base_rgb.segformer.encoder
        self.decoder     = base_rgb.decode_head
        config           = self.encoder_rgb.config

        # --- Hybrid Encoder (eigener patch_embeddings[0].proj) ---
        # Gleicher Pretrain-Checkpoint wie RGB für fairen Vergleich.
        base_hyb = SegformerForSemanticSegmentation.from_pretrained(
            segformer_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        self.encoder_hybrid = base_hyb.segformer.encoder
        old_proj = self.encoder_hybrid.patch_embeddings[0].proj
        self.encoder_hybrid.patch_embeddings[0].proj = adapt_input_conv(
            old_proj, self.hybrid_ch_in
        )

        hidden_sizes = config.hidden_sizes  # z.B. [64, 128, 320, 512] für mit-b5

        # --- Cross-Attention: RGB→Hybrid (Primär, alle 4 Layer) ---
        self.cross_attn_rgb_from_hyb = nn.ModuleList([
            MultiHeadCrossAttention(in_dim=c, attn_dim=d, num_heads=num_heads)
            for c, d in zip(hidden_sizes, cross_attn_dims)
        ])

        # --- NEU: Cross-Attention: Hybrid→RGB (Bidirektional, Layer 2 & 3) ---
        # Verfeinert Hybrid-Features mit RGB-Kontext bevor finale Fusion.
        # Nur für semantisch reiche Layer (2, 3) wegen VRAM/Speed.
        self.cross_attn_hyb_from_rgb = nn.ModuleList([
            MultiHeadCrossAttention(in_dim=c, attn_dim=d, num_heads=num_heads)
            if i >= bidir_from_layer else None
            for i, (c, d) in enumerate(zip(hidden_sizes, cross_attn_dims))
        ])

        # --- NEU: DynamicGate (ersetzt statische gating_weights) ---
        self.dynamic_gates = nn.ModuleList([
            DynamicGate(channels=c) for c in hidden_sizes
        ])

        # --- Concat + 1×1 Conv Fusion ---
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(c * 2, c, kernel_size=1) for c in hidden_sizes
        ])

        # --- NEU: Aux-Head auf tiefstem Hybrid-Feature ---
        # Trainings-Signal der eigentliche DG-Hebel:
        # Zwingt den Hybrid-Encoder, semantisch nützliche Features zu lernen.
        # Im Inference deaktiviert (return_aux=False).
        last_ch = hidden_sizes[-1]
        self.aux_head = nn.Sequential(
            nn.Conv2d(last_ch, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    # --------------------------------------------------------------------------
    # Feature-Extraktion (deterministisch, no_grad korrekt)
    # --------------------------------------------------------------------------

    @torch.no_grad()
    def _extract_hybrid_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "fft":
            return self._fft_1ch(x)
        elif self.mode == "multiscale_fft":
            return self._multiscale_fft(x)
        elif self.mode == "dct":
            return self._dct_1ch(x)
        elif self.mode == "lab":
            return kornia.color.rgb_to_lab(x)
        elif self.mode == "hsv":
            return kornia.color.rgb_to_hsv(x)
        elif self.mode == "fft_dct":
            return torch.cat([self._fft_1ch(x), self._dct_1ch(x)], dim=1)
        elif self.mode == "fft_dct_edge":
            return torch.cat([
                self._fft_1ch(x),
                self._dct_1ch(x),
                utils.multiscale_scharr_edges(x),
            ], dim=1)
        elif self.mode == "wavelet":
            return self.wavelet_extractor(x)
        else:
            return utils.multiscale_scharr_edges(x)

    def _fft_1ch(self, x: torch.Tensor) -> torch.Tensor:
        fft   = torch.fft.fft2(x)
        shift = torch.fft.fftshift(fft, dim=(-2, -1))
        mag   = torch.abs(shift).mean(dim=1, keepdim=True)
        log_m = torch.log1p(mag)
        return (log_m - log_m.mean(dim=[2,3], keepdim=True)) / (log_m.std(dim=[2,3], keepdim=True) + 1e-6)

    def _dct_1ch(self, x: torch.Tensor) -> torch.Tensor:
        dct = dct_2d(x).mean(dim=1, keepdim=True)
        return (dct - dct.mean(dim=[2,3], keepdim=True)) / (dct.std(dim=[2,3], keepdim=True) + 1e-6)

    def _multiscale_fft(self, x: torch.Tensor, scales=(1.0, 0.5, 0.25)) -> torch.Tensor:
        H, W = x.shape[2:]
        channels = []
        for s in scales:
            xi = F.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False) if s != 1.0 else x
            mag = torch.abs(torch.fft.fftshift(torch.fft.fft2(xi), dim=(-2,-1))).mean(1, keepdim=True)
            log_m = torch.log(mag + 1e-10)
            if s != 1.0:
                log_m = F.interpolate(log_m, size=(H, W), mode='bilinear', align_corners=False)
            log_m = (log_m - log_m.mean(dim=[2,3], keepdim=True)) / (log_m.std(dim=[2,3], keepdim=True) + 1e-8)
            channels.append(log_m)
        return torch.cat(channels, dim=1)

    # --------------------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------------------

    def forward(
        self,
        image_rgb,
        image_struct=None,
        labels=None,
        return_attention: bool = False,
        return_aux: bool = None,   # None → auto (True im Training)
    ):
        image_rgb = image_rgb.float()
        src       = image_struct.float() if image_struct is not None else image_rgb

        # Auto-Entscheid: Aux-Head nur im Training
        if return_aux is None:
            return_aux = self.training

        # --- Hybrid-Feature berechnen ---
        feat_hybrid = self._extract_hybrid_features(src)

        # --- Beide Encoder parallel vorwärts ---
        rgb_states = list(
            self.encoder_rgb(image_rgb, output_hidden_states=True).hidden_states
        )
        hyb_states = list(
            self.encoder_hybrid(feat_hybrid, output_hidden_states=True).hidden_states
        )

        cross_features       = []
        attentions_per_layer = []

        for i in range(4):
            B, C, H, W = rgb_states[i].shape

            # Downsampling für Attention (VRAM-Effizienz)
            rgb_small  = F.interpolate(rgb_states[i], scale_factor=self.downsample_factor,
                                       mode='bilinear', align_corners=False)
            hyb_small  = F.interpolate(hyb_states[i], scale_factor=self.downsample_factor,
                                       mode='bilinear', align_corners=False)

            rgb_flat  = rgb_small.flatten(2).transpose(1, 2)   # [B, N, C]
            hyb_flat  = hyb_small.flatten(2).transpose(1, 2)

            # --- NEU: Bidirektionale Attention für Layer 2 & 3 ---
            # Hybrid-Features werden mit RGB-Kontext verfeinert,
            # bevor sie als K/V in die primäre Attention eingehen.
            if i >= self.bidir_from_layer and self.cross_attn_hyb_from_rgb[i] is not None:
                hyb_refined = hyb_flat + self.cross_attn_hyb_from_rgb[i](hyb_flat, rgb_flat)
            else:
                hyb_refined = hyb_flat

            # --- Primäre Cross-Attention: RGB(Q) ← Hybrid(K/V) ---
            if return_attention:
                attn_out, attn = self.cross_attn_rgb_from_hyb[i](
                    rgb_flat, hyb_refined, return_attn=True
                )
                attentions_per_layer.append(attn)
            else:
                attn_out = self.cross_attn_rgb_from_hyb[i](rgb_flat, hyb_refined)

            # --- NEU: DynamicGate (input-konditioniert) ---
            fused_flat = self.dynamic_gates[i](rgb_flat, attn_out)  # [B, N, C]

            # Zurück zu [B, C, H_small, W_small]
            Hs, Ws = rgb_small.shape[-2:]
            fused = fused_flat.transpose(1, 2).reshape(B, C, Hs, Ws)
            fused = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)

            # --- Concat + 1×1 Conv ---
            fused_conv = self.fusion_convs[i](
                torch.cat([rgb_states[i], fused], dim=1)
            )
            cross_features.append(fused_conv)

        # --- Haupt-Decoder ---
        logits = self.decoder(cross_features)  # [B, num_classes, H/4, W/4]

        # --- NEU: Aux-Head (nur im Training) ---
        if return_aux:
            # Tiefster Hybrid-Feature-Layer als Aux-Signal
            aux_feat   = hyb_states[-1]                              # [B, C_last, H', W']
            aux_logits = self.aux_head(aux_feat)                     # [B, num_classes, H', W']
            aux_logits = F.interpolate(
                aux_logits, size=logits.shape[-2:],
                mode='bilinear', align_corners=False
            )
            if return_attention:
                return logits, aux_logits, attentions_per_layer
            return logits, aux_logits

        if return_attention:
            return logits, attentions_per_layer
        return logits

    # --------------------------------------------------------------------------
    # Convenience: Loss berechnen (optional, kann auch im Trainer bleiben)
    # --------------------------------------------------------------------------

    def compute_loss(self, logits, aux_logits, labels, ignore_index=255):
        """
        Kombinierter Loss: Haupt-CE + gewichteter Aux-CE.
        labels: [B, H, W] mit ignore_index für ungültige Pixel.
        """
        # Auf Label-Auflösung interpolieren
        logits_up = F.interpolate(logits,     size=labels.shape[-2:], mode='bilinear', align_corners=False)
        aux_up    = F.interpolate(aux_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)

        main_loss = F.cross_entropy(logits_up, labels, ignore_index=ignore_index)
        aux_loss  = F.cross_entropy(aux_up,    labels, ignore_index=ignore_index)

        return main_loss + self.aux_loss_weight * aux_loss, main_loss, aux_loss
