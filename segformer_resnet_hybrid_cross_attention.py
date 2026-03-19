import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import kornia

from transformers import SegformerForSemanticSegmentation
from pytorch_wavelets import DWTForward


# =========================================================
# Helpers
# =========================================================

def _replace_first_conv_resnet(resnet: nn.Module, in_channels: int):
    old = resnet.conv1
    new = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )

    with torch.no_grad():
        if old.in_channels == 3:
            if in_channels == 3:
                new.weight.copy_(old.weight)

            elif in_channels == 1:
                new.weight.copy_(old.weight.mean(dim=1, keepdim=True))

            elif in_channels > 3:
                rep = (in_channels + 2) // 3
                w = old.weight.repeat(1, rep, 1, 1)[:, :in_channels, :, :]
                new.weight.copy_(w * (3.0 / float(in_channels)))

            else:
                new.weight[:, :in_channels].copy_(old.weight[:, :in_channels])
                if in_channels < 3:
                    fill = old.weight.mean(dim=1, keepdim=True)
                    new.weight[:, in_channels:].copy_(
                        fill.repeat(1, 3 - in_channels, 1, 1)
                    )

            if old.bias is not None:
                new.bias.copy_(old.bias)
        else:
            nn.init.kaiming_normal_(new.weight, nonlinearity="relu")
            if new.bias is not None:
                nn.init.zeros_(new.bias)

    resnet.conv1 = new


def _hybrid_in_channels(mode: str, wavelet_level: int = 1) -> int:
    mode = mode.lower()
    if mode in {"edge", "fft", "dct"}:
        return 1
    elif mode == "fft_dct":
        return 2
    elif mode in {"fft_dct_edge", "hsv", "lab"}:
        return 3
    elif mode == "wavelet":
        return 3 * 3 * wavelet_level  # RGB * 3 highpass bands * levels
    else:
        return 1


# =========================================================
# Wavelet extractor
# =========================================================

class WaveletExtractor(nn.Module):
    """
    Extracts wavelet highpass bands as auxiliary modality.

    Input:
        x: [B, 3, H, W]

    Output:
        [B, 3 * 3 * level, H, W]  if level >= 1
    """
    def __init__(self, wave: str = "haar", level: int = 1, normalize: bool = True):
        super().__init__()
        self.dwt = DWTForward(J=level, wave=wave)
        self.level = level
        self.normalize = normalize

    def forward(self, x):
        _, yh = self.dwt(x)
        # yh is a list of length = level
        # each entry typically has shape [B, C, 3, H_l, W_l]

        highs = []
        H, W = x.shape[-2], x.shape[-1]

        for level_data in yh:
            B, C, L, H_l, W_l = level_data.shape   # L should be 3 directions
            level_flat = level_data.reshape(B, C * L, H_l, W_l)
            level_up = F.interpolate(
                level_flat,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )
            highs.append(level_up)

        high = torch.cat(highs, dim=1)  # [B, C * 3 * level, H, W]

        if self.normalize:
            mean = high.mean(dim=[2, 3], keepdim=True)
            std = high.std(dim=[2, 3], keepdim=True) + 1e-6
            high = (high - mean) / std

        return high


# =========================================================
# Cross Attention
# =========================================================

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

    def forward(self, x_q, x_kv, return_attention=False):
        """
        x_q:  [B, Nq, C]
        x_kv: [B, Nk, C]
        """
        B, Nq, _ = x_q.shape
        _, Nk, _ = x_kv.shape

        Q = self.query(x_q).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x_kv).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x_kv).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        ctx = (attn @ V).transpose(1, 2).reshape(B, Nq, self.num_heads * self.head_dim)
        out = self.out_proj(ctx)

        if return_attention:
            return out, attn
        return out


# =========================================================
# Hybrid feature extractor for RGB
# =========================================================

class HybridFeatureExtractorRGB(nn.Module):
    """
    Input: [B, 3, H, W]

    Supported modes:
        - edge
        - fft
        - dct
        - fft_dct
        - fft_dct_edge
        - hsv
        - lab
        - wavelet
    """
    def __init__(self, mode: str = "edge", wavelet: str = "haar", wavelet_level: int = 1):
        super().__init__()
        self.mode = mode.lower()
        self.wavelet_level = wavelet_level

        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32)

        self.register_buffer("sobel_kx", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_ky", sobel_y.view(1, 1, 3, 3))

        try:
            import torch_dct  # noqa: F401
            self.has_dct = True
        except Exception:
            self.has_dct = False

        self.wavelet_extractor = WaveletExtractor(
            wave=wavelet,
            level=wavelet_level,
            normalize=True
        )

    def _znorm(self, t):
        m = t.mean(dim=[2, 3], keepdim=True)
        s = t.std(dim=[2, 3], keepdim=True) + 1e-6
        return (t - m) / s

    def _to_gray(self, x):
        r = x[:, 0:1]
        g = x[:, 1:2]
        b = x[:, 2:3]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

    def _edge_1ch(self, x_gray):
        gx = F.conv2d(x_gray, self.sobel_kx, padding=1)
        gy = F.conv2d(x_gray, self.sobel_ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
        return self._znorm(mag)

    def _fft_mag_1ch(self, x_rgb):
        fft = torch.fft.fft2(x_rgb)
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
        mag = torch.abs(fft_shift).mean(dim=1, keepdim=True)
        log_mag = torch.log1p(mag)
        return self._znorm(log_mag)

    def _dct_1ch(self, x_rgb):
        if not self.has_dct:
            if not hasattr(self, "_warned_no_dct"):
                print("[HybridFeatureExtractorRGB] torch_dct not installed - falling back to FFT.", flush=True)
                self._warned_no_dct = True
            return self._fft_mag_1ch(x_rgb)

        from torch_dct import dct_2d
        d = dct_2d(x_rgb).mean(dim=1, keepdim=True)
        return self._znorm(d)

    def _hsv_3ch(self, x_rgb):
        hsv = kornia.color.rgb_to_hsv(x_rgb)
        return self._znorm(hsv)

    def _lab_3ch(self, x_rgb):
        lab = kornia.color.rgb_to_lab(x_rgb)
        return self._znorm(lab)

    def _wavelet(self, x_rgb):
        return self.wavelet_extractor(x_rgb)

    def forward(self, x):
        mode = self.mode
        x_gray = self._to_gray(x)

        if mode == "edge":
            return self._edge_1ch(x_gray)

        elif mode == "fft":
            return self._fft_mag_1ch(x)

        elif mode == "dct":
            return self._dct_1ch(x)

        elif mode == "fft_dct":
            return torch.cat([
                self._fft_mag_1ch(x),
                self._dct_1ch(x),
            ], dim=1)

        elif mode == "fft_dct_edge":
            return torch.cat([
                self._fft_mag_1ch(x),
                self._dct_1ch(x),
                self._edge_1ch(x_gray),
            ], dim=1)

        elif mode == "hsv":
            return self._hsv_3ch(x)

        elif mode == "lab":
            return self._lab_3ch(x)

        elif mode == "wavelet":
            return self._wavelet(x)

        else:
            return self._edge_1ch(x_gray)


# =========================================================
# CNN pyramid
# =========================================================

class ResNetPyramid(nn.Module):
    def __init__(self, in_channels: int, depth: int = 34, pretrained: bool = True, freeze: bool = False):
        super().__init__()

        if depth == 18:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            m = torchvision.models.resnet18(weights=weights)
        elif depth == 34:
            weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            m = torchvision.models.resnet34(weights=weights)
        else:
            raise ValueError("depth must be 18 or 34")

        _replace_first_conv_resnet(m, in_channels=in_channels)

        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        g1 = self.layer1(x)
        g2 = self.layer2(g1)
        g3 = self.layer3(g2)
        g4 = self.layer4(g3)

        return [g1, g2, g3, g4]


# =========================================================
# Projection adapter
# =========================================================

class ProjAdapter(nn.Module):
    def __init__(self, cin: int, cout: int, use_groupnorm: bool = True):
        super().__init__()

        num_groups = 32 if cout >= 32 else 1
        layers = [nn.Conv2d(cin, cout, kernel_size=1, bias=False)]

        if use_groupnorm:
            layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=cout))

        layers.append(nn.GELU())
        layers.append(nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False))

        if use_groupnorm:
            layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=cout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================================================
# Main wrapper
# =========================================================

class SegformerResNetHybridCrossAttention(nn.Module):
    """
    RGB input:
        pixel_values: [B, 3, H, W]

    Main branch:
        SegFormer encoder

    Aux branch:
        RGB -> hybrid extractor -> ResNet pyramid -> projection adapters

    Fusion:
        - optional cross-attention per stage
        - symmetric gate: lam * main + (1-lam) * cross
        - concat + 1x1 conv
        - residual add to main feature
    """
    def __init__(
        self,
        segformer_name: str = "nvidia/mit-b5",
        num_classes: int = 16,
        mode: str = "fft_dct_edge",
        attn_dims=(64, 128, 256, 384),
        num_heads: int = 4,
        resnet_depth: int = 34,
        cnn_pretrained: bool = True,
        cnn_freeze: bool = False,
        use_groupnorm_on_proj: bool = True,
        downsample_factors=(1.0, 0.5, 0.5, 0.5),
        use_attention_on_layers=(False, True, True, True),
        gates_init: float = 1.0,
        wavelet: str = "haar",
        wavelet_level: int = 1,
    ):
        super().__init__()

        self.mode = mode.lower()
        self.use_attention_on_layers = tuple(bool(x) for x in use_attention_on_layers)
        self.downsample_factors = tuple(float(x) for x in downsample_factors)

        assert len(self.use_attention_on_layers) == 4
        assert len(self.downsample_factors) == 4

        base = SegformerForSemanticSegmentation.from_pretrained(
            segformer_name,
            num_labels=num_classes
        )

        self.encoder_main = base.segformer.encoder
        self.decoder = base.decode_head
        self.config = self.encoder_main.config
        hidden_sizes = list(self.config.hidden_sizes)

        self.hybrid = HybridFeatureExtractorRGB(
            mode=self.mode,
            wavelet=wavelet,
            wavelet_level=wavelet_level
        )
        hyb_in = _hybrid_in_channels(self.mode, wavelet_level=wavelet_level)

        self.cnn = ResNetPyramid(
            in_channels=hyb_in,
            depth=resnet_depth,
            pretrained=cnn_pretrained,
            freeze=cnn_freeze,
        )

        res_ch = [64, 128, 256, 512]

        self.proj = nn.ModuleList([
            ProjAdapter(cin, cout, use_groupnorm=use_groupnorm_on_proj)
            for cin, cout in zip(res_ch, hidden_sizes)
        ])

        self.cross = nn.ModuleList([
            MultiHeadCrossAttention(in_dim=c, attn_dim=d, num_heads=num_heads)
            for c, d in zip(hidden_sizes, attn_dims)
        ])

        self.fuse = nn.ModuleList([
            nn.Conv2d(c * 2, c, kernel_size=1)
            for c in hidden_sizes
        ])

        self.gates = nn.ParameterList([
            nn.Parameter(torch.tensor([gates_init], dtype=torch.float32))
            for _ in hidden_sizes
        ])

    def forward(self, pixel_values: torch.Tensor, labels=None, return_attention: bool = False):
        x = pixel_values.float()

        with torch.no_grad():
            hfeat = self.hybrid(x).detach()

        main_h = list(self.encoder_main(x, output_hidden_states=True).hidden_states)

        cnn_feats = self.cnn(hfeat)
        hyb_h = [self.proj[i](cnn_feats[i]) for i in range(4)]

        fused = []
        attentions = []

        for i in range(4):
            main_feat = main_h[i]
            aux_feat = hyb_h[i]

            B, C, H, W = main_feat.shape
            lam = torch.sigmoid(self.gates[i])

            if not self.use_attention_on_layers[i]:
                mixed = lam * main_feat + (1.0 - lam) * aux_feat
                cat = torch.cat([main_feat, mixed], dim=1)
                mix = self.fuse[i](cat)
                out = main_feat + mix
                fused.append(out)

                if return_attention:
                    attentions.append(None)
                continue

            down = self.downsample_factors[i]
            if down != 1.0:
                main_small = F.interpolate(main_feat, scale_factor=down, mode="bilinear", align_corners=False)
                aux_small = F.interpolate(aux_feat, scale_factor=down, mode="bilinear", align_corners=False)
            else:
                main_small = main_feat
                aux_small = aux_feat

            Hs, Ws = main_small.shape[-2], main_small.shape[-1]

            main_flat = main_small.flatten(2).transpose(1, 2)  # [B, N, C]
            aux_flat = aux_small.flatten(2).transpose(1, 2)    # [B, N, C]

            if return_attention:
                cross_out, attn = self.cross[i](main_flat, aux_flat, return_attention=True)
                attentions.append(attn)
            else:
                cross_out = self.cross[i](main_flat, aux_flat)

            fused_flat = lam * main_flat + (1.0 - lam) * cross_out
            fmap = fused_flat.transpose(1, 2).reshape(B, C, Hs, Ws)

            if (Hs, Ws) != (H, W):
                fmap = F.interpolate(fmap, size=(H, W), mode="bilinear", align_corners=False)

            cat = torch.cat([main_feat, fmap], dim=1)
            mix = self.fuse[i](cat)
            out = main_feat + mix
            fused.append(out)

        logits = self.decoder(fused)

        if return_attention:
            return logits, attentions
        return logits


# =========================================================
# Convenience builder
# =========================================================

def build_segformer_resnet_hybrid(
    checkpoint: str = "nvidia/mit-b5",
    num_labels: int = 16,
    mode: str = "fft_dct_edge",
    resnet_depth: int = 34,
    cnn_pretrained: bool = True,
    cnn_freeze: bool = False,
    wavelet: str = "haar",
    wavelet_level: int = 1,
):
    return SegformerResNetHybridCrossAttention(
        segformer_name=checkpoint,
        num_classes=num_labels,
        mode=mode,
        attn_dims=(64, 128, 256, 384),
        num_heads=4,
        resnet_depth=resnet_depth,
        cnn_pretrained=cnn_pretrained,
        cnn_freeze=cnn_freeze,
        use_groupnorm_on_proj=True,
        downsample_factors=(1.0, 0.5, 0.5, 0.5),
        use_attention_on_layers=(False, True, True, True),
        gates_init=1.0,
        wavelet=wavelet,
        wavelet_level=wavelet_level,
    )