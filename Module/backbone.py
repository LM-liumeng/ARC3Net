import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from typing import Optional, Union, List

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    raise ImportError("please install mamba_ssm: `pip install mamba-ssm` (only support Linux + CUDA)")

from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from einops import rearrange, repeat


# ==============================================================================
# 1. 基础辅助层
# ==============================================================================

class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, H, W, C] -> LN -> [B, C, H, W]
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)


class PatchEmbed(nn.Module):
    """ 4x4 Downsampling / Patch Embedding """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_down(x)


class Downsample(nn.Module):
    """ 2x2 Downsampling """

    def __init__(self, dim, keep_dim=False):
        super().__init__()
        dim_out = dim if keep_dim else 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        return self.reduction(x)


# ==============================================================================
# 2. 核心组件: Mamba Mixer (SSM) 与 Window Attention
# ==============================================================================

class MambaVisionMixer(nn.Module):
    """
    NVIDIA MambaVision 标准 Mixer
    使用 mamba_ssm.ops.selective_scan_interface.selective_scan_fn (CUDA加速)
    """

    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)

        # 初始化 dt
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # SSM 参数 A, D
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # 1D Conv
        self.conv1d_x = nn.Conv1d(
            self.d_inner // 2, self.d_inner // 2, bias=conv_bias // 2,
            kernel_size=d_conv, groups=self.d_inner // 2, **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            self.d_inner // 2, self.d_inner // 2, bias=conv_bias // 2,
            kernel_size=d_conv, groups=self.d_inner // 2, **factory_kwargs,
        )

    def forward(self, hidden_states):
        # hidden_states: [B, L, D]
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        # 1. 卷积与激活
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same',
                            groups=self.d_inner // 2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same',
                            groups=self.d_inner // 2))

        # 2. SSM 参数投影
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        A = -torch.exp(self.A_log.float())

        # 3. 核心算子：Selective Scan (CUDA)
        y = selective_scan_fn(
            x, dt, A, B, C, self.D.float(), z=None,
            delta_bias=self.dt_proj.bias.float(), delta_softplus=True
        )

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class WindowAttention(nn.Module):
    """ 标准 Window Attention """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: [B*NumWindows, WindowSize*WindowSize, C]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # PyTorch 2.0+ Scaled Dot Product Attention (FlashAttention compatible)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x


# ==============================================================================
# 3. Block 与 Layer 结构
# ==============================================================================

class Block(nn.Module):
    def __init__(self, dim, num_heads, counter, transformer_blocks, mlp_ratio=4., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # 混合架构逻辑：在特定的层使用 Transformer Attention，其他层使用 Mamba Mixer
        if counter in transformer_blocks:
            self.mixer = WindowAttention(dim, num_heads=num_heads)
        else:
            self.mixer = MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        # MLP 部分
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, downsample=True, mlp_ratio=4., drop_path=0.,
                 transformer_blocks=[]):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim=dim, counter=i, transformer_blocks=transformer_blocks,
                  num_heads=num_heads, mlp_ratio=mlp_ratio,
                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)
        ])

        self.downsample = Downsample(dim=dim) if downsample else None
        self.window_size = window_size

        # 判断本层是否包含 Transformer block (需要 window partition)
        self.has_transformer = len(transformer_blocks) > 0

    def forward(self, x):
        # x: [B, C, H, W] -> channel last
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        B, H, W, C = x.shape

        # Window Partition Logic
        shortcut = x
        if self.has_transformer:
            # Padding
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            _, Hp, Wp, _ = x.shape
            # [B, H, W, C] -> [N_windows*B, Window_Size^2, C]
            x = window_partition(x.permute(0, 3, 1, 2), self.window_size)
        else:
            # Flatten for Mamba [B, L, C]
            x = x.view(B, -1, C)

        # Blocks Forward
        for blk in self.blocks:
            x = blk(x)

        # Window Reverse Logic
        if self.has_transformer:
            x = window_reverse(x, self.window_size, Hp, Wp).permute(0, 2, 3, 1)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :].contiguous()
        else:
            x = x.view(B, H, W, C)

        # 返回 [B, C, H, W] 以供 downsample 使用
        x = x.permute(0, 3, 1, 2)

        if self.downsample is not None:
            x_down = self.downsample(x)
            return x_down  # 返回下采样后的特征
        else:
            return x


# ==============================================================================
# 4. Backbone 主类
# ==============================================================================

class MambaVisionBackbone(nn.Module):
    """
    专为 Dense Prediction (人群计数) 设计的 Backbone。
    返回特征金字塔列表 [c1, c2, c3, c4]。
    """

    def __init__(self,
                 dims=[80, 160, 320, 640],
                 depths=[3, 3, 10, 5],
                 num_heads=[2, 4, 8, 16],
                 window_size=[8, 8, 14, 7],
                 mlp_ratio=4,
                 drop_path_rate=0.2):
        super().__init__()

        # 1. Patch Embedding (4x 下采样)
        self.patch_embed = PatchEmbed(in_chans=3, in_dim=32, dim=dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()

        # 2. 构建 Stages
        for i in range(len(depths)):
            # 自动决定哪些层使用 Transformer Block (通常是后半部分)
            tf_blocks = list(range(depths[i] // 2 + 1, depths[i])) if depths[i] % 2 != 0 else list(
                range(depths[i] // 2, depths[i]))

            level = MambaVisionLayer(
                dim=dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=(i < 3),  # 前3个 stage 结束后下采样
                transformer_blocks=tf_blocks
            )
            self.levels.append(level)

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.patch_embed(x)  # Output: Stride 4 (C1)

        features = []

        # 我们需要收集 C1, C2, C3, C4
        # PatchEmbed 输出即为 C1 (Stride 4)
        features.append(x)

        # 通过 Stages
        # Level 0: Input Stride 4 -> Blocks -> Downsample -> Output Stride 8 (C2)
        # Level 1: Input Stride 8 -> Blocks -> Downsample -> Output Stride 16 (C3)
        # Level 2: Input Stride 16 -> Blocks -> Downsample -> Output Stride 32 (C4)
        # Level 3: Input Stride 32 -> Blocks -> No Downsample -> Output Stride 32

        for i, level in enumerate(self.levels):
            x = level(x)

            # Level 0 输出是 C2
            # Level 1 输出是 C3
            # Level 2 输出是 C4 (第一次出现)
            # Level 3 输出是 Refined C4

            if i < 3:
                features.append(x)
            else:
                # 替换掉最后一个 C4，使用经过 Level 3 进一步处理的特征
                features[-1] = x

        # Return list: [Stride 4, Stride 8, Stride 16, Stride 32]
        return features


if __name__ == "__main__":
    # 简单测试
    device = torch.device('cuda')
    model = MambaVisionBackbone().to(device)
    x = torch.randn(1, 3, 512, 512).to(device)
    feats = model(x)
    for i, f in enumerate(feats):
        print(f"Feature {i} shape: {f.shape}")