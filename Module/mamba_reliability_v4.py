"""Reliability-guided crowd counting model.

This version keeps the original two-stage reliability-guided idea, but makes
the guidance trainable and structurally effective:

1. Foreground and reliability cues share a supervised guide head.
2. Low-reliability regions fall back to stable average feature fusion.
3. Stage-2 relation residuals are explicitly gated by reliability.
4. Every density branch returned by the model is used by the new loss.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from Module.VMamba_B import mamba_vision_B_21k


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _stage_flag(stage: int) -> int:
    return 1 if int(stage) == 1 else 2


def _resize_guide(guide: torch.Tensor, target_hw) -> torch.Tensor:
    if guide.shape[-2:] == tuple(target_hw):
        return guide
    return F.interpolate(guide, size=target_hw, mode="bilinear", align_corners=False)


class ForegroundReliabilityGuide(nn.Module):
    """Predict foreground probability and expected prediction reliability."""

    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.context = nn.ModuleList(
            [
                nn.Conv2d(in_channels, hidden_channels, 3, padding=d, dilation=d, bias=False)
                for d in (1, 3, 5)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, 1, bias=False),
            _gn(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            _gn(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.foreground_logit = nn.Conv2d(hidden_channels, 1, 1)
        self.reliability_logit = nn.Conv2d(hidden_channels, 1, 1)

        nn.init.constant_(self.foreground_logit.bias, 0.0)
        nn.init.constant_(self.reliability_logit.bias, 0.0)

    def forward(self, feature: torch.Tensor):
        context = self.fuse(torch.cat([layer(feature) for layer in self.context], dim=1))
        foreground_logit = self.foreground_logit(context)
        reliability_logit = self.reliability_logit(context)
        return {
            "foreground_logit": foreground_logit,
            "foreground_pred": torch.sigmoid(foreground_logit),
            "reliability_logit": reliability_logit,
            "reliability_pred": torch.sigmoid(reliability_logit),
        }


class ReliabilityConditionalFusion(nn.Module):
    """Use adaptive fusion only where the model predicts that it is reliable."""

    def __init__(self, current_channels: int, previous_channels: int, out_channels: int):
        super().__init__()
        self.align_current = nn.Sequential(
            nn.Conv2d(current_channels, out_channels, 1, bias=False),
            _gn(out_channels),
        )
        self.align_previous = nn.Sequential(
            nn.Conv2d(previous_channels, out_channels, 1, bias=False),
            _gn(out_channels),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * 2 + 2, out_channels, 1, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, 2, 1),
        )
        self.post = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
        )

        # sigmoid(-2) keeps Stage-2 adaptation conservative at initialization.
        self.adaptive_strength_logit = nn.Parameter(torch.tensor(-2.0))
        self._stage = 1

    def set_stage(self, stage: int):
        self._stage = _stage_flag(stage)

    def forward(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
        foreground: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        previous = F.interpolate(previous, size=current.shape[-2:], mode="bilinear", align_corners=False)
        current_aligned = self.align_current(current)
        previous_aligned = self.align_previous(previous)
        stable = 0.5 * (current_aligned + previous_aligned)

        if self._stage == 1:
            return self.post(stable)

        foreground = _resize_guide(foreground, current.shape[-2:]).clamp(0.0, 1.0)
        reliability = _resize_guide(reliability, current.shape[-2:]).clamp(0.0, 1.0)
        logits = self.attention(
            torch.cat([current_aligned, previous_aligned, foreground, reliability], dim=1)
        )
        weights = F.softmax(logits, dim=1)
        adaptive = weights[:, :1] * current_aligned + weights[:, 1:] * previous_aligned

        # Background retains a small adaptive path; unreliable regions mostly
        # use stable fusion instead of trusting an uncalibrated attention map.
        guide = reliability * (0.25 + 0.75 * foreground)
        strength = torch.sigmoid(self.adaptive_strength_logit) * guide
        return self.post(stable + strength * (adaptive - stable))


class ReliabilityPatchRelation(nn.Module):
    def __init__(self, channels: int, token_dim: int = 128, grid=(8, 8), num_heads: int = 4):
        super().__init__()
        self.grid = tuple(grid)
        self.proj_in = nn.Conv2d(channels, token_dim, 1, bias=False)
        self.norm_in = nn.LayerNorm(token_dim)
        self.attention = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.norm_out = nn.LayerNorm(token_dim)
        self.proj_out = nn.Conv2d(token_dim, channels, 1, bias=False)
        self.residual_strength_logit = nn.Parameter(torch.tensor(-2.0))
        self._enabled = False

    def set_enabled(self, enabled: bool):
        self._enabled = bool(enabled)

    def forward(
        self,
        feature: torch.Tensor,
        foreground: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        if not self._enabled:
            return feature

        batch, _, height, width = feature.shape
        grid_h, grid_w = self.grid
        tokens_2d = F.adaptive_avg_pool2d(self.proj_in(feature), self.grid)
        token_dim = tokens_2d.shape[1]
        tokens = self.norm_in(tokens_2d.flatten(2).transpose(1, 2))
        related, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        related = self.norm_out(related)
        related = related.transpose(1, 2).reshape(batch, token_dim, grid_h, grid_w)
        related = F.interpolate(related, size=(height, width), mode="bilinear", align_corners=False)
        related = self.proj_out(related)

        foreground = _resize_guide(foreground, (height, width)).clamp(0.0, 1.0)
        reliability = _resize_guide(reliability, (height, width)).clamp(0.0, 1.0)
        gate = reliability * (0.25 + 0.75 * foreground)
        return feature + torch.sigmoid(self.residual_strength_logit) * gate * related


class ReliabilityCrossScaleRelation(nn.Module):
    def __init__(
        self,
        high_channels: int,
        low_channels: int,
        token_dim: int = 128,
        grid=(8, 8),
        num_heads: int = 4,
    ):
        super().__init__()
        self.grid = tuple(grid)
        self.query_proj = nn.Conv2d(high_channels, token_dim, 1, bias=False)
        self.key_value_proj = nn.Conv2d(low_channels, token_dim, 1, bias=False)
        self.query_norm = nn.LayerNorm(token_dim)
        self.key_value_norm = nn.LayerNorm(token_dim)
        self.attention = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(token_dim)
        self.out_proj = nn.Conv2d(token_dim, high_channels, 1, bias=False)
        self.residual_strength_logit = nn.Parameter(torch.tensor(-2.0))
        self._enabled = False

    def set_enabled(self, enabled: bool):
        self._enabled = bool(enabled)

    def forward(
        self,
        high_feature: torch.Tensor,
        low_feature: torch.Tensor,
        foreground: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        if not self._enabled:
            return high_feature

        batch, _, height, width = high_feature.shape
        grid_h, grid_w = self.grid
        query_2d = F.adaptive_avg_pool2d(self.query_proj(high_feature), self.grid)
        key_value_2d = F.adaptive_avg_pool2d(self.key_value_proj(low_feature), self.grid)
        token_dim = query_2d.shape[1]

        query = self.query_norm(query_2d.flatten(2).transpose(1, 2))
        key_value = self.key_value_norm(key_value_2d.flatten(2).transpose(1, 2))
        related, _ = self.attention(query, key_value, key_value, need_weights=False)
        related = self.out_norm(related)
        related = related.transpose(1, 2).reshape(batch, token_dim, grid_h, grid_w)
        related = F.interpolate(related, size=(height, width), mode="bilinear", align_corners=False)
        related = self.out_proj(related)

        foreground = _resize_guide(foreground, (height, width)).clamp(0.0, 1.0)
        reliability = _resize_guide(reliability, (height, width)).clamp(0.0, 1.0)
        gate = reliability * (0.25 + 0.75 * foreground)
        return high_feature + torch.sigmoid(self.residual_strength_logit) * gate * related


class DensityUncertaintyHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.density = nn.Conv2d(channels, 1, 1)
        self.log_sigma = nn.Conv2d(channels, 1, 1)
        nn.init.constant_(self.density.bias, -4.0)
        nn.init.constant_(self.log_sigma.bias, -2.0)

    def forward(self, feature: torch.Tensor):
        return F.softplus(self.density(feature)), self.log_sigma(feature)


class ReliabilityGuidedCrowdCounter(nn.Module):
    """Two-stage reliability-guided crowd counter."""

    def __init__(self, pretrained=True, patch_grid=(8, 8), relation_grid=(8, 8)):
        super().__init__()
        self.backbone = mamba_vision_B_21k(pretrained=pretrained)
        self.patch_grid = tuple(patch_grid)
        self.relation_grid = tuple(relation_grid)

        self._stage = 1
        self._built = False
        self._sigma_frozen = False

        self.guide = None
        self.fusion_deep = None
        self.fusion_shallow = None
        self.reduce = None
        self.base_shared = None
        self.regression_branch = None
        self.consistency_branch = None
        self.low_projection = None
        self.patch_relation = None
        self.cross_scale_relation = None
        self.final_head = None
        self.base_head = None
        self.low_head = None

    @staticmethod
    @torch.no_grad()
    def update_teacher_ema(teacher, student, ema=0.999):
        teacher_parameters = dict(teacher.named_parameters())
        for name, student_parameter in student.named_parameters():
            teacher_parameter = teacher_parameters[name]
            teacher_parameter.mul_(ema).add_(student_parameter, alpha=1.0 - ema)

        teacher_buffers = dict(teacher.named_buffers())
        for name, student_buffer in student.named_buffers():
            teacher_buffer = teacher_buffers[name]
            if teacher_buffer.dtype.is_floating_point:
                teacher_buffer.mul_(ema).add_(student_buffer, alpha=1.0 - ema)
            else:
                teacher_buffer.copy_(student_buffer)

    @staticmethod
    def set_teacher_mc_mode(teacher):
        """Enable stochastic depth for MC passes without updating BatchNorm."""
        teacher.eval()
        for module in teacher.modules():
            if module.__class__.__name__ == "DropPath":
                module.train()
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def set_sigma_frozen(self, frozen: bool):
        self._sigma_frozen = bool(frozen)
        self._apply_sigma_state()

    def _apply_sigma_state(self):
        if not self._built:
            return
        trainable = not self._sigma_frozen
        for head in (self.final_head, self.base_head, self.low_head):
            for parameter in head.log_sigma.parameters():
                parameter.requires_grad = trainable

    def set_stage(self, stage: int):
        self._stage = _stage_flag(stage)
        if not self._built:
            return
        self.fusion_deep.set_stage(self._stage)
        self.fusion_shallow.set_stage(self._stage)
        self.patch_relation.set_enabled(self._stage == 2)
        self.cross_scale_relation.set_enabled(self._stage == 2)
        self._apply_sigma_state()

    def _build(self, channels, reference: torch.Tensor):
        c1, c2, c3, c4 = channels
        device, dtype = reference.device, reference.dtype

        self.guide = ForegroundReliabilityGuide(c4).to(device=device, dtype=dtype)
        self.fusion_deep = ReliabilityConditionalFusion(c3, c2, c2).to(device=device, dtype=dtype)
        self.fusion_shallow = ReliabilityConditionalFusion(c2, c1, c1).to(device=device, dtype=dtype)

        self.reduce = nn.Sequential(
            nn.Conv2d(c1, 256, 1, bias=False),
            _gn(256),
            nn.SiLU(inplace=True),
        ).to(device=device, dtype=dtype)
        self.base_shared = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            _gn(256),
            nn.SiLU(inplace=True),
        ).to(device=device, dtype=dtype)
        self.regression_branch = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            _gn(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            _gn(256),
            nn.SiLU(inplace=True),
        ).to(device=device, dtype=dtype)
        self.consistency_branch = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            _gn(256),
            nn.SiLU(inplace=True),
        ).to(device=device, dtype=dtype)
        self.low_projection = nn.Sequential(
            nn.Conv2d(c3, 256, 1, bias=False),
            _gn(256),
            nn.SiLU(inplace=True),
        ).to(device=device, dtype=dtype)

        self.patch_relation = ReliabilityPatchRelation(256, grid=self.relation_grid).to(
            device=device, dtype=dtype
        )
        self.cross_scale_relation = ReliabilityCrossScaleRelation(
            256, 256, grid=self.relation_grid
        ).to(device=device, dtype=dtype)
        self.final_head = DensityUncertaintyHead(256).to(device=device, dtype=dtype)
        self.base_head = DensityUncertaintyHead(256).to(device=device, dtype=dtype)
        self.low_head = DensityUncertaintyHead(256).to(device=device, dtype=dtype)

        self._built = True
        self.set_stage(self._stage)

    @staticmethod
    def _crop_for_input(tensor: torch.Tensor, input_hw, padded_hw):
        height, width = input_hw
        padded_height, padded_width = padded_hw
        out_height, out_width = tensor.shape[-2:]
        crop_height = math.ceil(height * out_height / float(padded_height))
        crop_width = math.ceil(width * out_width / float(padded_width))
        return tensor[..., :crop_height, :crop_width]

    def forward(self, image: torch.Tensor, stage: int = 1, is_teacher: bool = False):
        del is_teacher  # Kept for compatibility; caller controls no_grad/eval mode.
        self.set_stage(stage)

        _, _, height, width = image.shape
        padded_height = ((height - 1) // 32 + 1) * 32
        padded_width = ((width - 1) // 32 + 1) * 32
        if padded_height != height or padded_width != width:
            image = F.pad(image, (0, padded_width - width, 0, padded_height - height))

        features = self.backbone(image)
        if not isinstance(features, (list, tuple)) or len(features) < 4:
            raise RuntimeError("backbone must return four feature maps")
        f1, f2, f3, f4 = features[:4]

        if not self._built:
            self._build((f1.shape[1], f2.shape[1], f3.shape[1], f4.shape[1]), f4)

        guide = self.guide(f4)
        foreground = guide["foreground_pred"]
        reliability = guide["reliability_pred"]

        fused_deep = self.fusion_deep(f3, f2, foreground, reliability)
        fused_shallow = self.fusion_shallow(fused_deep, f1, foreground, reliability)
        base_feature = self.base_shared(self.reduce(fused_shallow))
        low_feature = self.low_projection(f3)

        regression_feature = self.regression_branch(base_feature)
        regression_feature = self.cross_scale_relation(
            regression_feature, low_feature, foreground, reliability
        )
        regression_feature = self.patch_relation(regression_feature, foreground, reliability)
        consistency_feature = self.consistency_branch(regression_feature)

        density_mu, density_log_sigma = self.final_head(regression_feature)
        base_mu, base_log_sigma = self.base_head(base_feature)
        low_mu, low_log_sigma = self.low_head(low_feature)

        target_hw = density_mu.shape[-2:]
        foreground_up = _resize_guide(foreground, target_hw)
        reliability_up = _resize_guide(reliability, target_hw)
        foreground_logit_up = _resize_guide(guide["foreground_logit"], target_hw)
        reliability_logit_up = _resize_guide(guide["reliability_logit"], target_hw)

        input_hw = (height, width)
        padded_hw = (padded_height, padded_width)
        crop = lambda tensor: self._crop_for_input(tensor, input_hw, padded_hw)

        output = {
            "density_mu": crop(density_mu),
            "density_log_sigma": crop(density_log_sigma),
            "base_mu": crop(base_mu),
            "base_log_sigma": crop(base_log_sigma),
            "low_mu": crop(low_mu),
            "low_log_sigma": crop(low_log_sigma),
            "foreground_pred": crop(foreground_up).clamp(0.0, 1.0),
            "foreground_logit": crop(foreground_logit_up),
            "reliability_pred": crop(reliability_up).clamp(0.0, 1.0),
            "reliability_logit": crop(reliability_logit_up),
            "feat_reg": crop(regression_feature),
            "feat_cons": crop(consistency_feature),
            "input_hw": input_hw,
        }

        # Compatibility aliases for existing diagnostics and training utilities.
        output["mask_pred"] = output["foreground_pred"]
        output["unc_pred"] = output["reliability_pred"]
        output["count"] = output["density_mu"].sum(dim=[2, 3])
        output["pred_hw"] = output["density_mu"].shape[-2:]
        output["low_hw"] = output["low_mu"].shape[-2:]
        return output


DARPComplexCrowdReliabilityV4 = ReliabilityGuidedCrowdCounter
