# model.py
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Module.VMamba_B import mamba_vision_B_21k


# -------------------------
# small helpers
# -------------------------
def _gn(c: int, max_groups: int = 32) -> nn.GroupNorm:
    g = min(max_groups, c)
    while g > 1 and (c % g) != 0:
        g -= 1
    return nn.GroupNorm(g, c)


def _to_stage_flag(stage: int) -> int:
    return 1 if int(stage) == 1 else 2


# =========================================================
# Foreground mask generator (Stage1 supervised by GT-mask, Stage2 frozen)
# =========================================================
class ForegroundMaskGenerator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.dilated = nn.ModuleList([
            nn.Conv2d(in_channels, 128, 3, padding=1, dilation=1),
            nn.Conv2d(in_channels, 128, 3, padding=3, dilation=3),
            nn.Conv2d(in_channels, 128, 3, padding=5, dilation=5),
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(128 * 3, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [m(x) for m in self.dilated]
        y = torch.cat(feats, dim=1)
        return torch.sigmoid(self.fuse(y))


# =========================================================
# Uncertainty cue generator (predict a continuous confidence map in [0,1])
# =========================================================
class UncertaintyCueGenerator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            _gn(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            _gn(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1, bias=True),
        )
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


# =========================================================
# Conditional ABF cross-layer attention fusion
# =========================================================
class ConditionalABF(nn.Module):
    def __init__(self, c_cur: int, c_prev: int, out_c: int):
        super().__init__()
        self.align_cur = nn.Sequential(
            nn.Conv2d(c_cur, out_c, 1, bias=False),
            _gn(out_c),
        )
        self.align_prev = nn.Sequential(
            nn.Conv2d(c_prev, out_c, 1, bias=False),
            _gn(out_c),
        )

        self.attn = nn.Sequential(
            nn.Conv2d(out_c * 2 + 2, out_c, 1, bias=False),
            _gn(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, 2, 1, bias=True),
        )

        self.post = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            _gn(out_c),
            nn.ReLU(inplace=True),
        )

        self.alpha = nn.Parameter(torch.tensor(0.0))

        self._stage = 1
        self._freeze_attn = True

    def set_stage(self, stage: int):
        stage = _to_stage_flag(stage)
        self._stage = stage

        if stage == 1:
            self._freeze_attn = True
            with torch.no_grad():
                self.alpha.fill_(0.0)
        else:
            self._freeze_attn = False
            if float(self.alpha.detach().cpu()) == 0.0:
                with torch.no_grad():
                    self.alpha.fill_(0.05)

        for p in self.attn.parameters():
            p.requires_grad = (not self._freeze_attn)
        self.alpha.requires_grad = (not self._freeze_attn)

    def forward(self, x_cur: torch.Tensor, x_prev: torch.Tensor, mask: torch.Tensor, unc: torch.Tensor) -> torch.Tensor:
        x_prev = F.interpolate(x_prev, size=x_cur.shape[-2:], mode="bilinear", align_corners=False)

        a = self.align_cur(x_cur)
        b = self.align_prev(x_prev)

        if mask.shape[-2:] != x_cur.shape[-2:]:
            mask = F.interpolate(mask, size=x_cur.shape[-2:], mode="bilinear", align_corners=False)
        if unc.shape[-2:] != x_cur.shape[-2:]:
            unc = F.interpolate(unc, size=x_cur.shape[-2:], mode="bilinear", align_corners=False)

        mask = mask.clamp(0.0, 1.0)
        unc = unc.clamp(0.0, 1.0)

        logits = self.attn(torch.cat([a, b, mask, unc], dim=1))
        logits = logits * self.alpha

        w = F.softmax(logits, dim=1)  # [B,2,H,W]
        y = w[:, 0:1] * a + w[:, 1:2] * b
        return self.post(y)


# =========================================================
# Patch-token global relation module (Stage2 only)
# =========================================================
class PatchTokenRelation(nn.Module):
    def __init__(self, in_c: int, token_dim: int = 128, grid=(16, 16), num_heads: int = 4):
        super().__init__()
        self.grid = tuple(grid)
        self.proj_in = nn.Conv2d(in_c, token_dim, 1, bias=False)
        self.norm = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads, batch_first=True)
        self.proj_out = nn.Conv2d(token_dim, in_c, 1, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.0))
        self._enabled = False

    def set_enabled(self, enabled: bool):
        self._enabled = bool(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._enabled:
            return x

        B, C, H, W = x.shape
        gh, gw = self.grid

        t = self.proj_in(x)
        t_pool = F.adaptive_avg_pool2d(t, (gh, gw))
        td = t_pool.shape[1]

        tokens = t_pool.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)

        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        attn_out = self.norm(attn_out)

        feat = attn_out.transpose(1, 2).reshape(B, td, gh, gw)
        feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
        feat = self.proj_out(feat)

        return x + self.scale * feat


# =========================================================
# Cross-scale relation (High tokens attend to Low tokens)
# =========================================================
class CrossScaleRelation(nn.Module):
    def __init__(self, high_c: int, low_c: int, token_dim: int = 128, grid_high=(16, 16), grid_low=(16, 16), num_heads: int = 4):
        super().__init__()
        self.grid_high = tuple(grid_high)
        self.grid_low = tuple(grid_low)

        self.q_proj = nn.Conv2d(high_c, token_dim, 1, bias=False)
        self.kv_proj = nn.Conv2d(low_c, token_dim, 1, bias=False)

        self.q_norm = nn.LayerNorm(token_dim)
        self.kv_norm = nn.LayerNorm(token_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads, batch_first=True)

        self.out_proj = nn.Conv2d(token_dim, high_c, 1, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.0))

        self._enabled = False

    def set_enabled(self, enabled: bool):
        self._enabled = bool(enabled)

    def forward(self, x_high: torch.Tensor, x_low: torch.Tensor) -> torch.Tensor:
        if not self._enabled:
            return x_high

        B, C, H, W = x_high.shape
        gh, gw = self.grid_high
        glh, glw = self.grid_low

        q = self.q_proj(x_high)
        kv = self.kv_proj(x_low)

        q_pool = F.adaptive_avg_pool2d(q, (gh, gw))
        kv_pool = F.adaptive_avg_pool2d(kv, (glh, glw))

        td = q_pool.shape[1]

        q_tok = q_pool.flatten(2).transpose(1, 2)
        kv_tok = kv_pool.flatten(2).transpose(1, 2)

        q_tok = self.q_norm(q_tok)
        kv_tok = self.kv_norm(kv_tok)

        out_tok, _ = self.cross_attn(q_tok, kv_tok, kv_tok, need_weights=False)
        out_tok = self.q_norm(out_tok)

        out_feat = out_tok.transpose(1, 2).reshape(B, td, gh, gw)
        out_feat = F.interpolate(out_feat, size=(H, W), mode="bilinear", align_corners=False)
        out_feat = self.out_proj(out_feat)

        return x_high + self.scale * out_feat


# =========================================================
# Heads
# =========================================================
class DensitySigmaHead(nn.Module):
    def __init__(self, in_c: int):
        super().__init__()
        self.mu = nn.Conv2d(in_c, 1, 1)
        self.log_sigma = nn.Conv2d(in_c, 1, 1)
        nn.init.constant_(self.mu.bias, -4.0)
        nn.init.constant_(self.log_sigma.bias, -2.0)

    def forward(self, x: torch.Tensor):
        mu = F.softplus(self.mu(x))
        log_sigma = self.log_sigma(x)
        return mu, log_sigma


class ResidualPosNegHead(nn.Module):
    def __init__(self, in_c: int):
        super().__init__()
        self.pos = nn.Conv2d(in_c, 1, 1)
        self.neg = nn.Conv2d(in_c, 1, 1)
        nn.init.constant_(self.pos.bias, -4.0)
        nn.init.constant_(self.neg.bias, -4.0)

    def forward(self, x: torch.Tensor):
        pos = F.softplus(self.pos(x))
        neg = F.softplus(self.neg(x))
        return pos, neg


# =========================================================
# Main model
# =========================================================
class DARPComplexCrowd(nn.Module):
    """
    Stage-aware model:
      - Stage1:
          ConditionalABF attention frozen (alpha=0 => stable avg fusion)
          Token relation OFF, Cross-scale relation OFF
          sigma learnable
          mask_gen + unc_gen learnable
      - Stage2:
          ConditionalABF learnable
          Token relation ON (CONS branch)
          Cross-scale relation ON (REG branch: high attends to low)
          sigma: controllable by set_sigma_frozen()
          mask_gen + unc_gen frozen
    """

    def __init__(self, pretrained=True, patch_grid=(16, 16), relation_grid=(16, 16)):
        super().__init__()
        self.backbone = mamba_vision_B_21k(pretrained=pretrained)

        self._built = False
        self._stage = 1

        self.patch_grid = tuple(patch_grid)
        self.relation_grid = tuple(relation_grid)

        # NEW: sigma freeze flag (effective in stage2; stage1 always trainable)
        self._sigma_frozen = False

        # placeholders
        self.mask_gen = None
        self.unc_gen = None

        self.abf2 = None
        self.abf1 = None

        self.reduce_p1 = None
        self.base_shared = None

        self.reg_branch = None
        self.cons_branch = None

        self.token_relation = None
        self.cross_scale = None

        self.head_final = None
        self.low_proj = None
        self.head_low = None
        self.head_res = None

    # ------------------- teacher utils -------------------
    @staticmethod
    @torch.no_grad()
    def update_teacher_ema(teacher, student, ema=0.999):
        st = student.state_dict()
        th = teacher.state_dict()
        for k in th.keys():
            if th[k].dtype.is_floating_point:
                th[k].data.mul_(ema).add_(st[k].data, alpha=1.0 - ema)
        teacher.load_state_dict(th, strict=True)

    # ------------------- sigma control -------------------
    def set_sigma_frozen(self, frozen: bool):
        """
        Control whether log_sigma convs are trainable.
        - Stage1: always trainable (this method still sets flag but will be overridden by stage logic)
        - Stage2: respects this flag
        """
        self._sigma_frozen = bool(frozen)
        if not self._built:
            return

        # apply according to current stage
        self._apply_sigma_freeze()

    def _apply_sigma_freeze(self):
        if not self._built:
            return

        # Stage1: always trainable
        if self._stage == 1:
            for p in self.head_final.log_sigma.parameters():
                p.requires_grad = True
            for p in self.head_low.log_sigma.parameters():
                p.requires_grad = True
            return

        # Stage2: controlled by _sigma_frozen
        req = (not self._sigma_frozen)
        for p in self.head_final.log_sigma.parameters():
            p.requires_grad = req
        for p in self.head_low.log_sigma.parameters():
            p.requires_grad = req

    # ------------------- stage control -------------------
    def set_stage(self, stage: int):
        stage = _to_stage_flag(stage)
        self._stage = stage
        if not self._built:
            return

        self.abf2.set_stage(stage)
        self.abf1.set_stage(stage)

        self.token_relation.set_enabled(stage == 2)
        self.cross_scale.set_enabled(stage == 2)

        # Stage2 freeze mask/unc generators
        if stage == 2:
            for p in self.mask_gen.parameters():
                p.requires_grad = False
            for p in self.unc_gen.parameters():
                p.requires_grad = False
            self.mask_gen.eval()
            self.unc_gen.eval()
        else:
            for p in self.mask_gen.parameters():
                p.requires_grad = True
            for p in self.unc_gen.parameters():
                p.requires_grad = True
            self.mask_gen.train()
            self.unc_gen.train()

        # NEW: sigma freeze controlled (stage1 always trainable)
        self._apply_sigma_freeze()

    # ------------------- build -------------------
    def _build(self, c1, c2, c3, c4, ref: torch.Tensor):
        dev, dt = ref.device, ref.dtype

        self.mask_gen = ForegroundMaskGenerator(c4).to(device=dev, dtype=dt)
        self.unc_gen = UncertaintyCueGenerator(c4).to(device=dev, dtype=dt)

        self.abf2 = ConditionalABF(c_cur=c3, c_prev=c2, out_c=c2).to(device=dev, dtype=dt)
        self.abf1 = ConditionalABF(c_cur=c2, c_prev=c1, out_c=c1).to(device=dev, dtype=dt)

        self.reduce_p1 = nn.Sequential(
            nn.Conv2d(c1, 256, 1, bias=False),
            _gn(256),
            nn.ReLU(inplace=True),
        ).to(device=dev, dtype=dt)

        self.base_shared = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            _gn(256),
            nn.ReLU(inplace=True),
        ).to(device=dev, dtype=dt)

        self.reg_branch = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            _gn(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            _gn(256),
            nn.ReLU(inplace=True),
        ).to(device=dev, dtype=dt)

        self.cons_branch = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            _gn(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            _gn(256),
            nn.ReLU(inplace=True),
        ).to(device=dev, dtype=dt)

        self.token_relation = PatchTokenRelation(
            in_c=256, token_dim=128, grid=self.relation_grid, num_heads=4
        ).to(device=dev, dtype=dt)

        self.low_proj = nn.Sequential(
            nn.Conv2d(c3, 256, 1, bias=False),
            _gn(256),
            nn.ReLU(inplace=True),
        ).to(device=dev, dtype=dt)

        self.cross_scale = CrossScaleRelation(
            high_c=256,
            low_c=256,
            token_dim=128,
            grid_high=self.relation_grid,
            grid_low=self.relation_grid,
            num_heads=4,
        ).to(device=dev, dtype=dt)

        self.head_final = DensitySigmaHead(256).to(device=dev, dtype=dt)
        self.head_low = DensitySigmaHead(256).to(device=dev, dtype=dt)
        self.head_res = ResidualPosNegHead(256).to(device=dev, dtype=dt)

        self._built = True
        self.set_stage(self._stage)

    # ------------------- forward -------------------
    def forward(self, x: torch.Tensor, stage: int = 1, is_teacher: bool = False):
        self.set_stage(stage)

        B, C, H, W = x.shape
        H_pad = ((H - 1) // 32 + 1) * 32
        W_pad = ((W - 1) // 32 + 1) * 32
        pad_h, pad_w = H_pad - H, W_pad - W
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        feats = self.backbone(x)
        if not isinstance(feats, (list, tuple)) or len(feats) < 4:
            raise RuntimeError("backbone must return at least 4 feature maps (f1,f2,f3,f4)")
        f1, f2, f3, f4 = feats[:4]

        if not self._built:
            self._build(f1.shape[1], f2.shape[1], f3.shape[1], f4.shape[1], ref=f4)

        if (self._stage == 2) or (is_teacher and self._stage == 2):
            with torch.no_grad():
                mask_pred = self.mask_gen(f4)
                unc_pred = self.unc_gen(f4)
        else:
            mask_pred = self.mask_gen(f4)
            unc_pred = self.unc_gen(f4)

        p2 = self.abf2(f3, f2, mask=mask_pred, unc=unc_pred)
        p1 = self.abf1(p2, f1, mask=mask_pred, unc=unc_pred)

        p1r = self.reduce_p1(p1)
        base = self.base_shared(p1r)

        low_feat = self.low_proj(f3)

        feat_reg = self.reg_branch(base)
        feat_cons = self.cons_branch(base)

        feat_reg = self.cross_scale(feat_reg, low_feat)
        feat_cons = self.token_relation(feat_cons)

        density_mu, density_log_sigma = self.head_final(feat_reg)
        low_mu, low_log_sigma = self.head_low(low_feat)
        res_pos, res_neg = self.head_res(feat_reg)

        dh, dw = density_mu.shape[-2:]
        crop_h = math.ceil(H * (dh / float(H_pad)))
        crop_w = math.ceil(W * (dw / float(W_pad)))

        def crop(t):
            return t[..., :crop_h, :crop_w]

        mask_up = F.interpolate(mask_pred, size=(dh, dw), mode="bilinear", align_corners=False)
        unc_up = F.interpolate(unc_pred, size=(dh, dw), mode="bilinear", align_corners=False)

        out = {
            "density_mu": crop(density_mu).clamp_min(0.0),
            "density_log_sigma": crop(density_log_sigma),
            "low_mu": low_mu,
            "low_log_sigma": low_log_sigma,
            "res_pos": crop(res_pos).clamp_min(0.0),
            "res_neg": crop(res_neg).clamp_min(0.0),
            "mask_pred": crop(mask_up).clamp(0.0, 1.0),
            "unc_pred": crop(unc_up).clamp(0.0, 1.0),
            "feat_reg": crop(feat_reg),
            "feat_cons": crop(feat_cons),
            "input_hw": (H, W),
            "pred_hw": (crop_h, crop_w),
            "low_hw": (low_mu.shape[-2], low_mu.shape[-1]),
        }
        out["count"] = out["density_mu"].sum(dim=[2, 3])
        return out
