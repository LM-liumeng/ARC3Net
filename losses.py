# losses.py
import math
import torch
import torch.nn.functional as F


# =========================================================
# mass-preserve resize
# =========================================================
def resize_density_mass_preserve(gt: torch.Tensor, target_hw, eps: float = 1e-8) -> torch.Tensor:
    th, tw = int(target_hw[0]), int(target_hw[1])
    gh, gw = gt.shape[-2], gt.shape[-1]
    if (gh, gw) == (th, tw):
        return gt

    if th <= gh and tw <= gw:
        out = F.interpolate(gt, size=(th, tw), mode="area")
    else:
        out = F.interpolate(gt, size=(th, tw), mode="bilinear", align_corners=False)

    s0 = gt.sum(dim=[2, 3], keepdim=True)
    s1 = out.sum(dim=[2, 3], keepdim=True).clamp_min(eps)
    return out * (s0 / s1)


# =========================================================
# patch pool (mass-preserve per patch)
# =========================================================
def patch_pool_density(d: torch.Tensor, grid=(16, 16)) -> torch.Tensor:
    gh, gw = int(grid[0]), int(grid[1])
    H, W = d.shape[-2], d.shape[-1]
    pooled = F.adaptive_avg_pool2d(d, (gh, gw))
    scale = (H / float(gh)) * (W / float(gw))
    return pooled * scale


# =========================================================
# GT-mask from density (soft mask, stable)
# =========================================================
def gt_mask_from_density(gt_rs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = gt_rs / (gt_rs.amax(dim=[2, 3], keepdim=True).clamp_min(eps))
    m = m.clamp(0.0, 1.0)
    return m.sqrt()


# =========================================================
# Gaussian NLL on density
# =========================================================
def density_gaussian_nll(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    gt: torch.Tensor,
    log_sigma_min: float = -3.0,
    log_sigma_max: float = 1.5
) -> torch.Tensor:
    ls = log_sigma.clamp(min=log_sigma_min, max=log_sigma_max)
    var = torch.exp(2.0 * ls)
    diff2 = (mu - gt) ** 2
    return 0.5 * (diff2 / (var + 1e-8) + torch.log(var + 1e-8) + math.log(2.0 * math.pi))


def sigma_floor_regularizer(log_sigma: torch.Tensor, floor: float = -2.5) -> torch.Tensor:
    return F.relu(float(floor) - log_sigma).mean()


# =========================================================
# Gaussian blur (fixed kernel, stable)
# =========================================================
def gaussian_blur2d(x: torch.Tensor, ksize: int = 5, sigma: float = 1.0) -> torch.Tensor:
    if ksize % 2 == 0:
        ksize += 1
    device, dtype = x.device, x.dtype
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    kernel = torch.exp(-(ax ** 2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()

    k1 = kernel.view(1, 1, ksize, 1)
    k2 = kernel.view(1, 1, 1, ksize)

    C = x.shape[1]
    x = F.conv2d(x, k1.repeat(C, 1, 1, 1), padding=(ksize // 2, 0), groups=C)
    x = F.conv2d(x, k2.repeat(C, 1, 1, 1), padding=(0, ksize // 2), groups=C)
    return x


# =========================================================
# band targets: gt_pred, gt_low, gt_res
# =========================================================
def make_band_targets(gt_density: torch.Tensor, pred_hw, low_hw, eps: float = 1e-8):
    gt_pred = resize_density_mass_preserve(gt_density, pred_hw, eps=eps)

    gt_low_pred = gaussian_blur2d(gt_pred, ksize=5, sigma=1.2)
    gt_low = F.interpolate(gt_low_pred, size=low_hw, mode="area")

    s_pred = gt_pred.sum(dim=[2, 3], keepdim=True)
    s_low = gt_low.sum(dim=[2, 3], keepdim=True).clamp_min(eps)
    gt_low = gt_low * (s_pred / s_low)

    gt_low_up = F.interpolate(gt_low, size=pred_hw, mode="bilinear", align_corners=False)
    s_low_up = gt_low_up.sum(dim=[2, 3], keepdim=True).clamp_min(eps)
    gt_low_up = gt_low_up * (s_pred / s_low_up)

    gt_res = gt_pred - gt_low_up
    return gt_pred, gt_low, gt_res


# =========================================================
# Stage1 supervised loss
# =========================================================
def supervised_stage1_loss(
    out: dict,
    gt_density: torch.Tensor,
    patch_grid=(16, 16),
    w_den: float = 1.0,
    w_patch: float = 0.05,
    w_mask: float = 0.2,
    w_low: float = 0.3,
    w_res: float = 0.2,
    w_res_neg: float = 0.01,
    eps: float = 1e-6
):
    mu = out["density_mu"]
    ls = out["density_log_sigma"]
    pred_hw = mu.shape[-2:]

    low_mu = out["low_mu"]
    low_ls = out["low_log_sigma"]
    low_hw = low_mu.shape[-2:]

    res_pos = out["res_pos"]
    res_neg = out["res_neg"]
    res_pred = res_pos - res_neg

    gt_pred, gt_low, gt_res = make_band_targets(gt_density, pred_hw, low_hw)

    L_den = density_gaussian_nll(mu, ls, gt_pred).mean()
    L_sigma = sigma_floor_regularizer(ls, floor=-2.5)

    mu_patch = patch_pool_density(mu, patch_grid)
    gt_patch = patch_pool_density(gt_pred, patch_grid)
    L_patch = F.smooth_l1_loss(mu_patch, gt_patch)

    m_gt = gt_mask_from_density(gt_pred, eps=eps)
    m_pr = out["mask_pred"].clamp(0.0, 1.0)
    L_mask = F.binary_cross_entropy(m_pr, m_gt)

    L_low = density_gaussian_nll(low_mu, low_ls, gt_low).mean()

    L_res = F.smooth_l1_loss(res_pred, gt_res)
    L_res_neg = res_neg.mean()

    loss = (
        float(w_den) * L_den +
        float(w_patch) * L_patch +
        float(w_mask) * L_mask +
        float(w_low) * L_low +
        float(w_res) * L_res +
        float(w_res_neg) * L_res_neg +
        0.02 * L_sigma
    )

    parts = {
        "loss_total": float(loss.detach().cpu()),
        "L_den": float(L_den.detach().cpu()),
        "L_patch": float(L_patch.detach().cpu()),
        "L_mask": float(L_mask.detach().cpu()),
        "L_low": float(L_low.detach().cpu()),
        "L_res": float(L_res.detach().cpu()),
        "L_res_neg": float(L_res_neg.detach().cpu()),
        "pred_cnt_mean": float(mu.sum(dim=[2, 3]).detach().mean().cpu()),
        "gt_cnt_mean": float(gt_density.sum(dim=[2, 3]).detach().mean().cpu()),
        "L_sigma": float(L_sigma.detach().cpu()),
    }
    return loss, parts


def supervised_stage2_anchor(
    out: dict,
    gt_density: torch.Tensor,
    patch_grid=(16, 16),
    w_den: float = 1.0,
    w_patch: float = 0.05,
    w_mask: float = 0.2,
    w_low: float = 0.3,
    w_res: float = 0.2,
    w_res_neg: float = 0.01
):
    return supervised_stage1_loss(
        out, gt_density,
        patch_grid=patch_grid,
        w_den=w_den, w_patch=w_patch, w_mask=w_mask,
        w_low=w_low, w_res=w_res, w_res_neg=w_res_neg
    )


# =========================================================
# Stage2 weighted unlabeled regularizers + MC-var weighted relation
# =========================================================
def rampup_weight(epoch: int, ramp_epochs: int, max_w: float):
    if ramp_epochs <= 0:
        return float(max_w)
    t = float(max(0, min(epoch, ramp_epochs))) / float(ramp_epochs)
    w = math.exp(-5.0 * (1.0 - t) * (1.0 - t))
    return float(max_w) * float(w)


def _resize_like(x: torch.Tensor, hw, mode="bilinear"):
    if x.shape[-2:] == tuple(hw):
        return x
    if mode == "nearest":
        return F.interpolate(x, size=hw, mode="nearest")
    return F.interpolate(x, size=hw, mode="bilinear", align_corners=False)


@torch.no_grad()
def build_trust_weight_map(
    teacher_out: dict,
    target_hw,
    tau_sigma: float = -2.0,
    k_sigma: float = 2.0,
    eps: float = 1e-6,
    use_mask_pred: bool = True,
    mask_soft_strength: float = 0.5,  # NEW
):
    """
    主权重：w = w_resp * w_unc
      w_resp = sqrt(normalize(mu_t))
      w_unc  = sigmoid(-k*(log_sigma - tau))

    mask 作为 soft prior：
      w *= ((1 - s) + s * mask), s in [0,1]
      s=0 -> 不用 mask prior
      s=1 -> 退化为硬乘子
    """
    mu_t = _resize_like(teacher_out["density_mu"].detach(), target_hw, mode="bilinear")
    ls_t = _resize_like(teacher_out["density_log_sigma"].detach(), target_hw, mode="bilinear")

    mu_norm = mu_t / (mu_t.amax(dim=[2, 3], keepdim=True).clamp_min(eps))
    w_resp = mu_norm.sqrt().clamp(0.0, 1.0)

    w_unc = torch.sigmoid(-float(k_sigma) * (ls_t - float(tau_sigma))).clamp(0.0, 1.0)

    w = (w_resp * w_unc).clamp(0.0, 1.0)

    if use_mask_pred and ("mask_pred" in teacher_out):
        s = float(mask_soft_strength)
        s = max(0.0, min(1.0, s))
        fg = _resize_like(teacher_out["mask_pred"].detach(), target_hw, mode="bilinear").clamp(0.0, 1.0)
        w = (w * ((1.0 - s) + s * fg)).clamp(0.0, 1.0)

    return w


def masked_cosine_feature_loss(fs: torch.Tensor, ft: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    if ft.shape[-2:] != fs.shape[-2:]:
        ft = _resize_like(ft, fs.shape[-2:], mode="bilinear")
    if w.shape[-2:] != fs.shape[-2:]:
        w = _resize_like(w, fs.shape[-2:], mode="bilinear").clamp(0.0, 1.0)

    fs_n = F.normalize(fs, dim=1, eps=eps)
    ft_n = F.normalize(ft.detach(), dim=1, eps=eps)

    dist = 1.0 - (fs_n * ft_n).sum(dim=1, keepdim=True)
    denom = w.mean().clamp_min(eps)
    return (dist * w).mean() / denom


def masked_output_mse(mu_s: torch.Tensor, mu_t: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    if mu_t.shape[-2:] != mu_s.shape[-2:]:
        mu_t = _resize_like(mu_t, mu_s.shape[-2:], mode="bilinear")
    if w.shape[-2:] != mu_s.shape[-2:]:
        w = _resize_like(w, mu_s.shape[-2:], mode="bilinear").clamp(0.0, 1.0)

    diff2 = (mu_s - mu_t.detach()).pow(2)
    denom = w.mean().clamp_min(eps)
    return (diff2 * w).mean() / denom


def _tokens_from_feat(feat: torch.Tensor, grid=(16, 16), eps: float = 1e-6):
    gh, gw = int(grid[0]), int(grid[1])
    t = F.adaptive_avg_pool2d(feat, (gh, gw))
    t = t.flatten(2).transpose(1, 2)
    t = F.normalize(t, dim=-1, eps=eps)
    return t


def _affinity_from_tokens(tokens: torch.Tensor):
    return torch.bmm(tokens, tokens.transpose(1, 2))


@torch.no_grad()
def teacher_affinity_mc_mean_var(teacher_feat_list, grid=(16, 16), eps: float = 1e-6):
    As = []
    for ft in teacher_feat_list:
        tok = _tokens_from_feat(ft.detach(), grid=grid, eps=eps)
        As.append(_affinity_from_tokens(tok))
    A_stack = torch.stack(As, dim=0)
    return A_stack.mean(dim=0), A_stack.var(dim=0, unbiased=False)


def relation_loss_weighted_by_mcvar(
    fs: torch.Tensor,
    teacher_A_mean: torch.Tensor,
    teacher_A_var: torch.Tensor,
    w_pix: torch.Tensor,
    grid=(16, 16),
    beta: float = 5.0,
    eps: float = 1e-6,
):
    tok_s = _tokens_from_feat(fs, grid=grid, eps=eps)
    A_s = _affinity_from_tokens(tok_s)

    gh, gw = int(grid[0]), int(grid[1])
    w_tok = F.adaptive_avg_pool2d(w_pix, (gh, gw)).flatten(2).squeeze(1).clamp(0.0, 1.0)
    W_tok = w_tok.unsqueeze(2) * w_tok.unsqueeze(1)

    W_var = torch.exp(-float(beta) * teacher_A_var).clamp(0.0, 1.0)
    W = (W_tok * W_var).clamp(0.0, 1.0)

    diff = F.smooth_l1_loss(A_s, teacher_A_mean.detach(), reduction="none")
    denom = W.mean().clamp_min(eps)
    loss = (diff * W).mean() / denom

    stats = {
        "W_rel_mean": float(W.detach().mean().cpu()),
        "A_var_mean": float(teacher_A_var.detach().mean().cpu()),
    }
    return loss, stats


def stage2_unlabeled_regularizer_mcvar(
    student_out: dict,
    teacher_out_mean: dict,
    teacher_feat_list: list,
    epoch: int,
    ramp_epochs: int,
    max_w: float,
    w_feat: float = 1.0,
    w_rel: float = 0.2,
    w_out: float = 0.05,
    relation_grid=(16, 16),
    tau_sigma: float = -2.0,
    k_sigma: float = 2.0,
    use_mask_pred: bool = True,
    mask_soft_strength: float = 0.5,  # NEW
    rel_beta: float = 5.0,
):
    w_ramp = rampup_weight(epoch, ramp_epochs, max_w)

    fs = student_out["feat_cons"]
    ft = teacher_out_mean["feat_cons"]

    w_feat_map = build_trust_weight_map(
        teacher_out_mean,
        target_hw=fs.shape[-2:],
        tau_sigma=tau_sigma,
        k_sigma=k_sigma,
        use_mask_pred=use_mask_pred,
        mask_soft_strength=mask_soft_strength,  # NEW
    )
    L_feat = masked_cosine_feature_loss(fs, ft, w=w_feat_map)

    A_mean, A_var = teacher_affinity_mc_mean_var(teacher_feat_list, grid=relation_grid)
    L_rel, rel_stats = relation_loss_weighted_by_mcvar(
        fs=fs,
        teacher_A_mean=A_mean,
        teacher_A_var=A_var,
        w_pix=w_feat_map,
        grid=relation_grid,
        beta=rel_beta,
    )

    mu_s = student_out["density_mu"]
    mu_t = teacher_out_mean["density_mu"]
    w_out_map = build_trust_weight_map(
        teacher_out_mean,
        target_hw=mu_s.shape[-2:],
        tau_sigma=tau_sigma,
        k_sigma=k_sigma,
        use_mask_pred=use_mask_pred,
        mask_soft_strength=mask_soft_strength,  # NEW
    )
    L_out = masked_output_mse(mu_s, mu_t, w=w_out_map)

    u = w_ramp * (float(w_feat) * L_feat + float(w_rel) * L_rel + float(w_out) * L_out)

    parts = {
        "u_total": float(u.detach().cpu()),
        "u_feat": float(L_feat.detach().cpu()),
        "u_rel": float(L_rel.detach().cpu()),
        "u_out": float(L_out.detach().cpu()),
        "ramp": float(w_ramp),
        "w_feat_mean": float(w_feat_map.detach().mean().cpu()),
        "w_out_mean": float(w_out_map.detach().mean().cpu()),
        "W_rel_mean": float(rel_stats["W_rel_mean"]),
        "A_var_mean": float(rel_stats["A_var_mean"]),
        "mc_M": int(len(teacher_feat_list)),
    }
    return u, parts
