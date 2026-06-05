"""Losses for the reliability-guided crowd counting model.

The loss design treats reliability as a learned, supervised estimate of where
the current model can be trusted. Unlabeled consistency is reduced both
spatially and globally when teacher reliability is low.
"""

import math

import torch
import torch.nn.functional as F


def resize_density_mass_preserve(density: torch.Tensor, target_hw, eps: float = 1e-8):
    target_hw = (int(target_hw[0]), int(target_hw[1]))
    if density.shape[-2:] == target_hw:
        return density

    source_sum = density.sum(dim=[2, 3], keepdim=True)
    if target_hw[0] <= density.shape[-2] and target_hw[1] <= density.shape[-1]:
        resized = F.interpolate(density, size=target_hw, mode="area")
    else:
        resized = F.interpolate(density, size=target_hw, mode="bilinear", align_corners=False)
    resized_sum = resized.sum(dim=[2, 3], keepdim=True).clamp_min(eps)
    return resized * (source_sum / resized_sum)


def patch_count_map(density: torch.Tensor, grid=(8, 8)):
    grid = (int(grid[0]), int(grid[1]))
    height, width = density.shape[-2:]
    pooled = F.adaptive_avg_pool2d(density, grid)
    return pooled * (height / float(grid[0])) * (width / float(grid[1]))


def gaussian_blur2d(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.2):
    if kernel_size % 2 == 0:
        kernel_size += 1
    axis = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - (kernel_size - 1) / 2
    kernel = torch.exp(-(axis.square()) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = x.shape[1]
    vertical = kernel.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1)
    horizontal = kernel.view(1, 1, 1, kernel_size).repeat(channels, 1, 1, 1)
    x = F.conv2d(x, vertical, padding=(kernel_size // 2, 0), groups=channels)
    return F.conv2d(x, horizontal, padding=(0, kernel_size // 2), groups=channels)


def foreground_target(gt_density: torch.Tensor, eps: float = 1e-6):
    local = gaussian_blur2d(gt_density, kernel_size=5, sigma=1.0)
    scale = local.amax(dim=[2, 3], keepdim=True).clamp_min(eps)
    return (local / scale).clamp(0.0, 1.0).sqrt()


@torch.no_grad()
def reliability_target(pred_density: torch.Tensor, gt_density: torch.Tensor, eps: float = 1e-6):
    """Build a relative local reliability target with a global count-quality gate."""
    local_error = (torch.log1p(pred_density) - torch.log1p(gt_density)).abs()
    error_scale = local_error.mean(dim=[2, 3], keepdim=True).clamp_min(eps)
    local_quality = torch.exp(-local_error / error_scale).clamp(0.0, 1.0)

    pred_count = pred_density.sum(dim=[1, 2, 3])
    gt_count = gt_density.sum(dim=[1, 2, 3])
    count_error = (torch.log1p(pred_count) - torch.log1p(gt_count)).abs()
    count_quality = torch.exp(-count_error).view(-1, 1, 1, 1)
    return (local_quality * count_quality).clamp(0.0, 1.0)


def density_gaussian_nll(
    prediction: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    log_sigma_min: float = -4.0,
    log_sigma_max: float = 1.5,
):
    log_sigma = log_sigma.clamp(log_sigma_min, log_sigma_max)
    squared_error = (prediction - target).square()
    inverse_variance = torch.exp(-2.0 * log_sigma)
    return 0.5 * (squared_error * inverse_variance + 2.0 * log_sigma + math.log(2.0 * math.pi))


def sigma_calibration_loss(
    prediction: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    floor: float = 1e-4,
):
    with torch.no_grad():
        target_log_sigma = (prediction.detach() - target).abs().add(floor).log().clamp(-4.0, 1.5)
    return F.smooth_l1_loss(log_sigma.clamp(-4.0, 1.5), target_log_sigma)


def _resize_like(x: torch.Tensor, target_hw):
    if x.shape[-2:] == tuple(target_hw):
        return x
    return F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)


def _branch_nll(output: dict, prefix: str, target: torch.Tensor):
    prediction = output[f"{prefix}_mu"]
    log_sigma = output[f"{prefix}_log_sigma"]
    target = resize_density_mass_preserve(target, prediction.shape[-2:])
    return density_gaussian_nll(prediction, log_sigma, target).mean()


def supervised_reliability_loss(
    output: dict,
    gt_density: torch.Tensor,
    patch_grid=(8, 8),
    w_density: float = 1.0,
    w_count: float = 0.1,
    w_patch: float = 0.1,
    w_foreground: float = 0.1,
    w_reliability: float = 0.1,
    w_sigma_calibration: float = 0.05,
    w_base: float = 0.25,
    w_low: float = 0.2,
):
    prediction = output["density_mu"]
    log_sigma = output["density_log_sigma"]
    gt_resized = resize_density_mass_preserve(gt_density, prediction.shape[-2:])

    loss_density = density_gaussian_nll(prediction, log_sigma, gt_resized).mean()
    loss_sigma_calibration = sigma_calibration_loss(prediction, log_sigma, gt_resized)

    pred_count = prediction.sum(dim=[1, 2, 3])
    gt_count = gt_density.sum(dim=[1, 2, 3])
    loss_count = F.smooth_l1_loss(torch.log1p(pred_count), torch.log1p(gt_count))

    pred_patch = patch_count_map(prediction, patch_grid)
    gt_patch = patch_count_map(gt_resized, patch_grid)
    loss_patch = F.smooth_l1_loss(torch.log1p(pred_patch), torch.log1p(gt_patch))

    foreground_gt = foreground_target(gt_resized)
    if "foreground_logit" in output:
        foreground_logit = _resize_like(output["foreground_logit"], prediction.shape[-2:])
        loss_foreground = F.binary_cross_entropy_with_logits(foreground_logit, foreground_gt)
        foreground = torch.sigmoid(foreground_logit)
    else:
        foreground = _resize_like(output["foreground_pred"], prediction.shape[-2:]).clamp(
            1e-5, 1.0 - 1e-5
        )
        loss_foreground = F.binary_cross_entropy(foreground, foreground_gt)

    reliability_gt = reliability_target(prediction.detach(), gt_resized)
    if "reliability_logit" in output:
        reliability_logit = _resize_like(output["reliability_logit"], prediction.shape[-2:])
        loss_reliability = F.binary_cross_entropy_with_logits(reliability_logit, reliability_gt)
        reliability = torch.sigmoid(reliability_logit)
    else:
        reliability = _resize_like(output["reliability_pred"], prediction.shape[-2:]).clamp(
            1e-5, 1.0 - 1e-5
        )
        loss_reliability = F.binary_cross_entropy(reliability, reliability_gt)

    loss_base = _branch_nll(output, "base", gt_density)
    low_target = gaussian_blur2d(gt_density, kernel_size=5, sigma=1.2)
    low_target = low_target * (
        gt_density.sum(dim=[2, 3], keepdim=True)
        / low_target.sum(dim=[2, 3], keepdim=True).clamp_min(1e-8)
    )
    loss_low = _branch_nll(output, "low", low_target)

    loss = (
        float(w_density) * loss_density
        + float(w_count) * loss_count
        + float(w_patch) * loss_patch
        + float(w_foreground) * loss_foreground
        + float(w_reliability) * loss_reliability
        + float(w_sigma_calibration) * loss_sigma_calibration
        + float(w_base) * loss_base
        + float(w_low) * loss_low
    )

    parts = {
        "loss_total": float(loss.detach().cpu()),
        "L_den": float(loss_density.detach().cpu()),
        "L_count": float(loss_count.detach().cpu()),
        "L_patch": float(loss_patch.detach().cpu()),
        "L_foreground": float(loss_foreground.detach().cpu()),
        "L_reliability": float(loss_reliability.detach().cpu()),
        "L_sigma_cal": float(loss_sigma_calibration.detach().cpu()),
        "L_base": float(loss_base.detach().cpu()),
        "L_low": float(loss_low.detach().cpu()),
        "reliability_mean": float(reliability.detach().mean().cpu()),
        "reliability_target_mean": float(reliability_gt.detach().mean().cpu()),
        "pred_cnt_mean": float(pred_count.detach().mean().cpu()),
        "gt_cnt_mean": float(gt_count.detach().mean().cpu()),
    }
    return loss, parts


def rampup_weight(epoch: int, ramp_epochs: int, max_weight: float):
    if ramp_epochs <= 0:
        return float(max_weight)
    progress = float(max(0, min(epoch, ramp_epochs))) / float(ramp_epochs)
    return float(max_weight) * math.exp(-5.0 * (1.0 - progress) ** 2)


@torch.no_grad()
def build_reliability_trust_map(teacher_output: dict, target_hw, foreground_floor: float = 0.1):
    density = _resize_like(teacher_output["density_mu"].detach(), target_hw).clamp_min(0.0)
    log_sigma = _resize_like(teacher_output["density_log_sigma"].detach(), target_hw)
    sigma = torch.exp(log_sigma.clamp(-4.0, 1.5))

    reliability_source = teacher_output.get("reliability_pred", teacher_output.get("unc_pred"))
    if reliability_source is None:
        reliability = torch.ones_like(density)
    else:
        reliability = _resize_like(reliability_source.detach(), target_hw).clamp(0.0, 1.0)

    foreground_source = teacher_output.get("foreground_pred", teacher_output.get("mask_pred"))
    if foreground_source is None:
        foreground = torch.ones_like(density)
    else:
        foreground = _resize_like(foreground_source.detach(), target_hw).clamp(0.0, 1.0)

    density_scale = density.mean(dim=[2, 3], keepdim=True)
    uncertainty_confidence = (density + density_scale + 1e-6) / (
        density + density_scale + sigma + 1e-6
    )
    foreground_prior = float(foreground_floor) + (1.0 - float(foreground_floor)) * foreground
    return (reliability * uncertainty_confidence * foreground_prior).clamp(0.0, 1.0)


def weighted_feature_consistency(student_feature, teacher_feature, trust):
    teacher_feature = _resize_like(teacher_feature.detach(), student_feature.shape[-2:])
    trust = _resize_like(trust, student_feature.shape[-2:]).clamp(0.0, 1.0)
    student_feature = F.normalize(student_feature, dim=1, eps=1e-6)
    teacher_feature = F.normalize(teacher_feature, dim=1, eps=1e-6)
    distance = 1.0 - (student_feature * teacher_feature).sum(dim=1, keepdim=True)
    return (distance * trust).mean()


def weighted_output_consistency(student_density, teacher_density, trust):
    teacher_density = _resize_like(teacher_density.detach(), student_density.shape[-2:])
    trust = _resize_like(trust, student_density.shape[-2:]).clamp(0.0, 1.0)
    difference = (torch.log1p(student_density) - torch.log1p(teacher_density)).square()
    return (difference * trust).mean()


def _tokens_from_feature(feature: torch.Tensor, grid=(8, 8)):
    tokens = F.adaptive_avg_pool2d(feature, grid).flatten(2).transpose(1, 2)
    return F.normalize(tokens, dim=-1, eps=1e-6)


def _affinity(tokens: torch.Tensor):
    return torch.bmm(tokens, tokens.transpose(1, 2))


@torch.no_grad()
def teacher_affinity_mean_variance(teacher_feature_list, grid=(8, 8)):
    if not teacher_feature_list:
        raise ValueError("teacher_feature_list must contain at least one feature tensor")
    affinities = [_affinity(_tokens_from_feature(feature.detach(), grid)) for feature in teacher_feature_list]
    stacked = torch.stack(affinities, dim=0)
    return stacked.mean(dim=0), stacked.var(dim=0, unbiased=False)


def reliability_weighted_relation_loss(
    student_feature,
    teacher_affinity_mean,
    teacher_affinity_variance,
    trust,
    grid=(8, 8),
    variance_decay: float = 5.0,
):
    student_affinity = _affinity(_tokens_from_feature(student_feature, grid))
    token_trust = F.adaptive_avg_pool2d(trust, grid).flatten(2).squeeze(1).clamp(0.0, 1.0)
    pair_trust = token_trust.unsqueeze(2) * token_trust.unsqueeze(1)
    variance_trust = torch.exp(-float(variance_decay) * teacher_affinity_variance).clamp(0.0, 1.0)
    combined_trust = pair_trust * variance_trust
    difference = F.smooth_l1_loss(student_affinity, teacher_affinity_mean.detach(), reduction="none")
    return (difference * combined_trust).mean(), combined_trust


def stage2_reliability_regularizer(
    student_output: dict,
    teacher_output_mean: dict,
    teacher_feature_list: list,
    epoch: int,
    ramp_epochs: int,
    max_weight: float,
    relation_grid=(8, 8),
    w_feature: float = 1.0,
    w_relation: float = 0.2,
    w_output: float = 0.1,
    w_count: float = 0.1,
    w_reliability: float = 0.05,
    variance_decay: float = 5.0,
):
    trust_feature = build_reliability_trust_map(
        teacher_output_mean, student_output["feat_cons"].shape[-2:]
    )
    trust_output = build_reliability_trust_map(
        teacher_output_mean, student_output["density_mu"].shape[-2:]
    )

    loss_feature = weighted_feature_consistency(
        student_output["feat_cons"], teacher_output_mean["feat_cons"], trust_feature
    )
    teacher_affinity_mean, teacher_affinity_variance = teacher_affinity_mean_variance(
        teacher_feature_list, relation_grid
    )
    loss_relation, relation_trust = reliability_weighted_relation_loss(
        student_output["feat_cons"],
        teacher_affinity_mean,
        teacher_affinity_variance,
        trust_feature,
        grid=relation_grid,
        variance_decay=variance_decay,
    )
    loss_output = weighted_output_consistency(
        student_output["density_mu"], teacher_output_mean["density_mu"], trust_output
    )

    image_trust = trust_output.mean(dim=[1, 2, 3]).detach()
    student_count = student_output["density_mu"].sum(dim=[1, 2, 3])
    teacher_count = teacher_output_mean["density_mu"].detach().sum(dim=[1, 2, 3])
    count_error = F.smooth_l1_loss(
        torch.log1p(student_count), torch.log1p(teacher_count), reduction="none"
    )
    loss_count = (count_error * image_trust).mean()

    student_reliability = student_output.get("reliability_pred", student_output.get("unc_pred"))
    teacher_reliability = teacher_output_mean.get(
        "reliability_pred", teacher_output_mean.get("unc_pred")
    )
    if student_reliability is None or teacher_reliability is None:
        loss_reliability = student_output["density_mu"].new_zeros(())
    else:
        teacher_reliability = _resize_like(
            teacher_reliability.detach(), student_reliability.shape[-2:]
        )
        reliability_trust = _resize_like(trust_output, student_reliability.shape[-2:])
        loss_reliability = (
            (student_reliability - teacher_reliability).square() * reliability_trust
        ).mean()

    ramp = rampup_weight(epoch, ramp_epochs, max_weight)
    core = (
        float(w_feature) * loss_feature
        + float(w_relation) * loss_relation
        + float(w_output) * loss_output
        + float(w_count) * loss_count
        + float(w_reliability) * loss_reliability
    )
    loss = float(ramp) * core

    parts = {
        "u_total": float(loss.detach().cpu()),
        "u_feat": float(loss_feature.detach().cpu()),
        "u_rel": float(loss_relation.detach().cpu()),
        "u_out": float(loss_output.detach().cpu()),
        "u_count": float(loss_count.detach().cpu()),
        "u_reliability": float(loss_reliability.detach().cpu()),
        "ramp": float(ramp),
        "w_feat_mean": float(trust_feature.detach().mean().cpu()),
        "w_out_mean": float(trust_output.detach().mean().cpu()),
        "W_rel_mean": float(relation_trust.detach().mean().cpu()),
        "A_var_mean": float(teacher_affinity_variance.detach().mean().cpu()),
        "mc_M": int(len(teacher_feature_list)),
    }
    return loss, parts
