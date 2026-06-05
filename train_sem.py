"""Legacy semi-supervised training entry for mamba_semcount_v3.py."""

import argparse
import copy
import math
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from dataset.loaddata import build_dataloaders
from Module.mamba_semcount_v3 import DARPComplexCrowd
from losses import (
    supervised_stage1_loss,
    supervised_stage2_anchor,
    stage2_unlabeled_regularizer_mcvar,
)


def _flatten_config(data):
    flat = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat.update(_flatten_config(value))
        else:
            flat[key] = value
    return flat


def _parse_scalar(value):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if "," in value:
        return [_parse_scalar(part.strip()) for part in value.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _legacy_cli_overrides(unknown):
    overrides = []
    index = 0
    while index < len(unknown):
        token = unknown[index]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected argument without option name: {token}")

        token = token[2:]
        if "=" in token:
            key, value = token.split("=", 1)
            overrides.append(f"{key.replace('-', '_')}={value}")
            index += 1
            continue

        values = []
        index += 1
        while index < len(unknown) and not unknown[index].startswith("--"):
            values.append(unknown[index])
            index += 1

        key = token.replace("-", "_")
        value = ",".join(values) if values else "true"
        overrides.append(f"{key}={value}")
    return overrides


def _apply_overrides(args, overrides):
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override, expected key=value: {override}")
        key, value = override.split("=", 1)
        args[key.strip()] = _parse_scalar(value.strip())


def load_args():
    parser = argparse.ArgumentParser(description="Train legacy ARC3Net semi-supervised baseline")
    parser.add_argument("--config", type=str, default="configs/train_sem.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    cli, unknown = parser.parse_known_args()

    with open(cli.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    args = _flatten_config(config)
    args["config"] = cli.config
    _apply_overrides(args, cli.overrides + _legacy_cli_overrides(unknown))
    return SimpleNamespace(**args)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model, loader_val, device):
    model.eval()
    abs_err_sum, sq_err_sum, count = 0.0, 0.0, 0
    for batch in loader_val:
        image, gt_count = batch[0], batch[1]
        image = image.to(device, non_blocking=True)
        gt = gt_count.to(device, non_blocking=True).float().view(-1)

        output = model(image, stage=2, is_teacher=True)
        pred = output["density_mu"].sum(dim=[1, 2, 3]).detach().float().view(-1)
        error = pred - gt

        abs_err_sum += error.abs().sum().item()
        sq_err_sum += error.square().sum().item()
        count += gt.numel()

    mae = abs_err_sum / max(1, count)
    rmse = math.sqrt(sq_err_sum / max(1, count))
    return mae, rmse


def save_ckpt(path, model, optimizer, epoch, best_mae, best_rmse):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_mae": float(best_mae),
            "best_rmse": float(best_rmse),
        },
        path,
    )


def load_ckpt(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_mae", 1e9)), float(ckpt.get("best_rmse", 1e9))


def build_optimizer(model, base_lr, weight_decay, backbone_mult):
    backbone_params, head_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": base_lr * float(backbone_mult)},
            {"params": head_params, "lr": base_lr},
        ],
        lr=base_lr,
        weight_decay=weight_decay,
    )


@torch.no_grad()
def teacher_mc_forward(teacher, image, stage, mc_samples):
    previous_mode = teacher.training
    teacher.train()

    outputs = []
    features = []
    for _ in range(int(mc_samples)):
        output = teacher(image, stage=stage, is_teacher=True)
        outputs.append(output)
        features.append(output["feat_cons"].detach())

    teacher.train(previous_mode)

    def mean_stack(key):
        return torch.stack([output[key].detach() for output in outputs], dim=0).mean(dim=0)

    output_mean = {
        "density_mu": mean_stack("density_mu"),
        "density_log_sigma": mean_stack("density_log_sigma"),
        "mask_pred": mean_stack("mask_pred"),
        "feat_cons": mean_stack("feat_cons"),
    }
    return output_mean, features


def supervised_stage_loss(output, gt_density, args):
    return supervised_stage1_loss(
        output,
        gt_density,
        patch_grid=tuple(args.patch_grid),
        w_den=args.w_den,
        w_patch=args.w_patch,
        w_mask=args.w_mask,
        w_low=args.w_low,
        w_res=args.w_res,
        w_res_neg=args.w_res_neg,
    )


def stage1(model, loader_labeled, loader_val, device, args):
    model.train()
    model.set_stage(1)
    optimizer = build_optimizer(model, args.lr_stage1, args.weight_decay, args.lr_backbone_mult)

    start_epoch = 1
    best_mae, best_rmse = float("inf"), float("inf")
    if args.resume_stage1:
        last_epoch, best_mae, best_rmse = load_ckpt(args.resume_stage1, model, optimizer, device)
        start_epoch = last_epoch + 1

    save_best = os.path.join(args.save_dir, "stage1_best.pth")

    for epoch in range(start_epoch, args.epochs_stage1 + 1):
        model.train()
        model.set_stage(1)
        sums = {"loss": 0.0, "den": 0.0, "patch": 0.0, "mask": 0.0, "low": 0.0, "res": 0.0}
        updates = 0

        for image, gt_density in loader_labeled:
            image = image.to(device, non_blocking=True)
            gt_density = gt_density.to(device, non_blocking=True)

            output = model(image, stage=1, is_teacher=False)
            loss, parts = supervised_stage_loss(output, gt_density, args)
            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            sums["loss"] += parts["loss_total"]
            sums["den"] += parts["L_den"]
            sums["patch"] += parts["L_patch"]
            sums["mask"] += parts["L_mask"]
            sums["low"] += parts["L_low"]
            sums["res"] += parts["L_res"]
            updates += 1

        if updates == 0:
            raise RuntimeError(f"Stage1 epoch {epoch} produced no finite updates")
        if epoch % args.eval_every != 0 and epoch != args.epochs_stage1:
            continue

        mae, rmse = evaluate(model, loader_val, device)
        denom = float(updates)
        print(
            f"Stage1 Epoch {epoch:4d} | loss {sums['loss'] / denom:.4f} | "
            f"MAE {mae:.2f} RMSE {rmse:.2f} | Den {sums['den'] / denom:.4f} "
            f"Patch {sums['patch'] / denom:.6f} Mask {sums['mask'] / denom:.4f} "
            f"Low {sums['low'] / denom:.4f} Res {sums['res'] / denom:.4f}"
        )

        if math.isfinite(mae) and (mae < best_mae or (mae == best_mae and rmse < best_rmse)):
            best_mae, best_rmse = mae, rmse
            save_ckpt(save_best, model, optimizer, epoch, best_mae, best_rmse)

    return save_best


def stage2(model, loader_labeled, loader_unlabeled, loader_val, device, args, stage1_best_path):
    if os.path.exists(stage1_best_path):
        load_ckpt(stage1_best_path, model, optimizer=None, device=device)

    model.train()
    model.set_stage(2)
    model.set_sigma_frozen(False)
    optimizer = build_optimizer(model, args.lr_stage2, args.weight_decay, args.lr_backbone_mult)

    start_epoch = 1
    best_mae, best_rmse = float("inf"), float("inf")
    if args.resume_stage2:
        last_epoch, best_mae, best_rmse = load_ckpt(args.resume_stage2, model, optimizer, device)
        start_epoch = last_epoch + 1

    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    teacher.set_stage(2)
    teacher.set_sigma_frozen(True)
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    save_best = os.path.join(args.save_dir, "stage2_best.pth")
    unlabeled_iter = iter(loader_unlabeled)

    for epoch in range(start_epoch, args.epochs_stage2 + 1):
        sigma_train = int(epoch) <= int(args.sigma_unfreeze_epochs)
        model.train()
        model.set_stage(2)
        model.set_sigma_frozen(not sigma_train)
        teacher.eval()
        teacher.set_stage(2)

        sums = {
            "total": 0.0,
            "sup": 0.0,
            "unsup": 0.0,
            "u_feat": 0.0,
            "u_rel": 0.0,
            "u_out": 0.0,
            "ramp": 0.0,
            "w_out": 0.0,
            "w_rel": 0.0,
        }
        updates = 0

        for image_l, gt_density in loader_labeled:
            image_l = image_l.to(device, non_blocking=True)
            gt_density = gt_density.to(device, non_blocking=True)

            try:
                image_u_w, image_u_s = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(loader_unlabeled)
                image_u_w, image_u_s = next(unlabeled_iter)

            image_u_w = image_u_w.to(device, non_blocking=True)
            image_u_s = image_u_s.to(device, non_blocking=True)

            output_l = model(image_l, stage=2, is_teacher=False)
            loss_sup, _ = supervised_stage_loss(output_l, gt_density, args)
            if not torch.isfinite(loss_sup):
                continue

            with torch.no_grad():
                teacher_mean, teacher_features = teacher_mc_forward(
                    teacher, image_u_w, stage=2, mc_samples=args.mc_samples
                )

            output_s = model(image_u_s, stage=2, is_teacher=False)
            loss_unsup, unsup = stage2_unlabeled_regularizer_mcvar(
                student_out=output_s,
                teacher_out_mean=teacher_mean,
                teacher_feat_list=teacher_features,
                epoch=epoch,
                ramp_epochs=args.rampup_epochs,
                max_w=args.max_unsup_weight,
                w_feat=args.w_u_feat,
                w_rel=args.w_u_rel,
                w_out=args.w_u_out,
                relation_grid=tuple(args.relation_grid),
                tau_sigma=args.tau_sigma,
                k_sigma=args.k_sigma,
                use_mask_pred=args.use_mask_pred,
                mask_soft_strength=args.mask_soft_strength,
                rel_beta=args.rel_beta,
            )

            loss = loss_sup + loss_unsup
            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            DARPComplexCrowd.update_teacher_ema(teacher, model, ema=args.ema_decay)

            sums["total"] += float(loss.detach().cpu())
            sums["sup"] += float(loss_sup.detach().cpu())
            sums["unsup"] += float(loss_unsup.detach().cpu())
            sums["u_feat"] += unsup["u_feat"]
            sums["u_rel"] += unsup["u_rel"]
            sums["u_out"] += unsup["u_out"]
            sums["ramp"] += unsup["ramp"]
            sums["w_out"] += unsup["w_out_mean"]
            sums["w_rel"] += unsup["W_rel_mean"]
            updates += 1

        if updates == 0:
            raise RuntimeError(f"Stage2 epoch {epoch} produced no finite updates")
        if epoch % args.eval_every != 0 and epoch != args.epochs_stage2:
            continue

        mae, rmse = evaluate(model, loader_val, device)
        denom = float(updates)
        print(
            f"Stage2 Epoch {epoch:4d} | loss {sums['total'] / denom:.4f} "
            f"(sup {sums['sup'] / denom:.4f} + unsup {sums['unsup'] / denom:.4f}) | "
            f"MAE {mae:.2f} RMSE {rmse:.2f} | ramp {sums['ramp'] / denom:.3f} | "
            f"uFeat {sums['u_feat'] / denom:.4f} uRel {sums['u_rel'] / denom:.4f} "
            f"uOut {sums['u_out'] / denom:.6f} wOut {sums['w_out'] / denom:.3f} "
            f"Wrel {sums['w_rel'] / denom:.3f} sigma_train={int(sigma_train)}"
        )

        if math.isfinite(mae) and (mae < best_mae or (mae == best_mae and rmse < best_rmse)):
            best_mae, best_rmse = mae, rmse
            save_ckpt(save_best, model, optimizer, epoch, best_mae, best_rmse)

    return save_best


def main():
    args = load_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"Config: {args.config}")

    loader_labeled, loader_unlabeled, loader_val = build_dataloaders(args)
    model = DARPComplexCrowd(
        pretrained=True,
        patch_grid=tuple(args.patch_grid),
        relation_grid=tuple(args.relation_grid),
    ).to(device)

    sample = next(iter(loader_labeled))
    with torch.no_grad():
        model(sample[0].to(device), stage=1, is_teacher=True)

    print("=== Stage1: legacy supervised anchor ===")
    stage1_best = stage1(model, loader_labeled, loader_val, device, args)
    print("=== Stage2: legacy calibrated mean teacher ===")
    stage2(model, loader_labeled, loader_unlabeled, loader_val, device, args, stage1_best)
    print("Done.")


if __name__ == "__main__":
    main()
