"""Training entry for the reliability-guided ARC3Net implementation.

This script is adapted to:
  - Module/mamba_reliability_v4.py
  - reliability_losses.py

It keeps the original two-stage idea:
  Stage 1: supervised reliability-guided warm-up with relation modules off.
  Stage 2: supervised anchor + reliability-weighted mean-teacher regularization.
"""

import argparse
import copy
import math
import os
import random
from contextlib import nullcontext

import numpy as np
import torch

from dataset.loaddata import build_dataloaders
from Module.mamba_reliability_v4 import ReliabilityGuidedCrowdCounter
from reliability_losses import supervised_reliability_loss, stage2_reliability_regularizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ARC3Net reliability-guided model")

    # data
    parser.add_argument("--data_root", type=str, default="/data/LM/Dataset")
    parser.add_argument("--dataset_name", type=str, default="SHA")
    parser.add_argument("--labeled_ratio", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--downsample_ratio", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # schedule
    parser.add_argument("--epochs_stage1", type=int, default=2000)
    parser.add_argument("--epochs_stage2", type=int, default=800)
    parser.add_argument("--eval_every", type=int, default=10)

    # optimization
    parser.add_argument("--lr_stage1", type=float, default=1e-5)
    parser.add_argument("--lr_stage2", type=float, default=5e-6)
    parser.add_argument("--lr_backbone_mult", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--amp", action="store_true")

    # supervised reliability loss
    parser.add_argument("--w_density", type=float, default=1.0)
    parser.add_argument("--w_count", type=float, default=0.1)
    parser.add_argument("--w_patch", type=float, default=0.1)
    parser.add_argument("--w_foreground", type=float, default=0.1)
    parser.add_argument("--w_reliability", type=float, default=0.1)
    parser.add_argument("--w_sigma_calibration", type=float, default=0.05)
    parser.add_argument("--w_base", type=float, default=0.25)
    parser.add_argument("--w_low", type=float, default=0.2)
    parser.add_argument("--patch_grid", type=int, nargs=2, default=[8, 8])
    parser.add_argument("--relation_grid", type=int, nargs=2, default=[8, 8])

    # stage2 reliability-weighted mean teacher
    parser.add_argument("--rampup_epochs", type=int, default=200)
    parser.add_argument("--max_unsup_weight", type=float, default=0.15)
    parser.add_argument("--w_u_feat", type=float, default=1.0)
    parser.add_argument("--w_u_rel", type=float, default=0.2)
    parser.add_argument("--w_u_out", type=float, default=0.1)
    parser.add_argument("--w_u_count", type=float, default=0.1)
    parser.add_argument("--w_u_reliability", type=float, default=0.05)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--mc_samples", type=int, default=2)
    parser.add_argument("--rel_beta", type=float, default=5.0)

    # sigma schedule
    parser.add_argument(
        "--sigma_unfreeze_epochs",
        type=int,
        default=300,
        help="In Stage2, train sigma heads for this many epochs, then freeze them.",
    )

    # checkpointing
    parser.add_argument("--save_dir", type=str, default="./checkpoint_arc3net")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="./weights/mambavision_base_21k.pth.tar",
        help="Local MambaVision-B pretrained checkpoint. Use empty string for random init.",
    )
    parser.add_argument("--resume_stage1", type=str, default="")
    parser.add_argument("--resume_stage2", type=str, default="")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _extract_state_dict(checkpoint):
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    return state_dict


def load_backbone_pretrained(model, path):
    if not path:
        print("Backbone init: random")
        return
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Backbone checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    incompatible = model.backbone.load_state_dict(_extract_state_dict(checkpoint), strict=False)
    print(
        f"Backbone init: {path} | "
        f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
    )


def build_optimizer(model, base_lr, weight_decay, backbone_mult):
    backbone_params = []
    head_params = []
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


def supervised_anchor(output, gt_density, args):
    return supervised_reliability_loss(
        output,
        gt_density,
        patch_grid=tuple(args.patch_grid),
        w_density=args.w_density,
        w_count=args.w_count,
        w_patch=args.w_patch,
        w_foreground=args.w_foreground,
        w_reliability=args.w_reliability,
        w_sigma_calibration=args.w_sigma_calibration,
        w_base=args.w_base,
        w_low=args.w_low,
    )


@torch.no_grad()
def evaluate(model, loader_val, device, stage):
    model.eval()
    abs_error_sum = 0.0
    sq_error_sum = 0.0
    sample_count = 0

    for batch in loader_val:
        image, gt_count = batch[0], batch[1]
        image = image.to(device, non_blocking=True)
        gt_count = gt_count.to(device, non_blocking=True).float().view(-1)

        output = model(image, stage=stage, is_teacher=True)
        pred_count = output["density_mu"].sum(dim=[1, 2, 3]).float().view(-1)
        error = pred_count - gt_count

        abs_error_sum += error.abs().sum().item()
        sq_error_sum += error.square().sum().item()
        sample_count += gt_count.numel()

    mae = abs_error_sum / max(1, sample_count)
    rmse = math.sqrt(sq_error_sum / max(1, sample_count))
    return mae, rmse


def save_checkpoint(path, model, optimizer, epoch, best_mae, stage, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "stage": int(stage),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_mae": float(best_mae),
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_mae", 1e9))


@torch.no_grad()
def teacher_mc_forward(teacher, image, stage, mc_samples):
    ReliabilityGuidedCrowdCounter.set_teacher_mc_mode(teacher)
    outputs = []
    features = []

    for _ in range(max(1, int(mc_samples))):
        output = teacher(image, stage=stage, is_teacher=True)
        outputs.append(output)
        features.append(output["feat_cons"].detach())

    teacher.eval()

    def mean_stack(key):
        tensors = [output[key].detach() for output in outputs if key in output]
        if not tensors:
            return None
        return torch.stack(tensors, dim=0).mean(dim=0)

    output_mean = {
        "density_mu": mean_stack("density_mu"),
        "density_log_sigma": mean_stack("density_log_sigma"),
        "foreground_pred": mean_stack("foreground_pred"),
        "reliability_pred": mean_stack("reliability_pred"),
        "mask_pred": mean_stack("mask_pred"),
        "unc_pred": mean_stack("unc_pred"),
        "feat_cons": mean_stack("feat_cons"),
    }
    return output_mean, features


def _make_scaler(device, use_amp):
    enabled = bool(use_amp and device.type == "cuda")
    return torch.cuda.amp.GradScaler(enabled=enabled), enabled


def _amp_context(enabled):
    if not enabled:
        return nullcontext()
    return torch.cuda.amp.autocast()


def stage1(model, loader_train, loader_val, device, args):
    model.train()
    model.set_stage(1)
    model.set_sigma_frozen(False)

    optimizer = build_optimizer(model, args.lr_stage1, args.weight_decay, args.lr_backbone_mult)
    scaler, amp_enabled = _make_scaler(device, args.amp)

    start_epoch = 1
    best_mae = float("inf")
    if args.resume_stage1:
        last_epoch, best_mae = load_checkpoint(args.resume_stage1, model, optimizer, device)
        start_epoch = last_epoch + 1

    best_path = os.path.join(args.save_dir, "stage1_best.pth")
    latest_path = os.path.join(args.save_dir, "stage1_latest.pth")

    for epoch in range(start_epoch, args.epochs_stage1 + 1):
        model.train()
        model.set_stage(1)
        model.set_sigma_frozen(False)

        sums = {
            "loss": 0.0,
            "den": 0.0,
            "count": 0.0,
            "patch": 0.0,
            "fg": 0.0,
            "rel": 0.0,
            "sigma": 0.0,
            "base": 0.0,
            "low": 0.0,
            "rel_mean": 0.0,
        }
        updates = 0

        for image, gt_density in loader_train:
            image = image.to(device, non_blocking=True)
            gt_density = gt_density.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with _amp_context(amp_enabled):
                output = model(image, stage=1, is_teacher=False)
                loss, parts = supervised_anchor(output, gt_density, args)

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            sums["loss"] += parts["loss_total"]
            sums["den"] += parts["L_den"]
            sums["count"] += parts["L_count"]
            sums["patch"] += parts["L_patch"]
            sums["fg"] += parts["L_foreground"]
            sums["rel"] += parts["L_reliability"]
            sums["sigma"] += parts["L_sigma_cal"]
            sums["base"] += parts["L_base"]
            sums["low"] += parts["L_low"]
            sums["rel_mean"] += parts["reliability_mean"]
            updates += 1

        if updates == 0:
            raise RuntimeError(f"Stage1 epoch {epoch} produced no finite updates")

        if epoch % args.eval_every != 0 and epoch != args.epochs_stage1:
            continue

        mae, rmse = evaluate(model, loader_val, device, stage=1)
        denom = float(updates)
        print(
            f"Stage1 Epoch {epoch:4d} | loss {sums['loss'] / denom:.4f} | "
            f"MAE {mae:.2f} RMSE {rmse:.2f} | "
            f"Den {sums['den'] / denom:.4f} Count {sums['count'] / denom:.4f} "
            f"Patch {sums['patch'] / denom:.4f} FG {sums['fg'] / denom:.4f} "
            f"Rel {sums['rel'] / denom:.4f} Sigma {sums['sigma'] / denom:.4f} "
            f"Base {sums['base'] / denom:.4f} Low {sums['low'] / denom:.4f} "
            f"RelMean {sums['rel_mean'] / denom:.3f}"
        )

        if math.isfinite(mae) and mae < best_mae:
            best_mae = mae
            save_checkpoint(best_path, model, optimizer, epoch, best_mae, stage=1, args=args)
        save_checkpoint(latest_path, model, optimizer, epoch, best_mae, stage=1, args=args)

    if not os.path.isfile(best_path):
        raise RuntimeError("Stage1 did not produce a finite validation MAE")
    return best_path


def stage2(model, loader_labeled, loader_unlabeled, loader_val, device, args, stage1_best_path):
    if os.path.exists(stage1_best_path):
        load_checkpoint(stage1_best_path, model, optimizer=None, device=device)

    model.train()
    model.set_stage(2)
    model.set_sigma_frozen(False)

    optimizer = build_optimizer(model, args.lr_stage2, args.weight_decay, args.lr_backbone_mult)
    scaler, amp_enabled = _make_scaler(device, args.amp)

    start_epoch = 1
    best_mae = float("inf")
    if args.resume_stage2:
        last_epoch, best_mae = load_checkpoint(args.resume_stage2, model, optimizer, device)
        start_epoch = last_epoch + 1

    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    teacher.set_stage(2)
    teacher.set_sigma_frozen(True)
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    best_path = os.path.join(args.save_dir, "stage2_best.pth")
    latest_path = os.path.join(args.save_dir, "stage2_latest.pth")
    unlabeled_iter = iter(loader_unlabeled)

    for epoch in range(start_epoch, args.epochs_stage2 + 1):
        sigma_train = epoch <= int(args.sigma_unfreeze_epochs)
        model.train()
        model.set_stage(2)
        model.set_sigma_frozen(not sigma_train)
        teacher.eval()
        teacher.set_stage(2)
        teacher.set_sigma_frozen(True)

        sums = {
            "total": 0.0,
            "sup": 0.0,
            "unsup": 0.0,
            "den": 0.0,
            "count": 0.0,
            "patch": 0.0,
            "fg": 0.0,
            "rel": 0.0,
            "u_feat": 0.0,
            "u_rel": 0.0,
            "u_out": 0.0,
            "u_count": 0.0,
            "u_reliability": 0.0,
            "ramp": 0.0,
            "w_out": 0.0,
            "w_rel": 0.0,
            "avar": 0.0,
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

            with torch.no_grad():
                teacher_mean, teacher_features = teacher_mc_forward(
                    teacher, image_u_w, stage=2, mc_samples=args.mc_samples
                )

            optimizer.zero_grad(set_to_none=True)
            with _amp_context(amp_enabled):
                labeled_output = model(image_l, stage=2, is_teacher=False)
                loss_sup, sup_parts = supervised_anchor(labeled_output, gt_density, args)

                student_output = model(image_u_s, stage=2, is_teacher=False)
                loss_unsup, unsup_parts = stage2_reliability_regularizer(
                    student_output=student_output,
                    teacher_output_mean=teacher_mean,
                    teacher_feature_list=teacher_features,
                    epoch=epoch,
                    ramp_epochs=args.rampup_epochs,
                    max_weight=args.max_unsup_weight,
                    relation_grid=tuple(args.relation_grid),
                    w_feature=args.w_u_feat,
                    w_relation=args.w_u_rel,
                    w_output=args.w_u_out,
                    w_count=args.w_u_count,
                    w_reliability=args.w_u_reliability,
                    variance_decay=args.rel_beta,
                )
                loss = loss_sup + loss_unsup

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            ReliabilityGuidedCrowdCounter.update_teacher_ema(teacher, model, ema=args.ema_decay)

            sums["total"] += float(loss.detach().cpu())
            sums["sup"] += float(loss_sup.detach().cpu())
            sums["unsup"] += float(loss_unsup.detach().cpu())
            sums["den"] += sup_parts["L_den"]
            sums["count"] += sup_parts["L_count"]
            sums["patch"] += sup_parts["L_patch"]
            sums["fg"] += sup_parts["L_foreground"]
            sums["rel"] += sup_parts["L_reliability"]
            sums["u_feat"] += unsup_parts["u_feat"]
            sums["u_rel"] += unsup_parts["u_rel"]
            sums["u_out"] += unsup_parts["u_out"]
            sums["u_count"] += unsup_parts["u_count"]
            sums["u_reliability"] += unsup_parts["u_reliability"]
            sums["ramp"] += unsup_parts["ramp"]
            sums["w_out"] += unsup_parts["w_out_mean"]
            sums["w_rel"] += unsup_parts["W_rel_mean"]
            sums["avar"] += unsup_parts["A_var_mean"]
            updates += 1

        if updates == 0:
            raise RuntimeError(f"Stage2 epoch {epoch} produced no finite updates")

        if epoch % args.eval_every != 0 and epoch != args.epochs_stage2:
            continue

        mae, rmse = evaluate(model, loader_val, device, stage=2)
        denom = float(updates)
        print(
            f"Stage2 Epoch {epoch:4d} | loss {sums['total'] / denom:.4f} "
            f"(sup {sums['sup'] / denom:.4f} + unsup {sums['unsup'] / denom:.4f}) | "
            f"MAE {mae:.2f} RMSE {rmse:.2f} | "
            f"Den {sums['den'] / denom:.4f} Count {sums['count'] / denom:.4f} "
            f"Patch {sums['patch'] / denom:.4f} FG {sums['fg'] / denom:.4f} "
            f"Rel {sums['rel'] / denom:.4f} | "
            f"uFeat {sums['u_feat'] / denom:.4f} uRel {sums['u_rel'] / denom:.6f} "
            f"uOut {sums['u_out'] / denom:.6f} uCnt {sums['u_count'] / denom:.4f} "
            f"uR {sums['u_reliability'] / denom:.4f} | "
            f"ramp {sums['ramp'] / denom:.3f} wOut {sums['w_out'] / denom:.3f} "
            f"Wrel {sums['w_rel'] / denom:.3f} Avar {sums['avar'] / denom:.6f} "
            f"sigma_train={int(sigma_train)}"
        )

        if math.isfinite(mae) and mae < best_mae:
            best_mae = mae
            save_checkpoint(best_path, model, optimizer, epoch, best_mae, stage=2, args=args)
        save_checkpoint(latest_path, model, optimizer, epoch, best_mae, stage=2, args=args)

    if not os.path.isfile(best_path):
        raise RuntimeError("Stage2 did not produce a finite validation MAE")
    return best_path


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    loader_labeled, loader_unlabeled, loader_val = build_dataloaders(args)

    model = ReliabilityGuidedCrowdCounter(
        pretrained=False,
        patch_grid=tuple(args.patch_grid),
        relation_grid=tuple(args.relation_grid),
    )
    load_backbone_pretrained(model, args.pretrained_path)
    model = model.to(device)

    # Build lazy heads before optimizer construction and checkpoint loading.
    sample = next(iter(loader_labeled))[0].to(device)
    model.eval()
    with torch.no_grad():
        model(sample, stage=1, is_teacher=True)

    print("=== Stage1: supervised reliability-guided warm-up ===")
    stage1_best = stage1(model, loader_labeled, loader_val, device, args)

    print("=== Stage2: supervised anchor + reliability-weighted mean teacher ===")
    stage2(model, loader_labeled, loader_unlabeled, loader_val, device, args, stage1_best)

    print("Done.")


if __name__ == "__main__":
    main()
