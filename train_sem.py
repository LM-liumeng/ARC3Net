# train.py
import os
import math
import random
import argparse
import numpy as np
import copy

import torch
import torch.nn.functional as F

from dataset.loaddata import build_dataloaders
from Module.mamba_semcount_v3 import DARPComplexCrowd  # ensure this points to your updated model.py
from losses import (
    supervised_stage1_loss,
    supervised_stage2_anchor,
    stage2_unlabeled_regularizer_mcvar,
)


def parse_args():
    p = argparse.ArgumentParser()

    # data / train
    p.add_argument('--data_root', type=str, default='/data/LM/Dataset')
    p.add_argument('--dataset_name', type=str, default='SHA')
    p.add_argument('--labeled_ratio', type=float, default=0.4)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--crop_size', type=int, default=512)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--downsample_ratio', type=int, default=4)

    # epochs
    p.add_argument('--epochs_stage1', type=int, default=2000)
    p.add_argument('--epochs_stage2', type=int, default=800)
    p.add_argument('--eval_every', type=int, default=10)

    # learning rates
    p.add_argument('--lr_stage1', type=float, default=1e-5)
    p.add_argument('--lr_stage2', type=float, default=5e-6)
    p.add_argument('--lr_backbone_mult', type=float, default=0.5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--grad_clip', type=float, default=5.0)

    # loss weights (supervised anchor)
    p.add_argument('--w_den', type=float, default=1.0)
    p.add_argument('--w_patch', type=float, default=0.05)
    p.add_argument('--w_mask', type=float, default=0.2)
    p.add_argument('--w_low', type=float, default=0.3)
    p.add_argument('--w_res', type=float, default=0.2)
    p.add_argument('--w_res_neg', type=float, default=0.01)
    p.add_argument('--patch_grid', type=int, nargs=2, default=[8, 8])

    # stage2 unlabeled regularizers
    p.add_argument('--rampup_epochs', type=int, default=200)
    p.add_argument('--max_unsup_weight', type=float, default=0.15)
    p.add_argument('--w_u_feat', type=float, default=1.0)
    p.add_argument('--w_u_rel', type=float, default=0.2)
    p.add_argument('--w_u_out', type=float, default=0.05)
    p.add_argument('--ema_decay', type=float, default=0.999)

    # uncertainty-guided trust-map params
    p.add_argument('--tau_sigma', type=float, default=-2.0)
    p.add_argument('--k_sigma', type=float, default=2.0)
    p.add_argument('--use_mask_pred', action='store_true')
    p.set_defaults(use_mask_pred=True)

    # NEW: mask soft prior strength in trust-map (0=no mask prior, 1=hard multiply)
    p.add_argument('--mask_soft_strength', type=float, default=0.5)

    # NEW: sigma unfreeze epochs in stage2 (first N epochs train log_sigma, then freeze)
    p.add_argument('--sigma_unfreeze_epochs', type=int, default=300)

    # MC relation variance
    p.add_argument('--mc_samples', type=int, default=2)
    p.add_argument('--rel_beta', type=float, default=5.0)
    p.add_argument('--relation_grid', type=int, nargs=2, default=[8, 8])

    # save
    p.add_argument('--save_dir', type=str, default='./checkpoint')
    p.add_argument('--gpu_id', type=str, default='1')

    return p.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model, loader_val, device):
    model.eval()
    abs_err_sum, sq_err_sum, n = 0.0, 0.0, 0
    for batch in loader_val:
        img, gt_count = batch[0], batch[1]
        img = img.to(device, non_blocking=True)
        gt = gt_count.to(device, non_blocking=True).float().view(-1)

        out = model(img, stage=2, is_teacher=True)
        pred_cnt = out["density_mu"].sum(dim=[1, 2, 3]).detach().float().view(-1)

        diff = pred_cnt - gt
        abs_err_sum += diff.abs().sum().item()
        sq_err_sum += (diff * diff).sum().item()
        n += gt.numel()

    mae = abs_err_sum / max(1, n)
    rmse = math.sqrt(sq_err_sum / max(1, n))
    return mae, rmse


def save_ckpt(path, model, optimizer, epoch, best_mae):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_mae": float(best_mae),
    }, path)


def load_ckpt(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_mae", 1e9))


def build_optimizer(model, base_lr: float, wd: float, backbone_mult: float):
    bb_params, hd_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            bb_params.append(p)
        else:
            hd_params.append(p)

    opt = torch.optim.AdamW(
        [
            {"params": bb_params, "lr": base_lr * float(backbone_mult)},
            {"params": hd_params, "lr": base_lr},
        ],
        lr=base_lr,
        weight_decay=wd
    )
    return opt


@torch.no_grad()
def teacher_mc_forward(teacher, img, stage: int, M: int):
    """
    M stochastic forwards to estimate relation variance.
    If backbone has droppath/dropout, teacher.train() enables stochasticity.
    Returns:
      out_mean dict with density_mu/log_sigma/mask_pred/feat_cons
      feat_list list of feat_cons for MC affinity var
    """
    prev_mode = teacher.training
    teacher.train()  # enable stochasticity if exists

    outs = []
    feat_list = []
    for _ in range(int(M)):
        o = teacher(img, stage=stage, is_teacher=True)
        outs.append(o)
        feat_list.append(o["feat_cons"].detach())

    teacher.train(prev_mode)

    def mean_stack(key):
        xs = [o[key].detach() for o in outs]
        return torch.stack(xs, dim=0).mean(dim=0)

    out_mean = {
        "density_mu": mean_stack("density_mu"),
        "density_log_sigma": mean_stack("density_log_sigma"),
        "mask_pred": mean_stack("mask_pred"),
        "feat_cons": mean_stack("feat_cons"),
    }
    return out_mean, feat_list


# =========================================================
# Stage1
# =========================================================
def stage1(model, loader_l, loader_val, device, args):
    model.train()
    model.set_stage(1)

    optimizer = build_optimizer(model, args.lr_stage1, args.weight_decay, args.lr_backbone_mult)

    best_mae = float("inf")
    save_best = os.path.join(args.save_dir, "stage1_best.pth")

    for epoch in range(1, args.epochs_stage1 + 1):
        model.train()
        model.set_stage(1)

        sum_loss = sum_den = sum_patch = sum_mask = 0.0
        sum_low = sum_res = 0.0
        n_upd = 0

        for img, gt_density in loader_l:
            img = img.to(device, non_blocking=True)
            gt_density = gt_density.to(device, non_blocking=True)

            out = model(img, stage=1, is_teacher=False)

            loss, parts = supervised_stage1_loss(
                out, gt_density,
                patch_grid=tuple(args.patch_grid),
                w_den=args.w_den,
                w_patch=args.w_patch,
                w_mask=args.w_mask,
                w_low=args.w_low,
                w_res=args.w_res,
                w_res_neg=args.w_res_neg
            )

            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            sum_loss += parts["loss_total"]
            sum_den += parts["L_den"]
            sum_patch += parts["L_patch"]
            sum_mask += parts["L_mask"]
            sum_low += parts["L_low"]
            sum_res += parts["L_res"]
            n_upd += 1

        if epoch % args.eval_every == 0:
            mae, rmse = evaluate(model, loader_val, device)

            den = max(1, n_upd)
            print(
                f"Stage1 Epoch {epoch:4d} | loss {(sum_loss/den):.4f} | MAE {mae:.2f} RMSE {rmse:.2f} | "
                f"Den {(sum_den/den):.4f} Patch {(sum_patch/den):.6f} Mask {(sum_mask/den):.4f} | "
                f"Low {(sum_low/den):.4f} Res {(sum_res/den):.4f}"
            )

            if math.isfinite(mae) and mae < best_mae:
                best_mae = mae
                save_ckpt(save_best, model, optimizer, epoch, best_mae)

    return save_best


# =========================================================
# Stage2
# =========================================================
def stage2(model, loader_l, loader_u, loader_val, device, args, stage1_best_path: str):
    if os.path.exists(stage1_best_path):
        load_ckpt(stage1_best_path, model, optimizer=None, device=device)

    # Stage2 init: ensure sigma is UNFROZEN at start so optimizer includes it
    model.train()
    model.set_stage(2)
    model.set_sigma_frozen(False)

    optimizer = build_optimizer(model, args.lr_stage2, args.weight_decay, args.lr_backbone_mult)

    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    teacher.set_stage(2)
    teacher.set_sigma_frozen(True)  # teacher is frozen anyway, keep consistent
    for p in teacher.parameters():
        p.requires_grad = False

    best_mae = float("inf")
    save_best = os.path.join(args.save_dir, "stage2_best.pth")

    iter_u = iter(loader_u)

    for epoch in range(1, args.epochs_stage2 + 1):
        # NEW: sigma schedule (first N epochs unfrozen, then frozen)
        sigma_train = (int(epoch) <= int(args.sigma_unfreeze_epochs))
        model.set_sigma_frozen(not sigma_train)

        model.train()
        model.set_stage(2)
        teacher.eval()
        teacher.set_stage(2)

        sum_total = sum_sup = sum_unsup = 0.0
        u_feat_sum = u_rel_sum = u_out_sum = 0.0
        ramp_sum = wfeat_sum = wout_sum = 0.0
        Wrel_sum = Avar_sum = 0.0
        n_upd = 0

        for img_l, gt_density in loader_l:
            img_l = img_l.to(device, non_blocking=True)
            gt_density = gt_density.to(device, non_blocking=True)

            try:
                img_u_w, img_u_s = next(iter_u)
            except StopIteration:
                iter_u = iter(loader_u)
                img_u_w, img_u_s = next(iter_u)

            img_u_w = img_u_w.to(device, non_blocking=True)
            img_u_s = img_u_s.to(device, non_blocking=True)

            # supervised anchor
            out_l = model(img_l, stage=2, is_teacher=False)
            loss_sup, _ = supervised_stage2_anchor(
                out_l, gt_density,
                patch_grid=tuple(args.patch_grid),
                w_den=args.w_den,
                w_patch=args.w_patch,
                w_mask=args.w_mask,
                w_low=args.w_low,
                w_res=args.w_res,
                w_res_neg=args.w_res_neg
            )
            if not torch.isfinite(loss_sup):
                continue

            # teacher MC forward
            with torch.no_grad():
                teacher_out_mean, teacher_feat_list = teacher_mc_forward(
                    teacher, img_u_w, stage=2, M=args.mc_samples
                )

            # student forward on strong view
            out_s = model(img_u_s, stage=2, is_teacher=False)

            # weighted + mcvar unlabeled regularizer (uses feat_cons)
            loss_u, u_parts = stage2_unlabeled_regularizer_mcvar(
                student_out=out_s,
                teacher_out_mean=teacher_out_mean,
                teacher_feat_list=teacher_feat_list,
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
                mask_soft_strength=args.mask_soft_strength,  # NEW
                rel_beta=args.rel_beta,
            )

            loss = loss_sup + loss_u
            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            DARPComplexCrowd.update_teacher_ema(teacher, model, ema=args.ema_decay)

            sum_total += float(loss.detach().cpu())
            sum_sup += float(loss_sup.detach().cpu())
            sum_unsup += float(loss_u.detach().cpu())

            u_feat_sum += u_parts["u_feat"]
            u_rel_sum += u_parts["u_rel"]
            u_out_sum += u_parts["u_out"]
            ramp_sum += u_parts["ramp"]
            wfeat_sum += u_parts["w_feat_mean"]
            wout_sum += u_parts["w_out_mean"]
            Wrel_sum += u_parts["W_rel_mean"]
            Avar_sum += u_parts["A_var_mean"]
            n_upd += 1

        if epoch % args.eval_every == 0:
            mae, rmse = evaluate(model, loader_val, device)

            den = max(1, n_upd)
            print(
                f"Stage2 Epoch {epoch:4d} | "
                f"loss {(sum_total/den):.4f} (sup {(sum_sup/den):.4f} + unsup {(sum_unsup/den):.4f}) | "
                f"MAE {mae:.2f} RMSE {rmse:.2f} | "
                f"ramp {(ramp_sum/den):.3f} | "
                f"u_feat {(u_feat_sum/den):.4f} u_rel {(u_rel_sum/den):.4f} u_out {(u_out_sum/den):.6f} | "
                f"w_feat {(wfeat_sum/den):.3f} w_out {(wout_sum/den):.3f} | "
                f"Wrel {(Wrel_sum/den):.3f} Avar {(Avar_sum/den):.6f} | "
                f"sigma_train {int(sigma_train)}"
            )

            if math.isfinite(mae) and mae < best_mae:
                best_mae = mae
                save_ckpt(save_best, model, optimizer, epoch, best_mae)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    loader_l, loader_u, loader_val = build_dataloaders(args)

    model = DARPComplexCrowd(pretrained=True, patch_grid=tuple(args.patch_grid), relation_grid=tuple(args.relation_grid)).to(device)

    # trigger lazy build
    sample = next(iter(loader_l))
    with torch.no_grad():
        _ = model(sample[0].to(device), stage=1, is_teacher=True)

    print("=== Stage1: Supervised anchor (uncertainty NLL + patch + GT-mask + band deep sup) ===")
    stage1_best = stage1(model, loader_l, loader_val, device, args)

    print("=== Stage2: Calibrated Mean Teacher (weighted feature + weighted relation (MC-var) + weighted output) ===")
    stage2(model, loader_l, loader_u, loader_val, device, args, stage1_best_path=stage1_best)

    print("Done.")


if __name__ == "__main__":
    main()
