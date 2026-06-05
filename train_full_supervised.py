"""Full-supervision baseline for DARPComplexCrowd.

All training images use their ground-truth density maps. Stage 1 trains the
stable fusion model; Stage 2 enables the conditional and relation modules but
continues to use supervised losses only. No teacher, pseudo-label, or
unlabeled consistency loss is used.
"""

import argparse
import math
import os
import random

import numpy as np
import torch

from dataset.loaddata import build_dataloaders
from losses import supervised_stage1_loss, supervised_stage2_anchor
from Module.mamba_semcount_v3 import DARPComplexCrowd


def parse_args():
    parser = argparse.ArgumentParser(description="Train DARPComplexCrowd with 100% supervision")

    parser.add_argument("--data_root", type=str, default="/data/LM/Dataset")
    parser.add_argument("--dataset_name", type=str, default="SHA")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--downsample_ratio", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs_stage1", type=int, default=2000)
    parser.add_argument("--epochs_stage2", type=int, default=800)
    parser.add_argument("--eval_every", type=int, default=10)

    parser.add_argument("--lr_stage1", type=float, default=1e-5)
    parser.add_argument("--lr_stage2", type=float, default=5e-6)
    parser.add_argument("--lr_backbone_mult", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--w_den", type=float, default=1.0)
    parser.add_argument("--w_patch", type=float, default=0.05)
    parser.add_argument("--w_mask", type=float, default=0.2)
    parser.add_argument("--w_low", type=float, default=0.3)
    parser.add_argument("--w_res", type=float, default=0.2)
    parser.add_argument("--w_res_neg", type=float, default=0.01)
    parser.add_argument("--patch_grid", type=int, nargs=2, default=[8, 8])
    parser.add_argument("--relation_grid", type=int, nargs=2, default=[8, 8])

    parser.add_argument(
        "--freeze_sigma_after",
        type=int,
        default=0,
        help="Freeze Stage-2 sigma heads after this epoch; 0 keeps them trainable.",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="./weights/mambavision_base_21k.pth.tar",
        help="Local MambaVision-B checkpoint. Use an empty string for random initialization.",
    )
    parser.add_argument("--save_dir", type=str, default="./checkpoint_full_supervised")
    parser.add_argument("--gpu_id", type=str, default="0")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Pretrained checkpoint must be a state-dict-like mapping")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    return state_dict


def load_backbone_pretrained(model, path):
    if not path:
        print("Backbone initialization: random")
        return
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Backbone checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    incompatible = model.backbone.load_state_dict(_extract_state_dict(checkpoint), strict=False)
    print(
        "Backbone initialization: "
        f"{path} | missing={len(incompatible.missing_keys)} "
        f"unexpected={len(incompatible.unexpected_keys)}"
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


@torch.no_grad()
def evaluate(model, loader_val, device, stage):
    model.eval()
    absolute_error = 0.0
    squared_error = 0.0
    sample_count = 0

    for batch in loader_val:
        image, gt_count = batch[0], batch[1]
        image = image.to(device, non_blocking=True)
        gt_count = gt_count.to(device, non_blocking=True).float().view(-1)

        output = model(image, stage=stage, is_teacher=True)
        pred_count = output["density_mu"].sum(dim=[1, 2, 3]).float().view(-1)
        error = pred_count - gt_count

        absolute_error += error.abs().sum().item()
        squared_error += error.square().sum().item()
        sample_count += gt_count.numel()

    mae = absolute_error / max(1, sample_count)
    rmse = math.sqrt(squared_error / max(1, sample_count))
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


def load_model_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint


def supervised_loss(output, gt_density, stage, args):
    loss_fn = supervised_stage1_loss if stage == 1 else supervised_stage2_anchor
    return loss_fn(
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


def train_stage(model, loader_train, loader_val, device, args, stage, epochs, learning_rate):
    model.train()
    model.set_stage(stage)
    if stage == 2:
        model.set_sigma_frozen(False)

    optimizer = build_optimizer(
        model,
        base_lr=learning_rate,
        weight_decay=args.weight_decay,
        backbone_mult=args.lr_backbone_mult,
    )

    best_mae = float("inf")
    best_path = os.path.join(args.save_dir, f"stage{stage}_best.pth")
    latest_path = os.path.join(args.save_dir, f"stage{stage}_latest.pth")

    for epoch in range(1, epochs + 1):
        if stage == 2:
            freeze_sigma = args.freeze_sigma_after > 0 and epoch > args.freeze_sigma_after
            model.set_sigma_frozen(freeze_sigma)
        else:
            freeze_sigma = False

        model.train()
        model.set_stage(stage)

        sums = {
            "loss_total": 0.0,
            "L_den": 0.0,
            "L_patch": 0.0,
            "L_mask": 0.0,
            "L_low": 0.0,
            "L_res": 0.0,
            "L_res_neg": 0.0,
            "L_sigma": 0.0,
        }
        updates = 0

        for image, gt_density in loader_train:
            image = image.to(device, non_blocking=True)
            gt_density = gt_density.to(device, non_blocking=True)

            output = model(image, stage=stage, is_teacher=False)
            loss, parts = supervised_loss(output, gt_density, stage, args)
            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            for key in sums:
                sums[key] += parts[key]
            updates += 1

        if updates == 0:
            raise RuntimeError(f"Stage {stage} epoch {epoch} produced no finite updates")

        should_evaluate = epoch % args.eval_every == 0 or epoch == epochs
        if not should_evaluate:
            continue

        mae, rmse = evaluate(model, loader_val, device, stage)
        denom = float(updates)
        print(
            f"FullSup Stage{stage} Epoch {epoch:4d} | "
            f"loss {sums['loss_total'] / denom:.4f} | MAE {mae:.2f} RMSE {rmse:.2f} | "
            f"Den {sums['L_den'] / denom:.4f} Patch {sums['L_patch'] / denom:.6f} "
            f"Mask {sums['L_mask'] / denom:.4f} Low {sums['L_low'] / denom:.4f} "
            f"Res {sums['L_res'] / denom:.4f} ResNeg {sums['L_res_neg'] / denom:.4f} "
            f"Sigma {sums['L_sigma'] / denom:.4f} sigma_frozen={int(freeze_sigma)}"
        )

        if math.isfinite(mae) and mae < best_mae:
            best_mae = mae
            save_checkpoint(best_path, model, optimizer, epoch, best_mae, stage, args)
        save_checkpoint(latest_path, model, optimizer, epoch, best_mae, stage, args)

    if not os.path.isfile(best_path):
        raise RuntimeError(f"Stage {stage} did not produce a finite validation MAE")
    return best_path


def main():
    args = parse_args()
    args.labeled_ratio = 1.0

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    loader_train, _, loader_val = build_dataloaders(args)
    if len(loader_train.dataset) == 0:
        raise RuntimeError("The full-supervision training set is empty")
    print(f"Full supervision: {len(loader_train.dataset)} labeled training images")

    model = DARPComplexCrowd(
        pretrained=False,
        patch_grid=tuple(args.patch_grid),
        relation_grid=tuple(args.relation_grid),
    )
    load_backbone_pretrained(model, args.pretrained_path)
    model = model.to(device)

    # Build lazy decoder modules before creating optimizers or loading full checkpoints.
    sample_image = next(iter(loader_train))[0].to(device)
    model.eval()
    with torch.no_grad():
        model(sample_image, stage=1, is_teacher=True)

    print("=== Full supervision Stage1: stable fusion and supervised auxiliary heads ===")
    stage1_best = train_stage(
        model,
        loader_train,
        loader_val,
        device,
        args,
        stage=1,
        epochs=args.epochs_stage1,
        learning_rate=args.lr_stage1,
    )

    load_model_checkpoint(stage1_best, model, device)
    print("=== Full supervision Stage2: relation modules enabled, no unlabeled losses ===")
    train_stage(
        model,
        loader_train,
        loader_val,
        device,
        args,
        stage=2,
        epochs=args.epochs_stage2,
        learning_rate=args.lr_stage2,
    )

    print("Done.")


if __name__ == "__main__":
    main()
