# test_single_image_stage2_diag_vis.py
# 目的：
# 1) 输出密度图 + 计数
# 2) 输出 Stage2 信任门控相关可视化：w_resp / w_unc / mask / w_s
# 3) 使用 GT 点标注生成 patch-wise 计数误差热力图
# 4) 绘制 “平均误差 vs 权重分箱” 曲线（Full/Resp-only/Unc-only），对应你问的第5点

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

try:
    import scipy.io as sio
except Exception as e:
    sio = None
    print("WARN: scipy 未安装，无法读取 .mat GT。请先 pip install scipy。")

from Module.mamba_semcount_v3 import DARPComplexCrowd

# ------------------ 固定配置（按你原脚本风格） ------------------
MODEL_PATH = "./checkpoint/stage2_best.pth"
USE_STAGE = 2
CROP_SIZE = 512
PATCH_GRID = (8, 8)          # 用于 patch-wise 误差展示（建议 8x8 或 16x16）
RELATION_GRID = (8, 8)
PRETRAINED = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TEST_IMAGE_PATH = "/data/lm/dataset/shanghaitech_part_A/test/img/IMG_15.jpg"
OUTPUT_VIS_DIR = "./vis_output_stage2_diag"
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

# Stage2 trust 参数（与你 losses.py 默认一致/接近）
TAU_SIGMA = -2.0
K_SIGMA = 2.0
MASK_SOFT_STRENGTH = 0.5
USE_MASK_PRED = True
EPS = 1e-6
# ---------------------------------------------------------------


# ------------------ 工具：严格加载 + lazy build ------------------
def load_model():
    print("=== 实例化模型（lazy init） ===")
    model = DARPComplexCrowd(
        pretrained=PRETRAINED,
        patch_grid=PATCH_GRID,
        relation_grid=RELATION_GRID
    ).to(DEVICE)

    # trigger lazy build
    print("Triggering lazy build with dummy input...")
    dummy = torch.zeros(1, 3, CROP_SIZE, CROP_SIZE).to(DEVICE)
    with torch.no_grad():
        _ = model(dummy, stage=1, is_teacher=True)
    print("Lazy build 完成")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    print("权重加载成功（strict=True）")

    if hasattr(model, "set_stage"):
        model.set_stage(USE_STAGE)
    model.eval()
    return model


# ------------------ GT：从 ShanghaiTech 的 .mat 中读取点 ------------------
def infer_gt_mat_path(image_path: str):
    """
    ShanghaiTechA 常见结构：
    .../test/img/IMG_15.jpg
    .../test/ground_truth/GT_IMG_15.mat
    """
    base = os.path.basename(image_path).replace(".jpg", "").replace(".png", "")
    test_root = os.path.dirname(os.path.dirname(image_path))  # .../test
    gt_dir = os.path.join(test_root, "ground_truth")
    gt_mat = os.path.join(gt_dir, f"GT_{base}.mat")
    return gt_mat


def _extract_points_from_mat(mat_dict):
    """
    尽量鲁棒地从 .mat 结构里拿到 Nx2 点坐标（x,y）。
    ShanghaiTech 通常是 mat['image_info'][0,0][0,0][0] 形式。
    """
    # 常见路径 1：ShanghaiTech official
    if "image_info" in mat_dict:
        try:
            pts = mat_dict["image_info"][0, 0][0, 0][0]
            pts = np.asarray(pts, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[1] == 2:
                return pts
        except Exception:
            pass

    # 其他可能 key
    for k in ["annPoints", "points", "loc", "annotation"]:
        if k in mat_dict:
            pts = np.asarray(mat_dict[k], dtype=np.float32)
            if pts.ndim == 2 and pts.shape[1] == 2:
                return pts

    # 最后：遍历寻找 Nx2
    for k, v in mat_dict.items():
        try:
            arr = np.asarray(v)
            if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0:
                return arr.astype(np.float32)
        except Exception:
            continue

    return None


def load_gt_points_for_image(image_path: str):
    if sio is None:
        return None

    gt_mat = infer_gt_mat_path(image_path)
    if not os.path.exists(gt_mat):
        print(f"WARN: 未找到 GT mat: {gt_mat}")
        return None

    mat = sio.loadmat(gt_mat)
    pts = _extract_points_from_mat(mat)
    if pts is None:
        print(f"WARN: 解析 GT 失败: {gt_mat}")
        return None

    # MATLAB 点坐标常见为 1-index，这里减 1 影响极小且更合理
    pts = pts - 1.0
    return pts


# ------------------ Patch-wise 计数（GT点 vs 预测密度积分） ------------------
def points_to_patch_counts(points_xy, orig_wh, target_hw, grid=(8, 8)):
    """
    points_xy: Nx2 in (x,y) on original image coordinate (0-based)
    orig_wh: (W0,H0)
    target_hw: (H,W) after resize
    return: grid counts [gh,gw]
    """
    gh, gw = int(grid[0]), int(grid[1])
    W0, H0 = float(orig_wh[0]), float(orig_wh[1])
    H, W = int(target_hw[0]), int(target_hw[1])

    counts = np.zeros((gh, gw), dtype=np.float32)
    if points_xy is None or len(points_xy) == 0:
        return counts

    # scale points to resized coordinate
    sx, sy = W / max(W0, 1.0), H / max(H0, 1.0)
    pts = points_xy.copy()
    pts[:, 0] *= sx
    pts[:, 1] *= sy

    # assign to patch
    ph, pw = H / float(gh), W / float(gw)
    for (x, y) in pts:
        if x < 0 or y < 0 or x >= W or y >= H:
            continue
        cj = int(x / pw)
        ri = int(y / ph)
        cj = min(max(cj, 0), gw - 1)
        ri = min(max(ri, 0), gh - 1)
        counts[ri, cj] += 1.0
    return counts


def density_to_patch_counts(density_hw, grid=(8, 8)):
    """
    density_hw: [H,W] numpy
    return: [gh,gw] patch sums (mass-preserving by definition of sum)
    """
    gh, gw = int(grid[0]), int(grid[1])
    H, W = density_hw.shape
    # 用 torch 做自适应平均池化再乘面积 => patch sum
    d = torch.from_numpy(density_hw).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    pooled = F.adaptive_avg_pool2d(d, (gh, gw))  # [1,1,gh,gw]
    scale = (H / float(gh)) * (W / float(gw))
    patch_sum = (pooled * scale).squeeze().numpy()  # [gh,gw]
    return patch_sum.astype(np.float32)


# ------------------ Stage2 信任权重（与 losses.py 同定义） ------------------
def compute_trust_components(teacher_out, target_hw, tau_sigma=-2.0, k_sigma=2.0, use_mask=True, mask_strength=0.5, eps=1e-6):
    """
    返回：w_resp, w_unc, mask, w_full
    输出均为 torch.Tensor [1,1,H,W] on CPU
    """
    def _resize_like(x, hw):
        if x.shape[-2:] == tuple(hw):
            return x
        return F.interpolate(x, size=hw, mode="bilinear", align_corners=False)

    mu_t = _resize_like(teacher_out["density_mu"].detach(), target_hw)
    ls_t = _resize_like(teacher_out["density_log_sigma"].detach(), target_hw)

    mu_norm = mu_t / (mu_t.amax(dim=[2, 3], keepdim=True).clamp_min(eps))
    w_resp = mu_norm.sqrt().clamp(0.0, 1.0)

    w_unc = torch.sigmoid(-float(k_sigma) * (ls_t - float(tau_sigma))).clamp(0.0, 1.0)

    w = (w_resp * w_unc).clamp(0.0, 1.0)

    mask = None
    if use_mask and ("mask_pred" in teacher_out):
        s = float(mask_strength)
        s = max(0.0, min(1.0, s))
        mask = _resize_like(teacher_out["mask_pred"].detach(), target_hw).clamp(0.0, 1.0)
        w = (w * ((1.0 - s) + s * mask)).clamp(0.0, 1.0)

    return w_resp.cpu(), w_unc.cpu(), (mask.cpu() if mask is not None else None), w.cpu()


# ------------------ 第5点：分箱曲线（权重 vs 误差） ------------------
def binned_curve(weight_grid, error_grid, num_bins=10):
    """
    weight_grid: [gh,gw] in [0,1]
    error_grid:  [gh,gw] >=0
    return: bin_centers, mean_error_per_bin
    """
    w = weight_grid.flatten()
    e = error_grid.flatten()

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    idx = np.digitize(w, bins) - 1
    idx = np.clip(idx, 0, num_bins - 1)

    mean_err = np.full((num_bins,), np.nan, dtype=np.float32)
    for b in range(num_bins):
        m = (idx == b)
        if np.any(m):
            mean_err[b] = float(e[m].mean())
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, mean_err


# ------------------ 可视化工具 ------------------
def _to_np01(t):
    x = t.squeeze().detach().cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    return x

def annotate_grid(ax, grid, fmt="{:.1f}", fontsize=8):
    gh, gw = grid.shape
    for i in range(gh):
        for j in range(gw):
            ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center", color="w", fontsize=fontsize)

def save_figure(path):
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


@torch.no_grad()
def run_single_image_with_diag(model, image_path: str):
    # --- load & preprocess ---
    img_pil = Image.open(image_path).convert("RGB")
    W0, H0 = img_pil.size

    transform = transforms.Compose([
        transforms.Resize((CROP_SIZE, CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    x = transform(img_pil).unsqueeze(0).to(DEVICE)  # [1,3,H,W]

    model.eval()
    if hasattr(model, "set_stage"):
        model.set_stage(USE_STAGE)

    # --- forward (teacher-style output) ---
    out = model(x, stage=USE_STAGE, is_teacher=True)

    # 必要字段检查
    for k in ["density_mu", "density_log_sigma"]:
        if k not in out:
            raise KeyError(f"model output missing key: {k}")

    mu = out["density_mu"].cpu().squeeze(0).squeeze(0).numpy()  # [H,W]
    pred_count = float(mu.sum())

    # --- compute trust maps at mu resolution ---
    w_resp, w_unc, mask, w_full = compute_trust_components(
        out, target_hw=out["density_mu"].shape[-2:],
        tau_sigma=TAU_SIGMA, k_sigma=K_SIGMA,
        use_mask=USE_MASK_PRED, mask_strength=MASK_SOFT_STRENGTH, eps=EPS
    )
    w_resp_np = _to_np01(w_resp)
    w_unc_np = _to_np01(w_unc)
    w_full_np = _to_np01(w_full)
    mask_np = _to_np01(mask) if mask is not None else None

    # --- load GT points & compute patch-wise errors ---
    points = load_gt_points_for_image(image_path)
    gt_patch = None
    pred_patch = None
    abs_err = None
    signed_err = None

    if points is not None:
        gt_patch = points_to_patch_counts(points, orig_wh=(W0, H0), target_hw=(CROP_SIZE, CROP_SIZE), grid=PATCH_GRID)
        pred_patch = density_to_patch_counts(mu, grid=PATCH_GRID)
        signed_err = (pred_patch - gt_patch)
        abs_err = np.abs(signed_err)

    # --- pooled trust to patch grid (for curve) ---
    # 用 full/resp/unc 三种权重都做曲线，解释单一线索不稳
    w_full_grid = F.adaptive_avg_pool2d(torch.from_numpy(w_full_np).unsqueeze(0).unsqueeze(0), PATCH_GRID).squeeze().numpy()
    w_resp_grid = F.adaptive_avg_pool2d(torch.from_numpy(w_resp_np).unsqueeze(0).unsqueeze(0), PATCH_GRID).squeeze().numpy()
    w_unc_grid  = F.adaptive_avg_pool2d(torch.from_numpy(w_unc_np).unsqueeze(0).unsqueeze(0),  PATCH_GRID).squeeze().numpy()

    # ----------------- Figure 1: 基本预测图 -----------------
    fig1 = os.path.join(OUTPUT_VIS_DIR, f"pred_density_{os.path.basename(image_path)}.png")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_pil.resize((CROP_SIZE, CROP_SIZE)))
    plt.title("Input (resized)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(np.log1p(mu), cmap="magma")  # log 增强可读性
    plt.title(f"log(1+mu)\nCount={pred_count:.1f}")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 3, 3)
    if abs_err is None:
        plt.text(0.1, 0.5, "GT not found\n(skip error vis)", fontsize=12)
        plt.axis("off")
    else:
        plt.imshow(abs_err, cmap="inferno")
        plt.title("Patch abs error |c_pred-c_gt|")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
        annotate_grid(plt.gca(), abs_err, fmt="{:.0f}", fontsize=7)
    save_figure(fig1)

    # ----------------- Figure 2: 信任门控空间图（第5点的“空间部分”） -----------------
    fig2 = os.path.join(OUTPUT_VIS_DIR, f"trust_maps_{os.path.basename(image_path)}.png")
    plt.figure(figsize=(14, 7))

    plt.subplot(2, 3, 1)
    plt.imshow(w_resp_np, vmin=0, vmax=1, cmap="viridis")
    plt.title("w_resp (sqrt norm mu)")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2, 3, 2)
    plt.imshow(w_unc_np, vmin=0, vmax=1, cmap="viridis")
    plt.title("w_unc (sigmoid from log_sigma)")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2, 3, 3)
    if mask_np is None:
        plt.text(0.1, 0.5, "mask_pred not available", fontsize=12)
        plt.axis("off")
    else:
        plt.imshow(mask_np, vmin=0, vmax=1, cmap="viridis")
        plt.title("mask_pred (teacher)")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2, 3, 4)
    plt.imshow(w_full_np, vmin=0, vmax=1, cmap="viridis")
    plt.title("w_s (Full trust)")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2, 3, 5)
    if abs_err is None:
        plt.text(0.1, 0.5, "GT not found\n(skip error map)", fontsize=12)
        plt.axis("off")
    else:
        # 把 patch abs error 上采样到 512x512 方便空间对齐展示
        abs_err_up = F.interpolate(
            torch.from_numpy(abs_err).unsqueeze(0).unsqueeze(0),
            size=(CROP_SIZE, CROP_SIZE),
            mode="nearest"
        ).squeeze().numpy()
        plt.imshow(abs_err_up, cmap="inferno")
        plt.title("Abs patch error (upsampled)")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2, 3, 6)
    if signed_err is None:
        plt.text(0.1, 0.5, "GT not found\n(skip signed error)", fontsize=12)
        plt.axis("off")
    else:
        vmax = max(1.0, float(np.max(np.abs(signed_err))))
        plt.imshow(signed_err, cmap="bwr", vmin=-vmax, vmax=vmax)
        plt.title("Patch signed error (pred-gt)")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
        annotate_grid(plt.gca(), signed_err, fmt="{:.0f}", fontsize=7)

    save_figure(fig2)

    # ----------------- Figure 3: 分箱曲线（第5点的“统计部分”） -----------------
    fig3 = os.path.join(OUTPUT_VIS_DIR, f"binned_curve_{os.path.basename(image_path)}.png")
    plt.figure(figsize=(6, 4))
    if abs_err is None:
        plt.text(0.1, 0.5, "GT not found\n(skip binned curve)", fontsize=12)
        plt.axis("off")
        save_figure(fig3)
    else:
        centers, y_full = binned_curve(w_full_grid, abs_err, num_bins=10)
        _,       y_resp = binned_curve(w_resp_grid, abs_err, num_bins=10)
        _,       y_unc  = binned_curve(w_unc_grid,  abs_err, num_bins=10)

        plt.plot(centers, y_full, marker="o", label="Full (w_resp*w_unc*mask)")
        plt.plot(centers, y_resp, marker="o", label="Resp-only")
        plt.plot(centers, y_unc,  marker="o", label="Unc-only")
        plt.xlabel("Weight bin center")
        plt.ylabel("Mean abs patch error")
        plt.title("Error vs weight bins (patch-level)")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend()
        save_figure(fig3)

    print(f"Pred count = {pred_count:.2f}")
    if abs_err is not None:
        print(f"Patch MAE proxy (mean abs patch error) = {abs_err.mean():.3f}")

    return pred_count


def main():
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"图片不存在: {TEST_IMAGE_PATH}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"权重不存在: {MODEL_PATH}")
        return

    model = load_model()
    run_single_image_with_diag(model, TEST_IMAGE_PATH)
    print(f"All visualizations saved to: {OUTPUT_VIS_DIR}")


if __name__ == "__main__":
    main()

