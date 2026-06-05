# test_single_image.py (完善版：诊断 + 可视化密度图 + 参数量/FLOPs + 严格加载)
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from Module.mamba_semcount_v3 import DARPComplexCrowd

# 需要安装：pip install ptflops
from ptflops import get_model_complexity_info

# ------------------ 固定配置 ------------------
MODEL_PATH = "./checkpoint/stage2_best.pth"  # 你的 stage2 权重路径
USE_STAGE = 2
CROP_SIZE = 512
PATCH_GRID = (8, 8)
RELATION_GRID = (8, 8)
PRETRAINED = True  # 必须和训练时一致（True）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TEST_IMAGE_PATH = "/data/lm/dataset/shanghaitech_part_A/test/img/IMG_15.jpg"  # 你的测试图片
OUTPUT_VIS_DIR = "./vis_output"  # 密度图可视化保存目录
# -------------------------------------------------

os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)


def load_model():
    print("=== 实例化模型（lazy init） ===")
    model = DARPComplexCrowd(
        pretrained=PRETRAINED,
        patch_grid=PATCH_GRID,
        relation_grid=RELATION_GRID
    ).to(DEVICE)

    # === 关键：先 trigger lazy build ===
    print("Triggering lazy build with dummy input...")
    dummy = torch.zeros(1, 3, CROP_SIZE, CROP_SIZE).to(DEVICE)  # [B=1, C=3, H, W]
    with torch.no_grad():
        _ = model(dummy, stage=1, is_teacher=True)  # 任意 stage 都行，只要触发 build
    print("Lazy build 完成，模型现在完整")

    print(f"Build 后 keys 数量: {len(list(model.state_dict().keys()))}")

    # === 现在安全加载权重 ===
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)  # 现在应该完美匹配
    print("权重加载成功（strict=True）")

    # === 计算并打印参数量和FLOPs ===
    print("\n=== 模型复杂度统计 ===")
    # 参数量 (M)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"参数量: {params:.2f} M")

    # FLOPs (使用 ptflops，输入尺寸为 CROP_SIZE)
    try:
        flops_str, params_str = get_model_complexity_info(
            model,
            (3, CROP_SIZE, CROP_SIZE),
            as_strings=True,
            print_per_layer_stat=False,  # 可设为 True 查看逐层详情
            verbose=False
        )
        print(f"FLOPs: {flops_str} (输入尺寸 {CROP_SIZE}x{CROP_SIZE})")
        print(f"参数量 (ptflops 计算): {params_str}")
    except Exception as e:
        print("FLOPs 计算失败（可能某些自定义模块不支持）:", e)
        print("建议手动检查或忽略此项")

    # 重新 set_stage 到测试用的
    model.set_stage(USE_STAGE)

    return model


@torch.no_grad()
def predict_and_visualize(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((CROP_SIZE, CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    model.eval()
    if hasattr(model, "set_stage"):
        model.set_stage(USE_STAGE)

    output = model(img_tensor, stage=USE_STAGE, is_teacher=True)

    density_mu = output["density_mu"].cpu().squeeze(0).squeeze(0).numpy()  # [H, W]
    pred_count = density_mu.sum()

    # 可视化密度图
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img_pil)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(density_mu, cmap="jet")
    plt.colorbar(shrink=0.8)
    plt.title(f"Predicted Density Map\nCount: {pred_count:.2f}")
    plt.axis("off")

    base_name = os.path.basename(image_path)
    vis_path = os.path.join(OUTPUT_VIS_DIR, f"vis_{base_name}")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"密度图可视化已保存: {vis_path}")

    return pred_count


def main():
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"图片不存在: {TEST_IMAGE_PATH}")
        return

    model = load_model()

    print(f"\n正在预测: {TEST_IMAGE_PATH}")
    count = predict_and_visualize(model, TEST_IMAGE_PATH)
    print(f"预测人群数量: {count:.2f}")

    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        print(f"权重信息: epoch={ckpt.get('epoch', 'N/A')}, best_mae={ckpt.get('best_mae', 'N/A')}")


if __name__ == "__main__":
    main()