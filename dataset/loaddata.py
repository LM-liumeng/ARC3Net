import os
import random
import numpy as np
import cv2
import h5py
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms


# ==============================================================================
# Dataset registry
# ==============================================================================
class DatasetNames:
    SHA = "SHA"
    SHB = "SHB"
    QNRF = "qnrf"
    NWPU = "nwpu"


DATASET_PATHS = {
    DatasetNames.SHA: {
        "train_img": "shanghaitech_part_A/train/img",
        "train_gt":  "shanghaitech_part_A/train/new_data",
        "val_img":   "shanghaitech_part_A/test/img",
        "val_gt":    "shanghaitech_part_A/test/new_data",
    },
    DatasetNames.SHB: {
        "train_img": "shanghaitech_part_B/train/img",
        "train_gt":  "shanghaitech_part_B/train/new_data",
        "val_img":   "shanghaitech_part_B/test/img",
        "val_gt":    "shanghaitech_part_B/test/new_data",
    },
    DatasetNames.QNRF: {
        "train_img": "UCF-QNRF/train/img",
        "train_gt":  "UCF-QNRF/train/new_data",
        "val_img":   "UCF-QNRF/test/img",
        "val_gt":    "UCF-QNRF/test/new_data",
    },
    DatasetNames.NWPU: {
        "train_img": "NWPU/train/img",
        "train_gt":  "NWPU/train/new_data",
        "val_img":   "NWPU/test/img",
        "val_gt":    "NWPU/test/new_data",
    },
}


# ==============================================================================
# I/O
# ==============================================================================
def load_image(path: str) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # 关键：按 EXIF 纠正方向，可能会交换 W/H
    return img.convert("RGB")


def load_density(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "density" in f.keys():
            k = "density"
        elif "image" in f.keys():
            k = "image"
        else:
            raise KeyError(f"h5 file has no 'density' or 'image' key: {path}")
        gt = np.array(f[k], dtype=np.float32, copy=True)

    gt = np.squeeze(gt)
    if gt.ndim != 2:
        raise RuntimeError(f"GT density must be 2D, got {gt.shape} from {path}")
    gt = np.clip(gt, 0.0, None)
    return np.ascontiguousarray(gt, dtype=np.float32)


# ==============================================================================
# Density align (preserve count)
# ==============================================================================
def _mass_preserve_resize_density(gt: np.ndarray, target_hw, eps: float = 1e-8) -> np.ndarray:
    th, tw = int(target_hw[0]), int(target_hw[1])
    gh, gw = gt.shape
    if (gh, gw) == (th, tw):
        return np.ascontiguousarray(gt, dtype=np.float32)

    s0 = float(gt.sum())
    interp = cv2.INTER_AREA if (th < gh and tw < gw) else cv2.INTER_LINEAR
    out = cv2.resize(gt, (tw, th), interpolation=interp).astype(np.float32, copy=False)

    s1 = float(out.sum())
    if s1 < eps:
        return np.zeros((th, tw), dtype=np.float32)

    out = out * (s0 / (s1 + eps))
    out = np.clip(out, 0.0, None)
    return np.ascontiguousarray(out, dtype=np.float32)


def align_density_to_image(gt: np.ndarray, H: int, W: int, pad_tol: int = 2, scale_tol: float = 0.03) -> np.ndarray:
    gt = np.ascontiguousarray(gt, dtype=np.float32)
    gh, gw = gt.shape

    # 1) exact match
    if (gh, gw) == (H, W):
        return gt

    # 2) swapped match: gt is (W, H)
    if (gh, gw) == (W, H):
        gt = gt.T  # transpose to (H, W), sum preserved
        return np.ascontiguousarray(gt, dtype=np.float32)

    # 3) tiny pad case
    if gh <= H and gw <= W and (H - gh) <= pad_tol and (W - gw) <= pad_tol:
        out = np.zeros((H, W), dtype=np.float32)
        out[:gh, :gw] = gt
        return out

    # 4) near-uniform scaling
    sh = H / float(gh)
    sw = W / float(gw)
    if abs(sh - sw) <= scale_tol:
        return _mass_preserve_resize_density(gt, (H, W))

    raise RuntimeError(f"[GT-ALIGN-ERROR] density {(gh, gw)} != image {(H, W)}")



# ==============================================================================
# Fixed crop + block-sum downsample
# ==============================================================================
class FixedPairedCropDownsample:
    def __init__(self, crop_size: int, downsample: int):
        self.crop_size = int(crop_size)
        self.ds = int(downsample)
        if self.ds < 1:
            raise ValueError("downsample must be >= 1")
        if self.crop_size % self.ds != 0:
            raise ValueError(f"crop_size({self.crop_size}) must be divisible by downsample({self.ds})")

    def _pad_to_min_crop(self, img: Image.Image, den: np.ndarray):
        W, H = img.size
        pad_w = max(0, self.crop_size - W)
        pad_h = max(0, self.crop_size - H)
        if pad_w > 0 or pad_h > 0:
            img = TF.pad(img, (0, 0, pad_w, pad_h), fill=0)
            den = np.pad(den, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
        return img, np.ascontiguousarray(den, dtype=np.float32)

    def __call__(self, img: Image.Image, den: np.ndarray):
        img, den = self._pad_to_min_crop(img, den)

        W, H = img.size
        cs = self.crop_size

        i = random.randint(0, H - cs)
        j = random.randint(0, W - cs)

        img_crop = TF.crop(img, i, j, cs, cs)
        den_crop = den[i:i + cs, j:j + cs]

        if self.ds == 1:
            den_down = den_crop
        else:
            den_down = den_crop.reshape(cs // self.ds, self.ds, cs // self.ds, self.ds).sum(axis=(1, 3))

        den_down = np.clip(den_down, 0.0, None)
        return img_crop, np.ascontiguousarray(den_down, dtype=np.float32)


def pad_to_min_crop(img: Image.Image, crop_size: int) -> Image.Image:
    W, H = img.size
    pad_w = max(0, crop_size - W)
    pad_h = max(0, crop_size - H)
    if pad_w > 0 or pad_h > 0:
        img = TF.pad(img, (0, 0, pad_w, pad_h), fill=0)
    return img


# ==============================================================================
# Dataset
# ==============================================================================
class GeneralCrowdDataset(Dataset):
    def __init__(self,
                 data_root,
                 dataset_name,
                 split="train",
                 labeled=True,
                 crop_size=512,
                 downsample_ratio=4,
                 weak_pil=None,
                 strong_pil=None,
                 to_tensor_norm=None,
                 strong_tensor=None,
                 file_list=None,
                 seed=42):
        super().__init__()
        self.split = split
        self.labeled = bool(labeled)

        self.crop_size = int(crop_size)
        self.downsample_ratio = int(downsample_ratio)

        self.weak_pil = weak_pil
        self.strong_pil = strong_pil
        self.to_tensor_norm = to_tensor_norm
        self.strong_tensor = strong_tensor

        self.cropper = FixedPairedCropDownsample(self.crop_size, self.downsample_ratio)

        paths = DATASET_PATHS[dataset_name]
        if split == "train":
            self.img_dir = os.path.join(data_root, paths["train_img"])
            self.gt_dir = os.path.join(data_root, paths["train_gt"])
        else:
            self.img_dir = os.path.join(data_root, paths["val_img"])
            self.gt_dir = os.path.join(data_root, paths["val_gt"])

        if file_list is None:
            self.img_files = sorted([
                os.path.join(self.img_dir, x) for x in os.listdir(self.img_dir)
                if x.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
        else:
            self.img_files = list(file_list)

        self.rng = random.Random(int(seed))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = load_image(img_path)

        # ---------------- TRAIN ----------------
        if self.split == "train":
            # labeled
            if self.labeled:
                basename = os.path.basename(img_path)
                gt_path = os.path.join(self.gt_dir, os.path.splitext(basename)[0] + ".h5")
                gt = load_density(gt_path)

                W, H = img.size
                gt = align_density_to_image(gt, H, W)

                img_crop, gt_down = self.cropper(img, gt)

                if random.random() > 0.5:
                    img_crop = TF.hflip(img_crop)
                    gt_down = np.ascontiguousarray(np.fliplr(gt_down), dtype=np.float32)

                img_t = self.to_tensor_norm(img_crop) if self.to_tensor_norm else transforms.ToTensor()(img_crop)
                gt_t = torch.from_numpy(gt_down).unsqueeze(0).clone()

                return img_t, gt_t

            # unlabeled
            img = pad_to_min_crop(img, self.crop_size)

            W, H = img.size
            cs = self.crop_size
            i = random.randint(0, H - cs)
            j = random.randint(0, W - cs)
            img_crop = TF.crop(img, i, j, cs, cs)

            if random.random() > 0.5:
                img_crop = TF.hflip(img_crop)

            img_w = img_crop
            img_s = img_crop

            if self.weak_pil is not None:
                img_w = self.weak_pil(img_w)

            if self.strong_pil is not None:
                img_s = self.strong_pil(img_s)

            img_w = self.to_tensor_norm(img_w) if self.to_tensor_norm else transforms.ToTensor()(img_w)
            img_s = self.to_tensor_norm(img_s) if self.to_tensor_norm else transforms.ToTensor()(img_s)

            if self.strong_tensor is not None:
                img_s = self.strong_tensor(img_s)

            return img_w, img_s

        # ---------------- VAL ----------------
        basename = os.path.basename(img_path)
        gt_path = os.path.join(self.gt_dir, os.path.splitext(basename)[0] + ".h5")
        gt = load_density(gt_path)

        W, H = img.size
        gt = align_density_to_image(gt, H, W)

        pad_h = 1 if (H % 2 != 0) else 0
        pad_w = 1 if (W % 2 != 0) else 0
        if pad_h > 0 or pad_w > 0:
            img = TF.pad(img, (0, 0, pad_w, pad_h), fill=0)
            gt = np.pad(gt, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
            W, H = img.size

        gt_count = float(gt.sum())
        gt_count_t = torch.tensor([gt_count], dtype=torch.float32)

        img_t = self.to_tensor_norm(img) if self.to_tensor_norm else transforms.ToTensor()(img)
        return img_t, gt_count_t, (H, W)


# ==============================================================================
# Split and dataloaders
# ==============================================================================
def split_train_data(data_root, dataset_name, labeled_ratio=0.1, seed=42):
    paths = DATASET_PATHS[dataset_name]
    img_dir = os.path.join(data_root, paths["train_img"])

    all_files = sorted([
        os.path.join(img_dir, x) for x in os.listdir(img_dir)
        if x.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    rng = random.Random(int(seed))
    rng.shuffle(all_files)

    total = len(all_files)
    n_labeled = max(1, int(total * float(labeled_ratio)))
    n_labeled = min(n_labeled, total)

    labeled_files = all_files[:n_labeled]
    unlabeled_files = all_files[n_labeled:]
    if len(unlabeled_files) == 0:
        unlabeled_files = labeled_files.copy()

    return labeled_files, unlabeled_files


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(args):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    to_tensor_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # weak (light) - ensures teacher view is relatively clean
    weak_pil = transforms.RandomApply([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
    ], p=0.3)

    # strong (FixMatch style, appearance-only)
    strong_pil = transforms.Compose([
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    # tensor-level occlusion
    strong_tensor = transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)

    labeled_files, unlabeled_files = split_train_data(
        args.data_root, args.dataset_name, args.labeled_ratio, args.seed
    )
    print(f"Data Split: {len(labeled_files)} Labeled, {len(unlabeled_files)} Unlabeled")

    train_l = GeneralCrowdDataset(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        split="train",
        labeled=True,
        crop_size=args.crop_size,
        downsample_ratio=args.downsample_ratio,
        weak_pil=None,
        strong_pil=None,
        to_tensor_norm=to_tensor_norm,
        strong_tensor=None,
        file_list=labeled_files,
        seed=args.seed,
    )

    train_u = GeneralCrowdDataset(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        split="train",
        labeled=False,
        crop_size=args.crop_size,
        downsample_ratio=args.downsample_ratio,
        weak_pil=weak_pil,
        strong_pil=strong_pil,
        to_tensor_norm=to_tensor_norm,
        strong_tensor=strong_tensor,
        file_list=unlabeled_files,
        seed=args.seed,
    )

    val_set = GeneralCrowdDataset(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        split="val",
        labeled=True,
        crop_size=args.crop_size,
        downsample_ratio=args.downsample_ratio,
        weak_pil=None,
        strong_pil=None,
        to_tensor_norm=to_tensor_norm,
        strong_tensor=None,
        file_list=None,
        seed=args.seed,
    )

    persistent = bool(args.num_workers and args.num_workers > 0)

    loader_l = DataLoader(
        train_l,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        persistent_workers=persistent,
    )

    loader_u = DataLoader(
        train_u,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        persistent_workers=persistent,
    )

    loader_val = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        persistent_workers=persistent,
    )

    return loader_l, loader_u, loader_val
