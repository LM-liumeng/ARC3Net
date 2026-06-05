import os
import random
import numpy as np
import cv2
import h5py
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms


# ==============================================================================
# 1) Dataset registry
# ==============================================================================
class DatasetNames:
    SHA = 'SHA'
    SHB = 'SHB'
    QNRF = 'qnrf'
    NWPU = 'nwpu'


DATASET_PATHS = {
    DatasetNames.SHA: {
        'train_img': 'shanghaitech_part_A/train/img',
        'train_gt':  'shanghaitech_part_A/train/new_data',
        'val_img':   'shanghaitech_part_A/test/img',
        'val_gt':    'shanghaitech_part_A/test/new_data',
    },
    DatasetNames.SHB: {
        'train_img': 'shanghaitech_part_B/train/img',
        'train_gt':  'shanghaitech_part_B/train/new_data',
        'val_img':   'shanghaitech_part_B/test/img',
        'val_gt':    'shanghaitech_part_B/test/new_data',
    },
    DatasetNames.QNRF: {
        'train_img': 'UCF-QNRF/train/img',
        'train_gt':  'UCF-QNRF/train/new_data',
        'val_img':   'UCF-QNRF/test/img',
        'val_gt':    'UCF-QNRF/test/new_data',
    },
    DatasetNames.NWPU: {
        'train_img': 'NWPU/train/img',
        'train_gt':  'NWPU/train/new_data',
        'val_img':   'NWPU/test/img',
        'val_gt':    'NWPU/test/new_data',
    },
}


# ==============================================================================
# 2) I/O
# ==============================================================================
def load_image(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')


def load_density(path: str) -> np.ndarray:
    with h5py.File(path, 'r') as f:
        if 'density' in f.keys():
            k = 'density'
        elif 'image' in f.keys():
            k = 'image'
        else:
            raise KeyError(f"h5 file has no 'density' or 'image' key: {path}")
        gt = np.asarray(f[k], dtype=np.float32)
    return gt


# ==============================================================================
# 3) Paired crop + density downsample (mass-preserving, non-negative)
# ==============================================================================
class PairedCrop:
    """
    Random crop on PIL image + numpy density map; then downsample density by block-sum pooling.
    Ensures:
      - output image: crop_size x crop_size
      - output density: (crop_size/downsample) x (crop_size/downsample)
      - total count preserved
      - density non-negative
    """
    def __init__(self, size=256, downsample=4):
        self.size = int(size)
        self.downsample = int(downsample)
        if self.downsample < 1:
            raise ValueError("downsample must be >= 1")

    def pad_if_needed_img_den(self, img: Image.Image, den: np.ndarray):
        W, H = img.size
        if W >= self.size and H >= self.size:
            return img, den

        pad_w = max(0, self.size - W)
        pad_h = max(0, self.size - H)

        img = TF.pad(img, (0, 0, pad_w, pad_h), fill=0)
        den = np.pad(den, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0.0)
        return img, den

    def pad_if_needed_img_only(self, img: Image.Image):
        W, H = img.size
        if W >= self.size and H >= self.size:
            return img
        pad_w = max(0, self.size - W)
        pad_h = max(0, self.size - H)
        return TF.pad(img, (0, 0, pad_w, pad_h), fill=0)

    def _downsample_blocksum(self, den_crop: np.ndarray):
        ds = self.downsample
        if ds == 1:
            return np.clip(den_crop, 0.0, None)

        h, w = den_crop.shape
        # If not divisible, fallback to AREA + scale to preserve mass approximately
        if (h % ds != 0) or (w % ds != 0):
            new_h, new_w = h // ds, w // ds
            den_small = cv2.resize(den_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            den_small = den_small * (ds * ds)
            return np.clip(den_small, 0.0, None)

        den_small = den_crop.reshape(h // ds, ds, w // ds, ds).sum(axis=(1, 3))
        return np.clip(den_small, 0.0, None)

    def crop_params(self, img: Image.Image):
        img = self.pad_if_needed_img_only(img)
        W, H = img.size
        i = random.randint(0, H - self.size)
        j = random.randint(0, W - self.size)
        return img, i, j

    def __call__(self, img: Image.Image, den: np.ndarray):
        img, den = self.pad_if_needed_img_den(img, den)
        W, H = img.size

        i = random.randint(0, H - self.size)
        j = random.randint(0, W - self.size)

        img_crop = TF.crop(img, i, j, self.size, self.size)
        den_crop = den[i:i + self.size, j:j + self.size].astype(np.float32, copy=False)
        den_ds = self._downsample_blocksum(den_crop)
        return img_crop, den_ds


# ==============================================================================
# 4) Dataset
# ==============================================================================
class GeneralCrowdDataset(Dataset):
    """
    Train labeled:    returns (img_tensor, gt_density_tensor[1,h,w])
    Train unlabeled:  returns (img_weak_tensor, img_strong_tensor)  # STRICTLY aligned geometry
    Val:              returns (img_tensor, gt_count_tensor, (H,W) before pad)
    """
    def __init__(self, data_root, dataset_name, split='train', labeled=True,
                 crop_size=256, downsample_ratio=4,
                 weak_transform=None, strong_aug=None,
                 file_list=None):

        self.split = split
        self.labeled = bool(labeled)

        self.cropper = PairedCrop(size=crop_size, downsample=downsample_ratio)

        # weak_transform: PIL -> Tensor(Normalize)
        self.weak_transform = weak_transform
        # strong_aug: PIL -> PIL (PHOTOMETRIC ONLY; must NOT change geometry)
        self.strong_aug = strong_aug

        if dataset_name not in DATASET_PATHS:
            raise KeyError(f"Unknown dataset_name: {dataset_name}")

        paths = DATASET_PATHS[dataset_name]
        if split == 'train':
            self.img_dir = os.path.join(data_root, paths['train_img'])
            self.gt_dir = os.path.join(data_root, paths['train_gt'])
        else:
            self.img_dir = os.path.join(data_root, paths['val_img'])
            self.gt_dir = os.path.join(data_root, paths['val_gt'])

        if file_list is not None:
            self.img_files = list(file_list)
        else:
            self.img_files = sorted([
                os.path.join(self.img_dir, x) for x in os.listdir(self.img_dir)
                if x.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

        if len(self.img_files) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx, retry=0):
        if retry > 5:
            raise RuntimeError(f"Too many retries loading item {idx}")

        img_path = self.img_files[idx]
        try:
            img = load_image(img_path)
        except Exception as e:
            print(f"[WARN] Error loading {img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1), retry + 1)

        # ---------------- TRAIN ----------------
        if self.split == 'train':
            # -------- labeled --------
            if self.labeled:
                basename = os.path.basename(img_path)
                gt_name = os.path.splitext(basename)[0] + '.h5'
                gt_path = os.path.join(self.gt_dir, gt_name)
                gt = load_density(gt_path)

                # paired crop (image + density), then mass-preserving downsample
                img, gt = self.cropper(img, gt)

                # shared random flip
                if random.random() > 0.5:
                    img = TF.hflip(img)
                    gt = np.fliplr(gt).copy()

                # to tensor + normalize
                if self.weak_transform is not None:
                    img_t = self.weak_transform(img)
                else:
                    img_t = transforms.ToTensor()(img)

                gt_t = torch.from_numpy(gt).unsqueeze(0).float()  # [1,h,w]
                return img_t, gt_t

            # -------- unlabeled (STRICT geometry alignment) --------
            else:
                # 1) shared crop params
                img, i, j = self.cropper.crop_params(img)
                img_crop = TF.crop(img, i, j, self.cropper.size, self.cropper.size)

                # 2) shared flip (IMPORTANT)
                do_flip = (random.random() > 0.5)
                if do_flip:
                    img_crop = TF.hflip(img_crop)

                # 3) weak view: no extra photometric aug (keep stable for teacher)
                img_w = img_crop

                # 4) strong view: photometric-only aug
                img_s = img_crop
                if self.strong_aug is not None:
                    img_s = self.strong_aug(img_s)

                # 5) same tensor+norm for both
                if self.weak_transform is not None:
                    img_w = self.weak_transform(img_w)
                    img_s = self.weak_transform(img_s)
                else:
                    img_w = transforms.ToTensor()(img_w)
                    img_s = transforms.ToTensor()(img_s)

                return img_w, img_s

        # ---------------- VAL ----------------
        else:
            basename = os.path.basename(img_path)
            gt_name = os.path.splitext(basename)[0] + '.h5'
            gt_path = os.path.join(self.gt_dir, gt_name)
            gt = load_density(gt_path)
            count = float(np.sum(gt))

            W, H = img.size
            new_W = (W + 31) // 32 * 32
            new_H = (H + 31) // 32 * 32
            pad_w = new_W - W
            pad_h = new_H - H
            if pad_w > 0 or pad_h > 0:
                img = TF.pad(img, (0, 0, pad_w, pad_h), fill=0)

            img_t = self.weak_transform(img) if self.weak_transform else transforms.ToTensor()(img)
            return img_t, torch.tensor(count, dtype=torch.float32), (H, W)


# ==============================================================================
# 5) Split + build_dataloaders
# ==============================================================================
def split_train_data(data_root, dataset_name, labeled_ratio=0.1, seed=2025):
    paths = DATASET_PATHS[dataset_name]
    img_dir = os.path.join(data_root, paths['train_img'])

    all_files = sorted([
        os.path.join(img_dir, x) for x in os.listdir(img_dir)
        if x.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    rng = random.Random(seed)
    rng.shuffle(all_files)

    total = len(all_files)
    n_labeled = max(1, int(total * float(labeled_ratio)))
    labeled_files = all_files[:n_labeled]
    unlabeled_files = all_files[n_labeled:]
    return labeled_files, unlabeled_files


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    weak_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*mean_std),
    ])

    # IMPORTANT: strong_aug must be photometric-only (no rotate/affine/crop)
    strong_aug = transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    )

    labeled_files, unlabeled_files = split_train_data(
        args.data_root, args.dataset_name, args.labeled_ratio, args.seed
    )
    print(f"Data Split: {len(labeled_files)} Labeled, {len(unlabeled_files)} Unlabeled")

    train_l = GeneralCrowdDataset(
        data_root=args.data_root, dataset_name=args.dataset_name, split='train', labeled=True,
        crop_size=args.crop_size, downsample_ratio=4,
        weak_transform=weak_transform, strong_aug=None,
        file_list=labeled_files
    )
    train_u = GeneralCrowdDataset(
        data_root=args.data_root, dataset_name=args.dataset_name, split='train', labeled=False,
        crop_size=args.crop_size, downsample_ratio=4,
        weak_transform=weak_transform, strong_aug=strong_aug,
        file_list=unlabeled_files
    )
    val_set = GeneralCrowdDataset(
        data_root=args.data_root, dataset_name=args.dataset_name, split='val', labeled=True,
        crop_size=args.crop_size, downsample_ratio=4,
        weak_transform=weak_transform, strong_aug=None,
        file_list=None
    )

    loader_l = DataLoader(
        train_l, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=True, worker_init_fn=_seed_worker
    )

    loader_u = DataLoader(
        train_u, batch_size=args.batch_size * 2, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=True, worker_init_fn=_seed_worker
    )

    loader_val = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True, worker_init_fn=_seed_worker
    )

    return loader_l, loader_u, loader_val
