# dataset/loaddata.py  —— 最终稳定版（兼容所有训练脚本）
import os
import random
import torch
import numpy as np
import cv2
import h5py
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms


class DatasetNames:
    SHA = 'SHA'
    SHB = 'SHB'
    QNRF = 'qnrf'
    NWPU = 'nwpu'


DATASET_PATHS = {
    DatasetNames.SHA: {
        'train_img': '/data/LM/Dataset/shanghaitech_part_A/train/img',
        'train_gt': '/data/LM/Dataset/shanghaitech_part_A/train/new_data',
        'val_img': '/data/LM/Dataset/shanghaitech_part_A/test/img',
        'val_gt': '/data/LM/Dataset/shanghaitech_part_A/test/new_data'
    },
    DatasetNames.SHB: {
        'train_img': '/data/LM/Dataset/shanghaitech_part_B/train/img',
        'train_gt': '/data/LM/Dataset/shanghaitech_part_B/train/new_data',
        'val_img': '/data/LM/Dataset/shanghaitech_part_B/test/img',
        'val_gt': '/data/LM/Dataset/shanghaitech_part_B/test/new_data'
    },
    DatasetNames.QNRF: {
        'train_img': '/data/LM/Dataset/UCF-QNRF/train/img',
        'train_gt': '/data/LM/Dataset/UCF-QNRF/train/new_data',
        'val_img': '/data/LM/Dataset/UCF-QNRF/test/img',
        'val_gt': '/data/LM/Dataset/UCF-QNRF/test/new_data'
    },
    DatasetNames.NWPU: {
        'train_img': '/data/LM/Dataset/NWPU/train/img',
        'train_gt': '/data/LM/Dataset/NWPU/train/new_data',
        'val_img': '/data/LM/Dataset/NWPU/test/img',
        'val_gt': '/data/LM/Dataset/NWPU/test/new_data'
    },
}


def load_image(path):
    return Image.open(path).convert('RGB')


def load_density(path):
    with h5py.File(path, 'r') as f:
        key = 'density' if 'density' in f else 'image'
        return np.asarray(f[key], dtype=np.float32)


class PairedCrop:
    def __init__(self, size=512, scale=4):
        self.size = size  # 输入图像尺寸
        self.scale = scale  # 模型输出是 input 的 1/scale
        self.out_size = size // scale  # 128 if size=512, scale=4

    def __call__(self, img, den):
        w, h = img.size

        # pad to at least crop size
        if w < self.size or h < self.size:
            img = TF.pad(img, (0, 0, max(0, self.size - w), max(0, self.size - h)), fill=0)
            den = np.pad(den, ((0, max(0, self.size - h)), (0, max(0, self.size - w))), 'constant')
            w, h = img.size

        # random crop
        i = random.randint(0, h - self.size)
        j = random.randint(0, w - self.size)
        img = TF.crop(img, i, j, self.size, self.size)
        den = den[i:i + self.size, j:j + self.size]

        # 下采样 density map 到 output size (128×128)
        if self.scale > 1:
            den = cv2.resize(den, (self.out_size, self.out_size), interpolation=cv2.INTER_CUBIC)
            den *= (self.scale ** 2)  # 保持总人数不变

        return img, den


class CrowdDataset(Dataset):
    def __init__(self, root, dataset_name, split='train', labeled=True,
                 crop_size=512, transform=None, file_list=None):
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.labeled = labeled
        self.transform = transform
        self.cropper = PairedCrop(size=crop_size, downscale=4)

        paths = DATASET_PATHS[dataset_name]
        img_dir = os.path.join(root, paths[f'{split}_img'])
        gt_dir = os.path.join(root, paths[f'{split}_gt'])

        if file_list is not None:
            self.files = file_list
        else:
            self.files = sorted([
                f for f in os.listdir(img_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

        self.img_dir = img_dir
        self.gt_dir = gt_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img_path = os.path.join(self.img_dir, name)

        img = load_image(img_path)
        orig_h, orig_w = img.height, img.width

        if self.split == 'train':
            if self.labeled:
                gt_path = os.path.join(self.gt_dir, os.path.splitext(name)[0] + '.h5')
                den = load_density(gt_path)
                img, den = self.cropper(img, den)

                if random.random() > 0.5:
                    img = TF.hflip(img)
                    den = np.fliplr(den).copy()

                if self.transform:
                    img = self.transform(img)

                den = torch.from_numpy(den).unsqueeze(0).float()
                return img, den, name

            else:
                # Unlabeled: return weak + strong aug
                img_weak = self.cropper(img, np.zeros((orig_h, orig_w)))[0]  # dummy den
                img_strong = img_weak.copy()

                if random.random() > 0.5:
                    img_weak = TF.hflip(img_weak)
                    img_strong = TF.hflip(img_strong)

                if self.transform:
                    img_weak = self.transform(img_weak)
                    img_strong = self.transform(img_strong)

                # Strong augmentation
                strong_aug = transforms.Compose([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                    transforms.RandomGrayscale(p=0.2),
                ])
                img_strong = strong_aug(img_strong)

                return img_weak, img_strong, name

        else:  # val
            gt_path = os.path.join(self.gt_dir, os.path.splitext(name)[0] + '.h5')
            den = load_density(gt_path)
            count = np.sum(den)

            # Pad to multiple of 32
            pad_w = (32 - orig_w % 32) % 32
            pad_h = (32 - orig_h % 32) % 32
            if pad_w or pad_h:
                img = TF.pad(img, (0, 0, pad_w, pad_h), fill=0)

            if self.transform:
                img = self.transform(img)

            return img, torch.tensor(count, dtype=torch.float32), (orig_h, orig_w), name


def build_dataloaders(args):
    """
    完全向后兼容：无论 args 有没有 crop_size，都不会报错
    """
    torch.manual_seed(getattr(args, 'seed', 2025))
    np.random.seed(getattr(args, 'seed', 2025))
    random.seed(getattr(args, 'seed', 2025))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 安全获取 crop_size，默认为 512（DARP 推荐）
    crop_size = getattr(args, 'crop_size', 512)

    # 数据切分
    paths = DATASET_PATHS[args.dataset_name]
    all_img_paths = sorted([
        os.path.join(args.data_root, paths['train_img'], f)
        for f in os.listdir(os.path.join(args.data_root, paths['train_img']))
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    random.shuffle(all_img_paths)
    n_total = len(all_img_paths)
    n_labeled = max(1, int(n_total * getattr(args, 'labeled_ratio', 0.1)))
    labeled_files = [os.path.basename(p) for p in all_img_paths[:n_labeled]]
    unlabeled_files = [os.path.basename(p) for p in all_img_paths[n_labeled:]]

    print(f"Data Split: {len(labeled_files)} Labeled, {len(unlabeled_files)} Unlabeled")

    train_labeled = CrowdDataset(
        root=args.data_root,
        dataset_name=args.dataset_name,
        split='train',
        labeled=True,
        crop_size=crop_size,
        transform=transform,
        file_list=labeled_files
    )
    train_unlabeled = CrowdDataset(
        root=args.data_root,
        dataset_name=args.dataset_name,
        split='train',
        labeled=False,
        crop_size=crop_size,
        transform=transform,
        file_list=unlabeled_files
    )
    val_set = CrowdDataset(
        root=args.data_root,
        dataset_name=args.dataset_name,
        split='val',
        labeled=False,
        crop_size=crop_size,
        transform=transform
    )

    loader_l = DataLoader(train_labeled, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True, pin_memory=True)
    loader_u = DataLoader(train_unlabeled, batch_size=args.batch_size * 2, shuffle=True,
                          num_workers=args.num_workers, drop_last=True, pin_memory=True)
    loader_val = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    return loader_l, loader_u, loader_val