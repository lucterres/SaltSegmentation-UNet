import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# Transforms applied identically to image and mask (geometric only)
def get_transforms(height: int, width: int, augment: bool = False):
    """Return (img_transform, mask_transform) pair.

    Images: resize → tensor → normalise to [0,1].
    Masks:  resize with NEAREST (no interpolation artefacts) → tensor → binarise.
    Augmentation (geometric only, identical for image+mask): random horizontal flip.
    """
    img_ops = [
        transforms.ToPILImage(),
        transforms.Resize((height, width), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),          # → [0,1] float
    ]
    mask_ops = [
        transforms.ToPILImage(),
        transforms.Resize((height, width), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),          # → [0,1] float
    ]
    img_tf  = transforms.Compose(img_ops)
    mask_tf = transforms.Compose(mask_ops)
    return img_tf, mask_tf


class SegmentationDataset(Dataset):
    """TGS Salt dataset.  Works for both real and synthetic images/masks."""

    def __init__(self, img_paths, mask_paths, img_tf, mask_tf,
                 augment: bool = False) -> None:
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.img_tf     = img_tf
        self.mask_tf    = mask_tf
        self.augment    = augment

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image as grayscale (1 channel — TGS is already grayscale)
        img  = cv.imread(self.img_paths[idx],  cv.IMREAD_GRAYSCALE)
        mask = cv.imread(self.mask_paths[idx], cv.IMREAD_GRAYSCALE)

        # Apply transforms
        img  = self.img_tf(img)          # (1, H, W), float32 in [0,1]
        mask = self.mask_tf(mask)        # (1, H, W), float32 in {0,1}

        # Binarise mask (NEAREST resize keeps it close to binary; threshold to be safe)
        mask = (mask > 0.5).float()

        # Random horizontal flip (applied consistently via same random state)
        if self.augment and torch.rand(1).item() > 0.5:
            img  = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)

        return img, mask
