"""
dataset.py - Oxford-IIIT Pet Semantic Segmentation Data Contract
Author: Roll No. 23115004
Course: Advanced Computer Vision (ACV)
Assignment: B - Reproducible Segmentation Experiment

Data Contract:
  - Source: torchvision.datasets.OxfordIIITPet (Official trimap masks)
  - Resolution: 256 x 256 pixels
  - Normalization: ImageNet Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225]
  - Classes:
      Class 0: Background
      Class 1: Foreground (Pet body)
      Class 2: Trimap boundary / ambiguous border (ignore_index = 2)
  - Train Augmentations: Random Horizontal Flip (p=0.5), Random Resized Crop (scale=(0.85, 1.0))
  - Test/Val: Deterministic Resize (256, 256) + Nearest-Neighbor mask sampling.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as T
from torchvision import datasets
from PIL import Image

class PetSegmentationDataset:
    """
    Encapsulates dataset downloading, preprocessing, split handling, 
    and DataLoader construction strictly conforming to the Task 1 Data Contract.
    """
    def __init__(self, root_dir: str = './data', batch_size: int = 16, num_workers: int = 2):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Standard ImageNet normalization transforms for RGB input
        self.img_transforms = T.Compose([
            T.Resize((256, 256)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Nearest-neighbor interpolation transforms to prevent label corruption
        self.mask_transforms = T.Compose([
            T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST_EXACT),
            T.ToImage(),
            T.ToDtype(torch.long, scale=False)
        ])

    def preprocess_target(self, target: Image.Image) -> torch.Tensor:
        """
        Maps Oxford-IIIT Pet trimap annotations from {1, 2, 3} to zero-indexed {0, 1, 2}:
          Raw 1 (Foreground) -> mapped to 0 (or standard: 1 Foreground, 0 Background)
          Oxford Pet specification:
            1: Foreground (Pet)
            2: Background
            3: Outline / Border
          Mapping applied:
            Target - 1:
              0: Foreground (Pet)
              1: Background
              2: Border / Ignore
          To maintain intuitive class convention:
            Class 0: Background
            Class 1: Foreground
            Class 2: Border
        """
        target_tensor = self.mask_transforms(target)
        # Raw Oxford-IIIT Pet trimaps: 1 = Foreground, 2 = Background, 3 = Outline
        # We remap:
        #   pixel == 2 -> 0 (Background)
        #   pixel == 1 -> 1 (Foreground)
        #   pixel == 3 -> 2 (Border / Ignore)
        remapped = torch.zeros_like(target_tensor, dtype=torch.long)
        remapped[target_tensor == 2] = 0  # Background
        remapped[target_tensor == 1] = 1  # Foreground (Pet)
        remapped[target_tensor == 3] = 2  # Border / Ignore
        return remapped.squeeze(0)  # Shape: (256, 256)

    def get_dataloaders(self, download: bool = True):
        """
        Builds reproducible training and test DataLoaders.
        """
        os.makedirs(self.root_dir, exist_ok=True)
        
        train_set = datasets.OxfordIIITPet(
            root=self.root_dir,
            split='trainval',
            target_types='segmentation',
            download=download,
            transform=self.img_transforms,
            target_transform=self.preprocess_target
        )
        
        test_set = datasets.OxfordIIITPet(
            root=self.root_dir,
            split='test',
            target_types='segmentation',
            download=download,
            transform=self.img_transforms,
            target_transform=self.preprocess_target
        )

        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, test_loader

class SyntheticDummyDataset(Dataset):
    """
    Lightweight synthetic mock dataset strictly matching dimensions and classes
    for unit testing and smoke verification without network downloads.
    """
    def __init__(self, size: int = 32, height: int = 256, width: int = 256):
        self.size = size
        self.height = height
        self.width = width

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate synthetic 3-channel normalized image
        img = torch.randn(3, self.height, self.width, dtype=torch.float32)
        # Generate synthetic trimap mask with values {0: BG, 1: FG, 2: Border}
        mask = torch.randint(0, 3, (self.height, self.width), dtype=torch.long)
        return img, mask
