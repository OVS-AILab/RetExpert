# --------------------------------------------------------------------------
# RetExpert: A Test-time Clinically Adaptive Framework for Retinal Disease Detection
#
# Official Implementation of the Paper:
# "RetExpert: A test-time clinically adaptive framework for detecting multiple 
#  fundus diseases by harnessing ophthalmic foundation models"
#
# Authors: Hongyang Jiang, Zirong Liu, et al.
# Copyright (c) 2025 The Chinese University of Hong Kong & Wenzhou Medical University.
#
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from config import get_dataset_config

# --------------------------------------------------------------------------
# Image Processing Utilities
# --------------------------------------------------------------------------

def crop_fundus_image(image, threshold=10):
    """
    Removes the black borders (background) from fundus images.
    Optimized using NumPy masking instead of Python loops for performance.
    
    Args:
        image (PIL.Image or np.ndarray): Input image.
        threshold (int): Pixel intensity threshold to distinguish ROI from background.
        
    Returns:
        np.ndarray: Cropped image.
    """
    # Convert PIL to NumPy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    # Convert to grayscale to find the mask
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Create binary mask where pixels are above threshold
    mask = gray > threshold

    # If the image is completely black, return original
    if not np.any(mask):
        return image

    # Find the bounding box of the non-black region
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the image
    cropped_image = image[y_min : y_max + 1, x_min : x_max + 1]
    
    return cropped_image

# --------------------------------------------------------------------------
# Transform Builder
# --------------------------------------------------------------------------

def build_transform(is_train, args):
    """
    Builds data augmentation pipeline based on training mode.
    """
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    # Train transform: Heavy augmentation (AutoAugment, Mixup friendly)
    if is_train == 'train':
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # Eval/Test transform: Resize and Normalize only
    t = []
    # Calculate crop percentage to maintain aspect ratio before center crop
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
        
    size = int(args.input_size / crop_pct)
    
    t.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC))
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    
    return transforms.Compose(t)

# --------------------------------------------------------------------------
# Dataset Classes
# --------------------------------------------------------------------------

class BaseRetinaDataset(Dataset):
    """
    A unified dataset class for retinal disease classification.
    Handles loading labels from CSV/Excel and images from directories.
    """
    def __init__(self, root, mode, dataset_name, transform=None, target_transform=None):
        """
        Args:
            root (str): Root directory of the dataset.
            mode (str): 'train', 'val', or 'test'.
            dataset_name (str): Name of the dataset (must map to config.py).
            transform (callable): Image transformations.
        """
        self.root = root
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        
        # Load dataset specific configuration
        self.config = get_dataset_config(dataset_name)
        self.class_names = self.config['class_names']
        self.num_classes = self.config['num_classes']

        # Define annotation file path
        # Supports CSV or Excel files based on extension detection or config
        if dataset_name == 'ODIR':
            file_path = os.path.join(root, f'{mode}.xlsx')
            self.df = pd.read_excel(file_path)
            # ODIR specific: Image folder name
            self.img_dir = os.path.join(root, 'ODIR-5K_Training_Dataset_Processed')
            # ODIR specific: Column name for image filenames
            self.id_col = 'Fundus'
        else:
            # Default behavior for MuReD, ADAM, etc.
            file_path = os.path.join(root, f'{mode}.csv')
            self.df = pd.read_csv(file_path)
            self.img_dir = os.path.join(root, 'images')
            # Default ID column (usually the first column)
            self.id_col = self.df.columns[0] 
            if dataset_name == 'ADAM':
                 self.id_col = 'id_code'

        # Extract labels
        # Ensure columns exist
        missing_cols = [c for c in self.class_names if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns {missing_cols} in {file_path}")
            
        self.labels = self.df[self.class_names].values
        self.image_ids = self.df[self.id_col].tolist()

    def __len__(self):
        return len(self.labels)

    def _load_image(self, img_name):
        """
        Robust image loading handling multiple extensions.
        """
        # Possible extensions to try
        extensions = ['', '.png', '.jpg', '.jpeg', '.tif', '.bmp', '.JPG']
        
        for ext in extensions:
            # Construct full path
            # Handle cases where img_name already has extension
            if str(img_name).lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                full_path = os.path.join(self.img_dir, str(img_name))
            else:
                full_path = os.path.join(self.img_dir, str(img_name) + ext)
                
            if os.path.exists(full_path):
                try:
                    img = Image.open(full_path).convert("RGB")
                    return img, os.path.basename(full_path)
                except Exception as e:
                    print(f"Error loading {full_path}: {e}")
                    continue
                    
        raise FileNotFoundError(f"Image {img_name} not found in {self.img_dir} with common extensions.")

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        label = self.labels[idx]
        
        # 1. Load Image
        image, filename = self._load_image(img_id)

        # 2. Pre-process (Crop black borders)
        # Note: It is recommended to do this offline for speed, but kept here for compatibility.
        image_np = crop_fundus_image(image)
        image = Image.fromarray(image_np)

        # 3. Apply Transforms
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)
            
        # Ensure label is float for Multi-label BCE/RAL loss
        label = torch.tensor(label, dtype=torch.float32)

        return image, label, filename

# --------------------------------------------------------------------------
# Complex Dataset Classes (For irregular folder structures)
# --------------------------------------------------------------------------
# If you have datasets with subfolder structures (like 0/img.jpg, 1/img.jpg),
# you can define specific classes here inheriting from Dataset.
# For now, the BaseRetinaDataset covers MuReD, ODIR, and ADAM.

# --------------------------------------------------------------------------
# Main Builder Function
# --------------------------------------------------------------------------

def build_dataset(is_train, args):
    """
    Main entry point to build datasets.
    Uses args.dataset to determine which config to load.
    """
    transform = build_transform(is_train, args)
    
    # For standard datasets defined in config.py
    if args.dataset in ['MuReD', 'ODIR', 'ADAM']:
        dataset = BaseRetinaDataset(
            root=args.data_path,
            mode=is_train,
            dataset_name=args.dataset,
            transform=transform
        )
        return dataset
    
    # You can add elif blocks here for legacy datasets if needed
    # elif args.dataset == 'DiabeticRetinopathyArranged':
    #     return SingleLabelDataset_Legacy(...)
        
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not supported in build_dataset.")

# Legacy builders (kept for reference, can be removed if switched to BaseRetinaDataset)
def build_ODIR_dataset(is_train, args):
    # Wrapper for compatibility if old scripts call this specific function
    args.dataset = 'ODIR'
    return build_dataset(is_train, args)

def build_MESSDIOR2_dataset(is_train, args):
    # Wrapper for compatibility
    args.dataset = 'ADAM'
    return build_dataset(is_train, args)