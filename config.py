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

# config.py
import os

# ==============================================================================
# 1. Global Constants
# ==============================================================================

# default normalization values for ImageNet pre-trained models
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

# default input image size
DEFAULT_INPUT_SIZE = 224

# ==============================================================================
# 2. Dataset Metadata
# ==============================================================================

DATASET_CONFIGS = {
    'MuReD': {
        'num_classes': 20,
        'class_names': [
            'DR', 'NORMAL', 'MH', 'ODC', 'TSLN', 
            'ARMD', 'DN', 'MYA', 'BRVO', 'ODP',
            'CRVO', 'CNV', 'RS', 'ODE', 'LS',
            'CSR', 'HTR', 'ASR', 'CRS', 'OTHER'
        ],
        'default_path': './data/MuReD',
        'fdcm_key': 'MURED_FDCM' 
    },
    
    'ODIR': {
        'num_classes': 8,
        'class_names': [
            'Normal', 'Diabetes', 'Glaucoma', 'Cataract',
            'AMD', 'Hypertension', 'Myopia', 'Other'
        ],
        'default_path': './data/ODIR-5K',
        'fdcm_key': 'ODIR_FDCM'
    },
    
    'Messidor': {
        'num_classes': 2, # Messidor Usually for binary DR classification
        'class_names': ['Normal', 'DR_Referable'],
        'default_path': './data/Messidor2',
        'fdcm_key': None # tagged as None since no FDCM is used
    }
}

# ==============================================================================
# 3. Default Hyperparameters
# ==============================================================================

# RetExpert default hyperparameters
RET_EXPERT_DEFAULTS = {
    'adapter_dim': 64,
    'adapter_mode': 'ARC',
    'beta_uncertain': 0.2,
    'gamma_reg': 0.05,
    'alpha_rb_loss': 0.5,
}

def get_dataset_config(dataset_name):
    """
    Helper function to retrieve dataset config safely.
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{dataset_name}' not found in config.py. "
                         f"Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]


def set_trainable_parameters(model, mode):
    """
    Control which parameters are trainable based on the fine-tuning mode.
    
    Args:
        model: The model instance (without DDP wrapper preferred for name checking)
        mode: 'full', 'adapter', 'head', etc.
    """
    print(f"Setting trainable parameters for mode: {mode}")
    ttul_params = []
    # 1. Full fine-tuning
    if mode == 'full':
        for p in model.parameters():
            p.requires_grad = True
            ttul_params.append(p)
        return

    # 2. Force freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False

    # 3. Set specific parameters to be trainable
    for name, p in model.named_parameters():
        
        # --- Mode: RetExpert (Paper Default) ---
        # Ref: Section 2.3 [cite: 95, 96]
        # Train: Adapters, Projections, Heads, Normalization layers
        if mode == 'ttul':
            if 'adapter' in name or 'projection' in name or 'norm' in name:
                p.requires_grad = True
                ttul_params.append(p)
            else:
                p.requires_grad = False
            
        elif mode == 'adapter':
            if 'adapter' in name:        # AKU Adapter weights
                p.requires_grad = True 
                ttul_params.append(p)
            elif 'projection' in name:   # Adapter projection matrices
                p.requires_grad = True
                ttul_params.append(p)
            elif 'head' in name:         # Classification head
                p.requires_grad = True
                ttul_params.append(p)
            elif 'norm' in name:         # Normalization layers (LayerNorm)
                p.requires_grad = True
                ttul_params.append(p)

        # --- Mode: Linear Probing (Head Only) ---
        elif mode == 'head':
            if 'head' in name:
                p.requires_grad = True
                ttul_params.append(p)

        # --- Mode: Custom (Example: Encoder Only) ---
        elif mode == 'last_blocks':
            if 'head' in name or 'norm' in name:
                p.requires_grad = True
                ttul_params.append(p)
            # 假设 blocks 命名为 blocks.23, blocks.22...
            elif 'blocks.23' in name or 'blocks.22' in name:
                p.requires_grad = True
                ttul_params.append(p)

        # --- Error Handling ---
        else:
            if 'head' in name:
                p.requires_grad = True
                ttul_params.append(p)
                
    return ttul_params
