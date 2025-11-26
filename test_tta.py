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

# test_tta.py
import argparse
import copy
import time
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import datetime
from pathlib import Path
from config import set_trainable_parameters
from models.vit_adapter import build_retexpert_vit_large
from losses.retexpert_loss import RetExpertLoss
from utils.constants import MURED_FDCM, ODIR_FDCM
import utils.misc as misc
from utils.engine import compute_metrics
from utils.datasets import build_dataset, build_ODIR_dataset, build_MESSDIOR2_dataset

def get_args_parser():
    parser = argparse.ArgumentParser('RetExpert Test-Time Adaptation (TTA)', add_help=False)
    
    # basic parameters
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for TTA (usually 1 or small)')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--model', default='vit_large_patch16', type=str)
    parser.add_argument('--finetune', default='', help='Path to well-trained checkpoint')
    parser.add_argument('--adapter_dim', default=64, type=int)
    parser.add_argument('--adapter_mode', default='AKU', type=str)
    
    # TTA specific parameters
    parser.add_argument('--tta_lr', type=float, default=1e-3, help='Learning rate for TTA')
    parser.add_argument('--ttul_epochs', type=int, default=1, help='Epochs for Unsupervised stage')
    parser.add_argument('--ttpl_epochs', type=int, default=1, help='Epochs for Pseudo-supervised stage')
    parser.add_argument('--adapter_dim', default=64, type=int, help='Dimension of AKU Adapter')
    parser.add_argument('--adaptation_mode', default='ttul', type=str, 
                        help='Fine-tuning strategy: ttul, adapter, full, head, or custom')
    
    # data and environment
    parser.add_argument('--data_path', default='./data/MuReD', type=str)
    parser.add_argument('--dataset', default='MuReD', type=str, choices=['MuReD', 'ODIR'])
    parser.add_argument('--nb_classes', default=20, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--output_dir', default='./output_dir_tta')
    parser.add_argument('--seed', default=42, type=int)
    
    return parser

def softmax_entropy(x):
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class WeightedBCELoss(nn.Module):
    """
    Binary Cross Entropy Loss weighted by Uncertainty scores.
    Used in TTPL stage.
    """
    def forward(self, inputs, targets, weights):
        # inputs: logits, targets: binary labels
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # weights: (1 - uncertainty)
        weighted_loss = (bce_loss * weights).mean(dim=1)
        return weighted_loss.mean()

def run_tta_for_batch(base_model, images, args, criterion_aux):
    """
    Performs TTA on a single batch of images.
    Steps:
        1. Clone model
        2. TTUL (Optimize Adapters/Norms via Entropy)
        3. TTPL (Optimize Head via Uncertainty-weighted Pseudo-labels)
        4. Predict
    """
    # 1. Clone the model (Instance-specific adaptation)
    model = copy.deepcopy(base_model)
    model.eval() # BN layers (if any) should generally stay in eval mode
    
    # Setup Optimizers for different stages
    # Filter parameters based on strategies defined in Paper Section 2.6
    
    # --- Stage 1: TTUL (Test-Time Unsupervised Learning) ---
    # Target: Adapters and Norm layers
    
    ttul_params = set_trainable_parameters(model, args.adaptation_mode)
            
            
    optimizer_ttul = torch.optim.SGD(ttul_params, lr=args.tta_lr, momentum=0.9)
    
    # Optimization Loop for TTUL
    for _ in range(args.ttul_epochs):
        # Forward
        outputs, _ = model(images)
        
        # Loss: Soft Entropy Minimization (minimize prediction uncertainty)
        loss = softmax_entropy(outputs).mean()
        
        optimizer_ttul.zero_grad()
        loss.backward()
        optimizer_ttul.step()
        
    # --- Stage 2: TTPL (Test-Time Pseudo-Supervised Learning) ---
    # Target: Task-specific Head
    ttpl_params = []
    for name, p in model.named_parameters():
        if 'head' in name:
            p.requires_grad = True
            ttpl_params.append(p)
        else:
            p.requires_grad = False
            
    optimizer_ttpl = torch.optim.SGD(ttpl_params, lr=args.tta_lr, momentum=0.9)
    criterion_pl = WeightedBCELoss()
    
    # Generate Pseudo-labels and Uncertainty weights
    with torch.no_grad():
        outputs, _ = model(images)
        probs = torch.sigmoid(outputs)
        
        # Pseudo-labels (Binarized predictions)
        # Using 0.5 threshold as standard
        pseudo_labels = (probs > 0.5).float()
        
        # Calculate Uncertainty (using our RetExpertLoss helper)
        # We dummy epoch=0 here as we just need the u_k scores
        _, u_k = criterion_aux.uaml_loss(outputs, pseudo_labels, 0, 1)
        
        # Weights: Higher uncertainty -> Lower weight
        # w_k = 1 - u_k
        pl_weights = 1.0 - u_k
        # Optional: Clip or normalize weights if needed

    # Optimization Loop for TTPL
    for _ in range(args.ttpl_epochs):
        outputs, _ = model(images)
        loss = criterion_pl(outputs, pseudo_labels, pl_weights)
        
        optimizer_ttpl.zero_grad()
        loss.backward()
        optimizer_ttpl.step()
        
    # --- Final Prediction ---
    with torch.no_grad():
        model.eval()
        final_outputs, _ = model(images)
        
    return final_outputs

def main(args):
    device = torch.device(args.device)
    cudnn.benchmark = True
    
    # 1. Load Dataset (Test set only)
    if args.dataset == 'MuReD':
        dataset_test = build_dataset(is_train='test', args=args)
        fdcm_matrix = MURED_FDCM
    elif args.dataset == 'ODIR':
        dataset_test = build_ODIR_dataset(is_train='test', args=args)
        fdcm_matrix = ODIR_FDCM
    elif args.dataset == 'Messidor':
        dataset_test = build_MESSDIOR2_dataset(is_train='test', args=args)
        fdcm_matrix = None
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers, drop_last=False
    )
    
    # 2. Load Base Model
    print(f"Loading base model from {args.finetune}...")
    base_model = build_retexpert_vit_large(
        num_classes=args.nb_classes,
        adapter_mode=args.adapter_mode,
        adapter_dim=args.adapter_dim
    )
    
    checkpoint = torch.load(args.finetune, map_location='cpu')
    base_model.load_state_dict(checkpoint['model'], strict=False)
    base_model.to(device)
    base_model.eval() # Base model stays in eval mode
    
    # Helper for uncertainty calculation
    criterion_aux = RetExpertLoss(args.nb_classes, device, fdcm_matrix=fdcm_matrix)
    
    # 3. Start TTA Loop
    print(f"Starting TTA (TTUL={args.ttul_epochs} eps, TTPL={args.ttpl_epochs} eps)...")
    start_time = time.time()
    
    all_probs = []
    all_targets = []
    
    for i, (images, targets, _) in enumerate(data_loader_test):
        images = images.to(device, non_blocking=True)
        
        # Run TTA for this specific batch
        # Note: We pass the base_model, which is cloned inside the function
        outputs = run_tta_for_batch(base_model, images, args, criterion_aux)
        
        probs = torch.sigmoid(outputs)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.numpy())
        
        if i % 10 == 0:
            print(f"Processed batch {i}/{len(data_loader_test)}")

    # 4. Evaluate
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    metrics = compute_metrics(all_probs, all_targets)
    print("\n=== TTA Evaluation Results ===")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"Kappa:    {metrics['kappa']:.4f}")
    print(f"AUC:      {metrics['auc']:.4f}")
    
    # Save results
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.output_dir, "tta_results.txt"), "w") as f:
            f.write(json.dumps(metrics, indent=4))

    total_time = time.time() - start_time
    print(f'Total time: {str(datetime.timedelta(seconds=int(total_time)))}')

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)