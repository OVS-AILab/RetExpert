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

# utils/engine.py
import math
import sys
import random
import torch
import numpy as np
from typing import Iterable, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, cohen_kappa_score, average_precision_score, hamming_loss
)
import utils.misc as misc
import utils.lr_sched as lr_sched

def compute_metrics(probs, targets, threshold=0.5, average='samples'):
    """
    Compute comprehensive metrics for multi-label classification.
    Args:
        probs: Predicted probabilities (N, C)
        targets: Ground truth labels (N, C)
        threshold: Decision threshold
        average: Averaging strategy for sklearn metrics ('samples', 'macro', etc.)
    """
    preds = (probs >= threshold).astype(int)
    targets = targets.astype(int)

    acc = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average=average, zero_division=0)
    recall = recall_score(targets, preds, average=average, zero_division=0)
    f1 = f1_score(targets, preds, average=average, zero_division=0)
    
    # For AUC and AUPRC, handle cases where only one class is present in batch
    try:
        auc = roc_auc_score(targets, probs, average=average)
    except ValueError:
        auc = 0.0
        
    try:
        aupr = average_precision_score(targets, probs, average=average)
    except ValueError:
        aupr = 0.0

    # Kappa is typically for multi-class, for multi-label we often flatten or calculate per sample
    # Here we calculate based on flattened arrays to align with typical multi-label eval
    kappa = cohen_kappa_score(targets.flatten(), preds.flatten())
    hamming = hamming_loss(targets, preds)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "aupr": aupr,
        "kappa": kappa,
        "hamming": hamming
    }

def train_one_epoch(model: torch.nn.Module, 
                    criterion_cls: torch.nn.Module, # Main classification loss (e.g., RAL)
                    criterion_aux: torch.nn.Module, # RetExpert auxiliary loss
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int, 
                    loss_scaler, 
                    max_norm: float = 0,
                    mixup_fn: Optional[object] = None, 
                    log_writer=None,
                    args=None):
    
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, (samples, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Per-iteration lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            # Forward pass
            # outputlist contains intermediate features for SOA strategy
            outputs, outputlist = model(samples)
            
            # --- Core Strategy: SOA (Stochastic One-hot Activation) ---
            # Randomly select one intermediate block's output
            block_num = random.randint(0, len(outputlist) - 1)
            outputs_rb = outputlist[block_num]

            # 1. Main Classification Loss (e.g., RAL or BCE)
            loss_cls = criterion_cls(outputs, targets)

            # 2. Auxiliary Losses (UAML + FDCM + SOA)
            # Note: criterion_aux is our RetExpertLoss
            loss_uaml, loss_fdcm, loss_soa, _ = criterion_aux(
                outputs, targets, epoch, args.epochs, outputs_rb
            )

            # 3. Total Loss Aggregation
            # Formula: L_total = L_cls + beta * L_uaml + gamma * L_fdcm + alpha * L_soa
            loss = loss_cls + \
                   args.beta_UAML * loss_uaml + \
                   args.gamma_FDCM * loss_fdcm + \
                   args.alpha_SOA * loss_soa

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # Logging component losses for debugging
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value, epoch_1000x)
            log_writer.add_scalar('train/loss_cls', loss_cls.item(), epoch_1000x)
            log_writer.add_scalar('train/loss_uaml', loss_uaml.item(), epoch_1000x)
            log_writer.add_scalar('train/loss_fdcm', loss_fdcm.item(), epoch_1000x)
            log_writer.add_scalar('train/loss_soa', loss_soa.item(), epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # Lists to store all predictions and targets
    all_probs = []
    all_targets = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output, _ = model(images)
            
        probs = torch.sigmoid(output)
        
        all_probs.append(probs.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    # Concatenate all batches
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metrics
    metrics = compute_metrics(all_probs, all_targets, threshold=0.5)
    
    print(f"Evaluation Results:")
    print(f"F1-Score: {metrics['f1']:.4f} | Kappa: {metrics['kappa']:.4f} | AUC: {metrics['auc']:.4f}")
    
    return metrics