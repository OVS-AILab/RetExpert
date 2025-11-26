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


import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
