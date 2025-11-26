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

# losses/retexpert_loss.py
import torch
import torch.nn as nn
from torch.nn import functional as F

class RetExpertLoss(nn.Module):
    """
    Composite loss function for RetExpert framework.
    Integrates:
    1. UAML: Uncertainty-Aware Multi-Label learning
    2. FDCM: Fundus Disease Co-occurrence Matrix regularization
    3. KLD: Kullback-Leibler Divergence for SOA strategy (domain generalization)
    """
    def __init__(self, num_classes, device, fdcm_matrix=None, alpha_kl=2.0):
        """
        Args:
            num_classes (int): Number of disease categories.
            device (torch.device): Computing device.
            fdcm_matrix (torch.Tensor, optional): Pre-defined co-occurrence matrix (C x C).
                                                 If None, FDCM loss will be skipped.
            alpha_kl (float): Temperature scaling factor for KLD loss in SOA strategy.
        """
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.alpha_kl = alpha_kl
        
        # Load Fundus Disease Co-occurrence Matrix (FDCM)
        # Matrix logic corresponds to Equation (1) and Table 1 in the paper.
        if fdcm_matrix is not None:
            self.fdcm_matrix = 1.0 - fdcm_matrix.to(device) # Note: Code uses (1 - coefficient) logic
        else:
            self.fdcm_matrix = None

    def _kl_divergence(self, alpha):
        """Helper for UAML: Calculate KL divergence for Dirichlet distribution."""
        beta = torch.ones((1, self.num_classes)).to(self.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl

    def uaml_loss(self, outputs, targets, epoch, max_epochs):
        """
        Calculates Uncertainty-Aware Multi-Label (UAML) loss.
        Ref: Section 2.4, Uncertainty-aware learning.
        
        Args:
            outputs: Model logits (Batch x Classes)
            targets: Ground truth labels
            epoch: Current training epoch
            max_epochs: Total training epochs
            
        Returns:
            total_loss: Combined UAML loss
            u_k: Uncertainty scores for each class (for test-time adaptation)
        """
        # Evidential Deep Learning (EDL) logic
        evidences = F.softplus(outputs)
        alpha = evidences + 1
        
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        
        # Modified Cross-Entropy using Digamma
        # Computes expectation of log-likelihood under Dirichlet
        A = torch.sum(targets * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        
        # KL Divergence regularization with annealing
        annealing_coef = min(1, epoch / max_epochs)
        alp = E * (1 - targets) + 1
        B = annealing_coef * self._kl_divergence(alp)
        
        basic_loss = torch.mean(A + B)

        # Class-specific uncertainty regularization
        u_k = 1.0 / alpha  # Uncertainty score per class
        p_k = torch.sigmoid(outputs) 
        
        # Penalize uncertainty on confident predictions
        lambda_factor = 0.1 
        cun_ce_u = u_k * targets * torch.log(p_k + 1e-10)
        cun_ce = -torch.sum(cun_ce_u, dim=1, keepdim=True)

        total_loss = torch.mean(basic_loss + lambda_factor * cun_ce)
        
        return total_loss, u_k

    def fdcm_loss(self, outputs):
        """
        Calculates Fundus Disease Co-occurrence Matrix (FDCM) loss.
        Equation (1): L_FDCM = Mean( (p_i - p_j)^2 - (1 - m_ij)^2 )
        
        Args:
            outputs: Model logits (Batch x Classes)
        """
        if self.fdcm_matrix is None:
            return torch.tensor(0.0).to(self.device)

        # Ensure matrix is on correct device
        if self.fdcm_matrix.device != outputs.device:
            self.fdcm_matrix = self.fdcm_matrix.to(outputs.device)

        probs = torch.sigmoid(outputs)

        # Step 1: Calculate (p_i - p_j)^2 matrix for the batch
        # Expanding dimensions for broadcasting: (Batch, C, 1) and (Batch, 1, C)
        # This effectively computes the pairwise squared difference map
        p_squared = probs ** 2
        
        # Efficient calculation of pairwise squared differences: 
        # (a-b)^2 = a^2 + b^2 - 2ab
        # However, original code logic utilized direct expansion which is clearer for validation:
        p_squared_stack = p_squared.unsqueeze(-1).repeat(1, 1, self.num_classes) # (B, C, C)
        pT_squared_stack = p_squared_stack.transpose(1, 2)
        
        p_pT = torch.bmm(probs.unsqueeze(2), probs.unsqueeze(1)) # (B, C, C)
        two_p_pT = 2 * p_pT
        
        # Matrix of predicted co-occurrences (pairwise differences)
        predicted_cooccurrence = p_squared_stack - two_p_pT + pT_squared_stack
        
        # Step 2: Match with pre-defined FDCM matrix (Equation 1)
        # Expand target FDCM to match batch size
        label_cooccurrence_expanded = self.fdcm_matrix.unsqueeze(0).expand(probs.size(0), -1, -1)
        
        loss = F.mse_loss(predicted_cooccurrence, label_cooccurrence_expanded, reduction='mean')
        
        return loss

    def soa_kld_loss(self, outputs, outputs_rb):
        """
        Calculates KLD loss for Stochastic One-hot Activation (SOA).
        Constrains similarity between main features and random block features.
        Ref: Section 2.3, SOA strategy.
        """
        if outputs_rb is None:
            return torch.tensor(0.0).to(self.device)
            
        loss = F.kl_div(
            F.logsigmoid(outputs_rb / self.alpha_kl),
            F.logsigmoid(outputs / self.alpha_kl),
            reduction='sum',
            log_target=True
        ) * (self.alpha_kl ** 2) / outputs_rb.numel()
        
        return loss

    def forward(self, outputs, targets, epoch, max_epochs, outputs_rb=None):
        """
        Returns component losses for external aggregation.
        """
        # 1. UAML Loss
        loss_uaml, un_scores = self.uaml_loss(outputs, targets, epoch, max_epochs)
        
        # 2. FDCM Loss
        loss_fdcm = self.fdcm_loss(outputs)
        
        # 3. SOA (KLD) Loss
        loss_soa = self.soa_kld_loss(outputs, outputs_rb)
        
        return loss_uaml, loss_fdcm, loss_soa, un_scores