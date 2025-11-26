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

# models/vit_adapter.py
import torch
import torch.nn as nn
from functools import partial
import timm
from timm.models.vision_transformer import VisionTransformer, Block

class AKU_Adapter(nn.Module):
    """
    Lightweight Adapter module for fine-tuning Foundation Models.
    Corresponds to the 'Test-time Adapter' in RetExpert AKUhitecture.
    Structure: Down-Projection -> Activation -> Dropout -> Up-Projection
    """
    def __init__(self, adapter_dim, hidden_dim, dropout=0.1, position='att'):
        super().__init__()
        # Rescale parameter (initialized to 0 or Xavier depending on position)
        self.adapter_rescale = nn.Parameter(torch.empty(1, adapter_dim))
        
        # Bias parameters
        self.adapter_bias = nn.Parameter(torch.empty(hidden_dim))
        self.adapter_bias_b = nn.Parameter(torch.empty(adapter_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Initialization strategy
        if position == 'att':
            nn.init.zeros_(self.adapter_rescale)
        else:
            nn.init.xavier_uniform_(self.adapter_rescale)
        nn.init.zeros_(self.adapter_bias)
        nn.init.zeros_(self.adapter_bias_b)

    def forward(self, x, down_projection, up_projection):
        # Linear layer 1: Down-projection
        # Using functional matmul with external parameters (KU optimization strategy)
        adapter_output = torch.matmul(x, down_projection * self.adapter_rescale) + self.adapter_bias_b
        
        adapter_output = self.activation(adapter_output)
        adapter_output = self.dropout(adapter_output)
        
        # Linear layer 2: Up-projection
        adapter_output = torch.matmul(adapter_output, up_projection) + self.adapter_bias
        
        # Residual connection
        output = adapter_output + x
        return output

class RetExpertViT(VisionTransformer):
    """
    Vision Transformer modified for RetExpert framework.
    Features:
    1. Integrated AKU Adapters in Attention and MLP blocks.
    2. Returns intermediate outputs for SOA (Stochastic One-hot Activation) strategy.
    """
    def __init__(self, adapter_mode='AKU', adapter_dim=64, **kwargs):
        super().__init__(**kwargs)
        
        self.adapter_mode = adapter_mode # 'AKU' (Attn+MLP), 'AKU_att' (Attn only), or None
        self.depth = len(self.blocks)
        
        # Initialize Adapters and Projections if tuning is enabled
        if self.adapter_mode in ['AKU_att', 'AKU']:
            self._init_adapters(adapter_dim)

    def _init_adapters(self, adapter_dim):
        """Initialize adapter modules and projection parameters."""
        embed_dim = self.embed_dim
        
        self.att_adapters = nn.ModuleList()
        self.att_down_projections = nn.ParameterList()
        self.att_up_projections = nn.ParameterList()
        
        if self.adapter_mode == 'AKU':
            self.mlp_adapters = nn.ModuleList()
            self.mlp_down_projections = nn.ParameterList()
            self.mlp_up_projections = nn.ParameterList()

        for _ in range(self.depth):
            # Attention Adapter
            self.att_adapters.append(AKU_Adapter(adapter_dim, embed_dim, position='att'))
            
            # Projections for Attn Adapter
            att_down = nn.Parameter(torch.empty(embed_dim, adapter_dim))
            att_up = nn.Parameter(torch.empty(adapter_dim, embed_dim))
            nn.init.xavier_uniform_(att_down)
            nn.init.xavier_uniform_(att_up)
            self.att_down_projections.append(att_down)
            self.att_up_projections.append(att_up)

            # MLP Adapter (Only for 'AKU' mode)
            if self.adapter_mode == 'AKU':
                self.mlp_adapters.append(AKU_Adapter(adapter_dim, embed_dim, position='mlp'))
                
                # Projections for MLP Adapter
                mlp_down = nn.Parameter(torch.empty(embed_dim, adapter_dim))
                mlp_up = nn.Parameter(torch.empty(adapter_dim, embed_dim))
                nn.init.xavier_uniform_(mlp_down)
                nn.init.xavier_uniform_(mlp_up)
                self.mlp_down_projections.append(mlp_down)
                self.mlp_up_projections.append(mlp_up)

    def forward_features(self, x):
        """
        Modified forward pass to insert Adapters (Knowledge Units).
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        outputlist = [] # Stores CLS tokens from each block for SOA strategy

        for i, blk in enumerate(self.blocks):
            # 1. Attention Block with Adapter
            x_resid = x
            x = blk.norm1(x)
            
            # Insert Attention Adapter
            if self.adapter_mode in ['AKU_att', 'AKU']:
                x = self.att_adapters[i](x, self.att_down_projections[i], self.att_up_projections[i])
            
            x = blk.attn(x)
            x = blk.ls1(x) # LayerScale if exists
            x = blk.drop_path1(x)
            x = x + x_resid
            
            # 2. MLP Block with Adapter
            x_resid = x
            x = blk.norm2(x)
            
            # Insert MLP Adapter
            if self.adapter_mode == 'AKU':
                x = self.mlp_adapters[i](x, self.mlp_down_projections[i], self.mlp_up_projections[i])
            
            x = blk.mlp(x)
            x = blk.ls2(x)
            x = blk.drop_path2(x)
            x = x + x_resid
            
            # Collect output for SOA (Random Block Loss)
            # We take the CLS token (index 0) and pass it through the head for dimension alignment if needed
            # Typically SOA uses feature similarity, here we store the feature before final norm
            # Note: Original code does self.head(x[:, 0]) inside loop, which implies 'head' is shared or used for projection
            outputlist.append(self.head(x[:, 0]))

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome, outputlist

    def forward(self, x):
        x, outputlist = self.forward_features(x)
        x = self.head(x)
        return x, outputlist

def build_retexpert_vit_large(num_classes=20, adapter_mode='AKU', pretrained=True, **kwargs):
    """
    Builder function for RetExpert-ViT-Large (Patch 16).
    """
    model = RetExpertViT(
        patch_size=16, 
        embed_dim=1024, 
        depth=24, 
        num_heads=16, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        num_classes=num_classes,
        adapter_mode=adapter_mode,
        **kwargs
    )
    return model