import torch
import torch.nn as nn
from timm import create_model
from torch.nn import functional as F

# --- NEW MODULE 1: The Depth-Wise Convolution Module ---
# This is the "DWConv module" from the diagram, adapted for 1D Temporal data.

# It now correctly applies GELU and BatchNorm *BEFORE* the convolution,
# as described in the paper's diagram/text.
class DWConvModule(nn.Module):
    def __init__(self, embed_dim, kernel_size=3):
        super().__init__()
        # Per the paper's diagram: GELU -> BN -> DWConv
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm1d(embed_dim)
        self.dwconv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, # 'same' padding
            groups=embed_dim, # This makes it "depth-wise"
        )

    def forward(self, x):
        # Input x has shape (Batch, NumFrames, EmbedDim)
        # Conv1d expects (Batch, EmbedDim, NumFrames)
        x = x.permute(0, 2, 1) 
        
        # Apply layers in the order from the paper's diagram
        x = self.gelu(x)
        x = self.bn(x)
        x = self.dwconv(x)
        
        # Permute back to (Batch, NumFrames, EmbedDim)
        x = x.permute(0, 2, 1)
        return x

# --- NEW MODULE 2: The Hybrid Transformer Block ---
# This block is IDENTICAL to my previous version, but now calls the *corrected*
# DWConvModule. It runs global attention (MHSA) and local convolution (DWConv)
# in parallel and mixes their outputs.
class HybridEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, temporal_dim=512, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # The parallel local expert (now the *corrected* module)
        self.dwconv = DWConvModule(embed_dim=embed_dim)
        
        # The FFN (Feed-Forward Network)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Implements the logic from the diagram
        
        # --- First Residual Connection (Global + Local) ---
        norm_x = self.norm1(x)
        
        # Path 1: Global Attention (the Transformer)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        
        # Path 2: Local Convolution (the DWConv)
        # Per diagram, input to DWConv is the *original* x (pre-norm)
        conv_out = self.dwconv(x) 
        
        # Mix the two paths and add the residual connection
        x = x + attn_out + conv_out # This is the "mix"
        
        # --- Second Residual Connection (FFN) ---
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x

# --- NEW MODULE 3: The Full Encoder Stack ---
# This just stacks multiple HybridEncoderLayers
class HybridEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, temporal_dim, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            HybridEncoderLayer(embed_dim, num_heads, temporal_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim) # Final normalization

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# --- FINAL STEP: Update the Main Model ---
class SwinTemporalClassifier(nn.Module):
    def __init__(self, num_classes=2, num_frames=60, temporal_dim=512, num_temporal_layers=2, num_heads=8):
        """
        swin + HYBRID (transformer + conv) classifier
        """
        # --- THIS IS THE TYPO FIX ---
        super(SwinTemporalClassifier, self).__init__() # Added ()

        # load swin transformer as base model for spatial feature extraction
        self.swin = create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=0)
        swin_dim = self.swin.num_features  # 1024

        # freeze most layers
        for param in self.swin.parameters():
            param.requires_grad = False
        for param in self.swin.layers[-1].parameters():
            param.requires_grad = True
        for param in self.swin.norm.parameters():
            param.requires_grad = True

        # learnable temporal position embeddings (1 for each frame)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, swin_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # Replace the old nn.TransformerEncoder with our new HybridEncoder
        self.temporal_transformer = HybridEncoder(
            embed_dim=swin_dim,
            num_heads=num_heads,
            temporal_dim=temporal_dim,
            num_layers=num_temporal_layers
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(swin_dim, temporal_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(temporal_dim, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        # flatten frames so swin can process them as images
        x = x.view(B * T, C, H, W)
        frame_features = self.swin(x)  # (B*T, swin_dim)
        swin_dim = frame_features.shape[-1]

        # reshape back to (batch, frames, features)
        frame_features = frame_features.view(B, T, swin_dim)

        # add temporal positional encoding so model knows frame order
        frame_features = frame_features + self.temporal_pos_embed[:, :T, :]

        # temporal transformer models how frames relate to each other
        temporal_out = self.temporal_transformer(frame_features)  # (B, T, swin_dim)

        # take mean of all frame embeddings to get a video-level feature
        video_features = temporal_out.mean(dim=1)  # (B, swin_dim)

        # final classifier for normal vs abnormal
        output = self.classifier(video_features)
        return output