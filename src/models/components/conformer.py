import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGConformer(nn.Module):
    """
    EEG Conformer implementation.
    Reference: Song et al., "EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization"
    """
    def __init__(self, n_classes, channels, samples, emb_size=40, depth=6, heads=8, kernel_size=25, dropout=0.5, **kwargs):
        super(EEGConformer, self).__init__()
        self.n_classes = n_classes
        self.channels = channels
        self.samples = samples
        self.emb_size = emb_size
        
        # 1. Convolutional Module (Temporal + Spatial)
        # Temporal Convolution
        self.conv1 = nn.Conv2d(1, emb_size, (1, kernel_size), padding=(0, kernel_size // 2))
        self.bn1 = nn.BatchNorm2d(emb_size)
        
        # Spatial Convolution
        self.conv2 = nn.Conv2d(emb_size, emb_size, (channels, 1))
        self.bn2 = nn.BatchNorm2d(emb_size)
        
        # Pooling
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.drop = nn.Dropout(dropout)
        
        # Calculate output sequence length
        # Input: (Samples)
        # Conv1: (Samples) (padding keeps size approx same)
        # Conv2: (Samples)
        # Pool: (Samples - 75) / 15 + 1
        # Note: We need to be careful with exact padding in Conv1 to match dimensions if needed,
        # but standard Conv2d with padding=k//2 is usually sufficient.
        # Let's calculate dynamically in forward or pre-calc here.
        
        # Assuming input is exactly 'samples' long.
        # Conv1 output width = samples (due to padding)
        # Pool output width:
        self.out_samples = (samples - 75) // 15 + 1
        
        # 2. Transformer Module
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(self.out_samples + 1, emb_size))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, 
            nhead=heads, 
            dim_feedforward=emb_size * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 3. Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        # x: (Batch, Channels, Samples)
        if x.dim() == 3:
            x = x.unsqueeze(1) # (Batch, 1, Channels, Samples)
            
        # Conv Module
        x = self.conv1(x)           # (B, Emb, C, T)
        x = self.bn1(x)
        x = F.elu(x)
        
        x = self.conv2(x)           # (B, Emb, 1, T)
        x = self.bn2(x)
        x = F.elu(x)
        
        x = self.pool(x)            # (B, Emb, 1, T')
        x = self.drop(x)
        
        x = x.squeeze(2)            # (B, Emb, T')
        x = x.permute(0, 2, 1)      # (B, T', Emb)
        
        # Transformer Module
        b, t, e = x.shape
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, T'+1, Emb)
        
        # Add positional embeddings
        # Ensure positions matches the current sequence length (handle potential minor size mismatch due to rounding)
        if t + 1 > self.positions.size(0):
             # Fallback or error if sequence is longer than expected
             # For now, we assume fixed input size as per EEGNet/Conformer standard usage
             pass
        
        x += self.positions[:t+1, :].unsqueeze(0)
        
        x = self.transformer(x)
        
        # Classification
        cls_out = x[:, 0, :]
        return self.classifier(cls_out)