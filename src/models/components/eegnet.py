import torch
import torch.nn as nn

class EEGNet(nn.Module):
    """
    EEGNet model implementation.
    Reference: Lawhern et al., EEGNet: A Compact Convolutional Neural Network
    for EEG-based Brain-Computer Interfaces.
    """
    def __init__(self, nb_classes, Chans=22, Samples=1125,
                 dropoutRate=0.5, kernLength=64, F1=8,
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout',
                 **kwargs):  # 接受 Hydra 配置中的额外字段（如 name）
        super(EEGNet, self).__init__()
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3),
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 16 // 2), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )

        self.flatten = nn.Flatten()
        
        # Calculate the size of the feature map after the convolutional blocks
        # Input: (1, Chans, Samples)
        # Block 1:
        #   Conv2d(1, F1, (1, kernLength)): (F1, Chans, Samples) (padding keeps time dim same approx)
        #   Conv2d(F1, F1*D, (Chans, 1)): (F1*D, 1, Samples)
        #   AvgPool2d((1, 4)): (F1*D, 1, Samples // 4)
        # Block 2:
        #   Conv2d(..., (1, 16)): (F1*D, 1, Samples // 4)
        #   Conv2d(..., F2, (1, 1)): (F2, 1, Samples // 4)
        #   AvgPool2d((1, 8)): (F2, 1, Samples // 32)
        
        # Exact calculation depends on padding implementation, but usually it's close to Samples // 32
        # Let's do a dummy forward pass to determine the size dynamically or calculate it precisely.
        # For simplicity and robustness, I'll calculate it based on the layers.
        
        # Block 1 output time dim:
        out_samples_1 = Samples // 4
        # Block 2 output time dim:
        out_samples_2 = out_samples_1 // 8
        
        self.dense_input_size = F2 * out_samples_2
        
        self.classifier = nn.Linear(self.dense_input_size, nb_classes)

    def forward(self, x):
        # x shape: (Batch, Chans, Samples)
        # Add channel dimension: (Batch, 1, Chans, Samples)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x