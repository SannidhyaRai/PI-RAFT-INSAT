import torch
import torch.nn as nn

class PressureHead(nn.Conv2d):
    """
    Vertical Cloud-Top Pressure (CTP) prediction head.
    Inherits directly from nn.Conv2d to preserve flat state_dict parameter names
    and guarantee checkpoint compatibility.
    
    Input shape:  [B, 128, H/8, W/8]
    Output shape: [B, 1, H/8, W/8]
    """
    def __init__(self, in_channels=128, out_channels=1, kernel_size=3, padding=1):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        # TODO: scientific validation required
        # Note: This prediction head currently receives no gradients from the loss function.
