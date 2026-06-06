import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes)
            )
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class FeatureEncoder(nn.Module):
    """
    Downsamples high-resolution satellite imagery bands by an overall factor of 8.
    
    Input shape:  [B, 1, H, W]
    Output shape: [B, output_dim, H/8, W/8]
    """
    def __init__(self, output_dim=128):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            self.norm1, self.relu
        )
        self.layer2 = ResidualBlock(64, 64, stride=2)   # 1/4 Scale Tier
        self.layer3 = ResidualBlock(64, 96, stride=2)   # 1/8 Scale Tier
        self.conv_out = nn.Conv2d(96, output_dim, kernel_size=1)

    def forward(self, x):
        return self.conv_out(self.layer3(self.layer2(self.layer1(x))))
