"""
Wide ResNet-34-10 for CIFAR-sized inputs.

Zagoruyko and Komodakis, Wide Residual Networks, BMVC 2016.
The 3 x 3 stem uses stride 1 without max pooling. Group widths are
16, 160, 320, and 640; each residual group has five blocks.
The received implementation applies dropout after the second convolution.

Local reproduction edition, 2026-11-01: documentation only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WRNBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_io = in_planes == out_planes and stride == 1
        self.shortcut = (nn.Identity() if self.equal_io
                         else nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False))

    def forward(self, x):
        shortcut = x if self.equal_io else self.shortcut(x)
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out + shortcut


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, stride, drop_rate):
        super().__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(WRNBasicBlock(in_planes if i == 0 else out_planes, out_planes,
                                        stride if i == 0 else 1, drop_rate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet34_10(nn.Module):
    def __init__(self, num_classes=10, depth=34, widen=10, drop_rate=0.3):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth must be 6n+4"
        n = (depth - 4) // 6
        chans = [16, 16 * widen, 32 * widen, 64 * widen]
        self.conv1 = nn.Conv2d(3, chans[0], 3, padding=1, bias=False)
        self.block1 = NetworkBlock(n, chans[0], chans[1], 1, drop_rate)
        self.block2 = NetworkBlock(n, chans[1], chans[2], 2, drop_rate)
        self.block3 = NetworkBlock(n, chans[2], chans[3], 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(chans[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(chans[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)
