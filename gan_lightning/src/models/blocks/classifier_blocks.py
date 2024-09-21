from torch import nn


class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs
    ):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


class MobileNetV2Bottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=4, stride=2, **kwargs):
        super(MobileNetV2Bottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class Mobilenetv2_Feature_Extractor(nn.Module):
    """Global feature extractor module"""

    def __init__(
        self,
        in_channels=64,
        block_channels=(64, 96, 128),
        out_channels=128,
        t=6,
        num_blocks=3,
        **kwargs,
    ):
        super(Mobilenetv2_Feature_Extractor, self).__init__()
        self.bottleneck1 = self._make_layer(
            MobileNetV2Bottleneck, in_channels, block_channels[0], 3, t, 2
        )
        self.bottleneck2 = self._make_layer(
            MobileNetV2Bottleneck, block_channels[0], block_channels[1], 3, t, 2
        )
        self.bottleneck3 = self._make_layer(
            MobileNetV2Bottleneck, block_channels[1], block_channels[2], 3, t, 1
        )

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = nn.ModuleList()
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        return x


class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, **kwargs):
        super(DownSamplingBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels * 4, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels * 2, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(True),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        return conv2, conv4


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes, **kwargs):
        super(ClassificationHead, self).__init__()
        self.conv1 = _ConvBNReLU(in_channels, in_channels // 2, 3, 1, 1)
        self.conv2 = _ConvBNReLU(in_channels // 2, in_channels // 4, 3, 1, 1)
        self.conv3 = _ConvBNReLU(in_channels // 4, in_channels // 8, 3, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_conv = nn.Conv2d(in_channels // 8, num_classes, 1)
        self.fc = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = self.final_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
