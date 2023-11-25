from torch import nn


class simple_1d_generator_block(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = None,
        stride: int = None,
        final_layer: bool = False,
    ):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_channels, output_channels),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.generator(x)
