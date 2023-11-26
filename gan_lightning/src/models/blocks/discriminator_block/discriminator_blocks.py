from torch import nn


class simple_1d_discriminator_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(
            simple_1d_discriminator_block, self
        ).__init__()  # Call the superclass's __init__ method
        self.block = nn.Sequential(
            nn.Linear(input_channels, output_channels), nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)
