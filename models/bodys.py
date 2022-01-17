import torch.nn as nn


class ConvLayer(nn.Module):
    """Implements a convolutional layer block with 2 BN-ReLu-Conv layers and a ConvSkip layer
    """
    def __init__(self, n_input_channels, n_output_channels, **kwargs):
        super(ConvLayer, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(n_input_channels),
            nn.ReLU(),
            nn.Conv2d(n_input_channels, n_output_channels, kernel_size=3, **kwargs),
            nn.BatchNorm2d(n_output_channels),
            nn.ReLU(),
            nn.Conv2d(n_output_channels, n_output_channels, **kwargs),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(n_input_channels, n_output_channels, kernel_size=3, **kwargs),
            nn.BatchNorm2d(n_output_channels),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)
