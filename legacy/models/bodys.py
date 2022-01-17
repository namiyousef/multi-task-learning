import torch.nn as nn

class ResUBody(nn.Module):
    """
    Implements the ResUBody model. Deprecated.
    """

    def __init__(self, filters):
        super(ResUBody, self).__init__()

        self.L = len(filters)
        self.input_conv_layer = self._input_cov(filters[0], split=False)
        self.input_skip_layer = self._input_cov(filters[0], split=True)
        for i in range(self.L):
            current_filter = filters[i]
            next_filter = filters[i + 1]
            setattr(self, f'conv_layer_{i + 1}', ConvLayer(current_filter, next_filter, stride=2, padding=1))

    def forward(self, x):

        x = self.input_conv_layer(x) + self.input_skip_layer(x)
        skips = []
        for i in range(self.L - 1):
            skips.append(x)
            x = getattr(self, f'conv_layer_{i + 1}')(x)

        return x, skips

    # TODO this causes an issue with GPU/CPU. The layer is declared locally so it does not get bound to nn.Module. Fix for future releases
    def _input_cov(self, filters, split):

        def _call(_input):

            x = nn.Conv2d(3, filters, kernel_size=3, padding=1)(_input)

            if split == True:
                return x

            else:
                x = nn.BatchNorm2d(filters)(x)
                x = nn.ReLU()(x)
                x = nn.Conv2d(filters, filters, kernel_size=3, padding=1)(x)
                return x

        return _call


class ResUBodyNEW(nn.Module):
    def __init__(self, filters):
        super(ResUBodyNEW, self).__init__()

        self.L = len(filters)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.input_conv_layer = self._input_cov(filters[0], split=False)
        self.input_skip_layer = self._input_cov(filters[0], split=True)
        for i in range(self.L):
            current_filter = filters[i]
            next_filter = filters[i + 1]
            setattr(self, f'conv_layer_{i + 1}', ConvLayer(current_filter, next_filter, stride=2, padding=1))

    def forward(self, x):

        x = self.input_conv_layer(x)
        x = self.maxpool(x)
        skips = []
        for i in range(self.L - 1):
            skips.append(x)
            x = getattr(self, f'conv_layer_{i + 1}')(x)

        return x, skips

    def _input_cov(self, filters, split):

        def _call(_input):

            if split == True:
                return x

            else:
                x = nn.Conv2d(filters, filters, kernel_size=7, stride=2, padding=3,
                              bias=False)(_input)
                x = nn.BatchNorm2d(filters)(x)
                x = nn.ReLU()(x)

                return x

        return _call

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