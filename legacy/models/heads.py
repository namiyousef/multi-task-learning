import torch.nn as nn
import torch


class ClassificationHead(nn.Sequential):
    def __init__(self, n_input_features, num_classes):
        super(ClassificationHead, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(n_input_features, num_classes)

    def forward(self, inputs, skips):
        x = self.global_avg_pool(inputs)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        return x.double()


class BBHead(nn.Sequential):
    def __init__(self, n_input_features, num_classes):
        super(BBHead, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(n_input_features, num_classes)

    def forward(self, inputs, skips):
        x = self.global_avg_pool(inputs)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        return x.double()


class SegmentationHead(nn.Module):
    def __init__(self, filters):
        super(SegmentationHead, self).__init__()

        filters = filters[::-1]

        for i, filter in enumerate(filters, 1):
            setattr(self, f'upsample{i}', nn.ConvTranspose2d(filter, filter, kernel_size=2, stride=2))
            if i < len(filters):
                setattr(self, f'decode_conv{i}', ConvLayer(filter + filters[i], filters[i], stride=1, padding=1))

        setattr(self, f'decode_conv{i}', ConvLayer(2 * filter, filter, stride=1, padding=1))
        setattr(self, f'decode_conv{i + 1}', ConvLayer(filter, filter, stride=1, padding=1))

        self.output_layer = nn.Sequential(
            nn.Conv2d(filter, 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, skips):
        skips = skips[::-1]
        for i, skip in enumerate(skips, 1):
            x = getattr(self, f'upsample{i}')(x)
            x = getattr(self, f'decode_conv{i}')(torch.cat([x, skip], dim=1))
        x = getattr(self, f'upsample{i}')(x)
        x = getattr(self, f'decode_conv{i + 1}')(x)
        output = self.output_layer(x)
        return output


class ConvLayer(nn.Module):

    def __init__(self, n_input_channels, n_output_channels, **kwargs):
        super(ConvLayer, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(n_input_channels),
            nn.ReLU(),
            nn.Conv2d(n_input_channels, n_output_channels, kernel_size=3, **kwargs),
            nn.BatchNorm2d(n_output_channels),
            nn.ReLU(),
            nn.Conv2d(n_output_channels, n_output_channels, kernel_size=3, **kwargs),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(n_input_channels, n_output_channels, kernel_size=3, **kwargs),
            nn.BatchNorm2d(n_output_channels),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)