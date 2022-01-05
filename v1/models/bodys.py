import torch.nn as nn

class ResUBody(nn.Module):
    def __init__(self,  filters):
        super(ResUBody, self).__init__()

        self.input_conv_layer = self._input_cov(filters[0],split = False)
        self.input_skip_layer = self._input_cov(filters[0],split = True)
        self.conv_layer_1 = ConvLayer(filters[0], filters[1], 2, 1)
        self.conv_layer_2 = ConvLayer(filters[1], filters[2], 2, 1)

        self.output_layer = ConvLayer(filters[2], filters[3], 2, 1)
        

    def forward(self, x):

        x_1 = self.input_conv_layer(x) + self.input_skip_layer(x)
        x_2 = self.conv_layer_1(x_1)
        x_3 = self.conv_layer_2(x_2)
        output = self.output_layer(x_3)

        return output , [x_1,x_2,x_3]

    def _input_cov(self,filters,split):

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

class ConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ConvLayer, self).__init__()

        ### TAKEN FROM INTERNET NEEDS REWORDING

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)