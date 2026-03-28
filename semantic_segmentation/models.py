import torch.nn as nn


class VOCDecoderLinear(nn.Module):
    def __init__(self, config):
        super(VOCDecoderLinear, self).__init__()
        input_channels = config['ren']['architecture']['hidden_dim']
        output_channels = config['data']['voc_num_classes']
        self.bn = nn.BatchNorm2d(input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        z = self.bn(x)
        out = self.conv(z)
        return out
    

class ADEDecoderLinear(nn.Module):
    def __init__(self, config):
        super(ADEDecoderLinear, self).__init__()
        input_channels = config['ren']['architecture']['hidden_dim']
        output_channels = config['data']['ade_num_classes']
        self.bn = nn.BatchNorm2d(input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        z = self.bn(x)
        out = self.conv(z)
        return out