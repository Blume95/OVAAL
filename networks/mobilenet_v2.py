import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


def conv_bn(inp_c, oup_c, stride, BatchNorm):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp_c, out_channels=oup_c, kernel_size=3, stride=stride, padding=1, bias=False),
        BatchNorm(oup_c),
        nn.ReLU6(inplace=True)
    )


def fixed_padding(inputs, kernel_size, dilation):
    """
    Make sure output tensor size will not affect by kernel size
    :param inputs:
    :param kernel_size:
    :param dilation:
    :return:
    """
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ration, BatchNorm):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ration)

        self.use_res_connect = self.stride == 1 and inp == oup  # Two Types Blocks one for residual connection and the
        # other for down sampling.
        self.kernel_size = 3
        self.dilation = dilation

        if expand_ration == 1:  # don't need point-wise to reshape the tensor depth
            self.conv = nn.Sequential(
                # depth-wise convolution kernel size is always equal 3, padding will be done before convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # point-wise linear convolution(in paper https://arxiv.org/pdf/1801.04381.pdf show linear conv has
                # better performance )
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, 1, bias=False),
                BatchNorm(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, 1, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # linear pw
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, 1, bias=False),
                BatchNorm(oup),
            )

    def forward(self, x):
        x_pad = fixed_padding(x, self.kernel_size, dilation=self.dilation)
        if self.use_res_connect:
            x = x + self.conv(x_pad)
        else:
            x = self.conv(x_pad)
        return x


class MobileNetV2(nn.Module):
    def __init__(self,
                 output_stride=16,
                 BatchNorm=None,
                 pretrained=True, ):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        # All these parameters come from MobileNetv2 https://arxiv.org/pdf/1801.04381.pdf
        input_channel = 32
        current_stride = 1
        rate = 1
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # current_stride=2
            [6, 24, 2, 2],  # current_stride=2
            [6, 32, 3, 2],  # current_stride=4
            [6, 64, 4, 2],  # current_stride=8 dilation = 1
            [6, 96, 3, 1],  # current_stride=16 dilation = 1
            [6, 160, 3, 2],  # current_stride=16 dilation = 2
            [6, 320, 1, 1],  # current_stride=16 dilation = 2
        ]
        # first layer
        # 512x512x3 -> 256x256x32 stride =2
        self.features = [conv_bn(3, input_channel, 2, BatchNorm)]
        current_stride *= 2

        # bottleneck inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            if current_stride == output_stride:
                stride = 1
                dilation = rate
                rate *= s
            else:
                stride = s
                dilation = 1
                current_stride *= s
            output_channel = c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, stride, dilation, t, BatchNorm))
                else:
                    self.features.append(block(input_channel, output_channel, 1, dilation, t, BatchNorm))
                input_channel = output_channel

        self.features = nn.Sequential(*self.features)
        self._initialize_weights()

        if pretrained:
            print("Loading pretrain weights")
            self._load_pretrained_model()
        else:
            print("Without pretrain weights")

        self.low_level_features = self.features[0:4]
        self.high_level_features = self.features[4:]

    def forward(self, x):
        low_level_feat = self.low_level_features(x)
        x = self.high_level_features(low_level_feat)

        return x, low_level_feat

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('http://jeff95.me/models/mobilenet_v2-6a65762b.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


if __name__ == "__main__":
    input = torch.rand(1, 3, 224, 224)
    model = MobileNetV2(output_stride=16, BatchNorm=nn.BatchNorm2d)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    # output, low_level_features = model(input)
    # print(output.size())
    # print(low_level_features.size())
