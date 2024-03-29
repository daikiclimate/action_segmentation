import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.conv2,
            self.chomp2,
            self.relu2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=22):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, red_dim):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()
        self._red_dim = nn.Conv2d(512, red_dim, 1)
        self._use_attention = True
        self._use_attention = False
        if self._use_attention:
            self._nn_linear = nn.Linear(8 ** 3, 11)
            self._attention_liner = nn.Linear(8 ** 3, 11)
        self._lstm_attention = True
        self._sigmoid = nn.Sigmoid()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self._red_dim(x)

        s = x.size()
        if self._use_attention:
            nn_x = x.reshape(s[0], -1)
            attention = self._attention_liner(nn_x)
            nn_x = self._nn_linear(nn_x)
        if len(s) != 3:
            x = x.view(1, s[0], -1)
            x = x.permute([0, 2, 1]).contiguous()

        x = self.tcn(x)
        bs, seq, f = x.size()
        x = x.permute([0, 2, 1])
        x = x.view(bs * f, seq).contiguous()

        x = self.linear(x)
        if self._use_attention:
            attention = self._sigmoid(attention)
            x = nn_x * attention + x * (1 - attention)
        return x


if __name__ == "__main__":
    model = TCN(1000, 11, [20, 20])
    f = torch.randn(1, 1000, 20)
    print(f.shape)
    print(model(f).shape)
