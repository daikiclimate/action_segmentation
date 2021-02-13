import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class classifier(nn.Module):
    def __init__(self, input_channel=512, align_size=8):
        super(classifier, self).__init__()
        self.AdaptivePool = nn.AdaptiveAvgPool2d((align_size, align_size))
        self.hidden1 = 100
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, self.hidden1, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden1),
            nn.ReLU(inplace=True),
        )
        self.hidden2 = 10
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.hidden1, self.hidden2, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.hidden2),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden2 * 8 * 8, self.hidden1), nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(self.hidden1, 11)

    def forward(self, x):
        x = self.AdaptivePool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.hidden2 * 8 * 8)
        x = self.fc(x)
        x2 = self.fc2(x)
        return x, x2


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        dim = 100
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList(
            [
                copy.deepcopy(
                    SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)
                )
                for s in range(num_stages - 1)
            ]
        )
        self.conv = classifier()

    # def forward(self, x, mask):
    def forward(
        self,
        x,
    ):
        # out = self.stage1(x, mask)
        batch_size, seq_length, feature_dim, w, h = x.size()
        x, clf_output = self.conv(x.view(batch_size * seq_length, feature_dim, w, h))
        x = x.view(batch_size, -1, seq_length)
        # x = x.view(batch_size, feature_dim*w*h, seq_length)
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1))
            # out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs, clf_output


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps))
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.conv2 = nn.Conv2d(512, 4, 1)

    # def forward(self, x, mask):
    def forward(self, x):
        # x = self.conv2(x)

        # x = x.view(1, -1, 4 * 8 * 8)
        # x = x.view( -1, 512 * 8 * 8, 1)
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
            # out = layer(out, mask)
        out = self.conv_out(out)
        # out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        # def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out
        # return (x + out) * mask[:, 0:1, :]


if __name__ == "__main__":
    num_stages, num_layers, num_f_maps, dim, num_classes = 2, 3, 4, 5, 8
    model = MultiStageModel(num_stages, num_layers, num_f_maps, dim, num_classes)
    mask = torch.zeros(num_f_maps, num_classes, 5, dtype=torch.float)

    f = torch.randn(1, 1000, 8, 8, 16)
    # f = torch.randn(4, 5, 1)
    print(f.shape)
    print(model(f).shape)
