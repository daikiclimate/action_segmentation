import torch
import torch.nn as nn


class featModel(nn.Module):
    def __init__(self, input_channel=512):
        super(featModel, self).__init__()

        # input shape = (512, 8, 8)
        # if feature extraction model is vgg
        self.AdaptivePool = nn.AdaptiveAvgPool2d((8, 8))
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 100, kernel_size=3, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(100, 3, kernel_size=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(nn.Linear(3 * 8 * 8, 100), nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(100, 11)

    def forward(self, x):
        x = self.AdaptivePool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 3 * 8 * 8)
        x = self.fc(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    t = torch.tensor([0 for _ in range(512 * 8 * 8)], dtype=torch.float).reshape(
        1, 512, 8, 8
    )
    t = torch.tensor([0 for _ in range(512 * 9 * 9)], dtype=torch.float).reshape(
        1, 512, 9, 9
    )
    m = featModel()
    print(m(t).shape)
