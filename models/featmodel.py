import torch
import torch.nn as nn
import torchvision.models as models


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
        self.fc = nn.Linear(self.hidden2 * 8 * 8, self.hidden1)
        self.activ = nn.ReLU()
        nn.init.kaiming_normal_(self.fc.weight)

        self.fc2 = nn.Linear(self.hidden1, 11)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.AdaptivePool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.hidden2 * 8 * 8)
        x = self.fc(x)
        x = self.activ(x)
        x = self.fc2(x)
        return x


class FeatModel(nn.Module):
    def __init__(self, input_channel=512, align_size=8):
        super(FeatModel, self).__init__()
        self.classifier = classifier(input_channel=input_channel, align_size=align_size)

    def forward(self, x):
        x = self.classifier(x)
        return x


class ImgModel(nn.Module):
    def __init__(self, input_channel=512, align_size=8):
        super(ImgModel, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16 = self.vgg16.features
        self.classifier = classifier(input_channel=input_channel, align_size=align_size)

    def forward(self, x):
        x = self.vgg16(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    t = torch.tensor([0 for _ in range(512 * 8 * 8)], dtype=torch.float).reshape(
        1, 512, 8, 8
    )
    t = torch.tensor([0 for _ in range(512 * 9 * 9)], dtype=torch.float).reshape(
        1, 512, 9, 9
    )
    m = FeatModel()
    # m = ImgModel()
    print(m)
    # print(m(t).shape)
