import torch
import torch.nn as nn
import torchvision.models as models
from featModel import featModel, imgModel


class stacking(nn.Module):
    def __init__(self, seq_length=64, hidden=1280):
        super(stacking, self).__init__()
        self.seq_length = seq_length
        self.model = featModel()
        self.liner = nn.Sequential(
            nn.Linear(seq_length * 11, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, seq_length * 11),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(self.seq_length * 11)
        x = self.liner(x)
        x = x.view(self.seq_length, 11)
        return x


if __name__ == "__main__":
    # f = torch.rand(64,11)
    f = torch.rand(64, 512, 8, 8)
    m = stacking()
    print(m(f).shape)
