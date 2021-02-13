import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM_classifier(nn.Module):
    def __init__(
        self,
        width_mult,
        lstm_layers,
        lstm_hidden,
        bidirectional=True,
        dropout=True,
        device="cuda:0",
    ):
        super(LSTM_classifier, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device

        self.conv_1x1 = nn.Conv2d(512, 1, 1)

        self.rnn = nn.LSTM(
            int(64 * width_mult if width_mult > 1.0 else 64),
            self.lstm_hidden,
            self.lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        if self.bidirectional:
            self.lin = nn.Linear(2 * self.lstm_hidden, 9)
        else:
            self.lin = nn.Linear(self.lstm_hidden, 9)
        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (
                Variable(
                    torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden).to(
                        self.device
                    ),
                    requires_grad=True,
                ),
                Variable(
                    torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden).to(
                        self.device
                    ),
                    requires_grad=True,
                ),
            )
        else:
            return (
                Variable(
                    torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to(
                        self.device
                    ),
                    requires_grad=True,
                ),
                Variable(
                    torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to(
                        self.device
                    ),
                    requires_grad=True,
                ),
            )

    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        self.hidden = self.init_hidden(batch_size)

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.conv_1x1(c_in)

        # c_out = self.cnn(c_in)
        # c_out = c_out.mean(3).mean(2)
        # if self.dropout:
        #     c_out = self.drop(c_out)

        # LSTM forward
        r_in = c_out.view(batch_size, timesteps, -1)
        print(r_in.shape)

        r_out, states = self.rnn(r_in, self.hidden)
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, 9)

        return out


if __name__ == "__main__":
    model = LSTM_classifier(1, 1, 256).cuda()
    data = torch.rand((1, 64, 512, 8, 8)).cuda()
    out = model(data)
    print(out.shape)
