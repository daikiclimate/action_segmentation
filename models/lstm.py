import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        width_mult,
        lstm_layers,
        lstm_hidden,
        bidirectional=True,
    ):
        super(LSTMClassifier, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional

        self.conv_1x1 = nn.Conv2d(512, 1, 1)

        self.rnn = nn.LSTM(
            int(64 * width_mult if width_mult > 1.0 else 64),
            self.lstm_hidden,
            self.lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        if self.bidirectional:
            self.lin = nn.Linear(2 * self.lstm_hidden, 11)
        else:
            self.lin = nn.Linear(self.lstm_hidden, 11)

        self._nn_liner = nn.Linear(64, 11)
        self._attention_liner = nn.Linear(64, 11 * 2)
        self._attention_liner = nn.Linear(64, 11)
        self._lstm_attention = True
        self._softmax = nn.Softmax(dim=3)
        self._sigmoid = nn.Sigmoid()

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (
                Variable(
                    torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden),
                    requires_grad=True,
                ).cuda(),
                Variable(
                    torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden),
                    requires_grad=True,
                ).cuda(),
            )
        else:
            return (
                Variable(
                    torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden),
                    requires_grad=True,
                ).cuda(),
                Variable(
                    torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden),
                    requires_grad=True,
                ).cuda(),
            )

    def forward(self, x, lengths=None):
        x = x.unsqueeze(0)
        batch_size, timesteps, C, H, W = x.size()
        self.hidden = self.init_hidden(batch_size)

        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.conv_1x1(c_in)

        r_in = c_out.view(batch_size, timesteps, -1).contiguous()
        r_out, states = self.rnn(r_in, self.hidden)
        out = self.lin(r_out)
        if self._lstm_attention:
            out1 = self._nn_liner(r_in)
            attention = self._attention_liner(r_in)
            attention = self._sigmoid(attention)
            # attention = attention.view(batch_size, timesteps, -1, 2)
            # attention = self._softmax(attention)
            out = out1 * attention + out * (1 - attention)

        out = out.view(batch_size * timesteps, 11)
        return out


if __name__ == "__main__":
    model = LSTMClassifier(1, 1, 256).cuda()
    data = torch.rand((1, 64, 512, 8, 8)).cuda()
    out = model(data)
    print(out.shape)
