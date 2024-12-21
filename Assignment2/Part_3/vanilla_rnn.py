import torch
import torch.nn as nn


class VanillaRNN(nn.Module):

    def __init__(self, input_length, input_dim, hidden_dim, output_dim):
        super(VanillaRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.Whx = nn.Linear(input_dim, hidden_dim)
        self.Whh = nn.Linear(hidden_dim, hidden_dim)
        self.Wph = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            h_t = torch.tanh(self.Whx(x_t) + self.Whh(h_t))
        o_t = self.Wph(h_t)
        y_t = torch.softmax(o_t, dim=1)
        return y_t

