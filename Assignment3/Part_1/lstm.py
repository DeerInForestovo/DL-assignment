import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.Wgx = nn.Linear(input_dim, hidden_dim)
        self.Wgh = nn.Linear(hidden_dim, hidden_dim)
        self.Wix = nn.Linear(input_dim, hidden_dim)
        self.Wih = nn.Linear(hidden_dim, hidden_dim)
        self.Wfx = nn.Linear(input_dim, hidden_dim)
        self.Wfh = nn.Linear(hidden_dim, hidden_dim)
        self.Wox = nn.Linear(input_dim, hidden_dim)
        self.Woh = nn.Linear(hidden_dim, hidden_dim)
        self.Wph = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        h_t = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        c_t = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            g_t = torch.tanh(self.Wgx(x_t) + self.Wgh(h_t))
            i_t = torch.sigmoid(self.Wix(x_t) + self.Wih(h_t))
            f_t = torch.sigmoid(self.Wfx(x_t) + self.Wfh(h_t))
            o_t = torch.sigmoid(self.Wox(x_t) + self.Woh(h_t))
            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t
        p_t = self.Wph(h_t)
        return self.softmax(p_t)

