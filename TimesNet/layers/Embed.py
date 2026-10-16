import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode='circular',
            bias=False,
        )
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, kernel_size=3, max_len=5000):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, kernel_size=kernel_size)
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        del x_mark
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
