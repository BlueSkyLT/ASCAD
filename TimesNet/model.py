import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TimesNetConfig
from .layers.Conv_Blocks import Inception_Block_V1
from .layers.Embed import DataEmbedding


def fft_for_period(x, top_k=5):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = torch.abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, top_k)
    top_list = top_list.detach().cpu().numpy()
    period_list = x.shape[1] // top_list
    return period_list, torch.abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, config: TimesNetConfig):
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = 0
        self.k = config.top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(config.d_model, config.d_ff, num_kernels=config.num_kernels),
            nn.GELU(),
            Inception_Block_V1(config.d_ff, config.d_model, num_kernels=config.num_kernels),
        )

    def forward(self, x):
        batch_size, time_steps, channels = x.size()
        period_list, period_weight = fft_for_period(x, self.k)

        residuals = []
        total_length = self.seq_len + self.pred_len
        for idx in range(self.k):
            period = period_list[idx]
            if total_length % period != 0:
                length = ((total_length // period) + 1) * period
                padding = torch.zeros([x.shape[0], length - total_length, x.shape[2]], device=x.device, dtype=x.dtype)
                out = torch.cat([x, padding], dim=1)
            else:
                length = total_length
                out = x
            out = out.reshape(batch_size, length // period, period, channels).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
            residuals.append(out[:, :total_length, :])
        residuals = torch.stack(residuals, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, time_steps, channels, 1)
        residuals = torch.sum(residuals * period_weight, dim=-1)
        return residuals + x


class BaseTimesNetClassifier(nn.Module):
    def __init__(self, config: TimesNetConfig):
        super().__init__()
        self.config = config
        self.model = nn.ModuleList([TimesBlock(config) for _ in range(config.e_layers)])
        self.enc_embedding = DataEmbedding(
            c_in=config.enc_in,
            d_model=config.d_model,
            dropout=config.dropout,
            kernel_size=config.embedding_kernel_size,
        )
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.act = F.gelu
        self.dropout = nn.Dropout(config.dropout)
        self.projection = nn.Linear(config.d_model * config.seq_len, config.num_class)

    def forward_features(self, x, padding_mask=None):
        enc_out = self.enc_embedding(x, None)
        for block in self.model:
            enc_out = self.layer_norm(block(enc_out))
        output = self.act(enc_out)
        output = self.dropout(output)
        if padding_mask is not None:
            output = output * padding_mask.unsqueeze(-1)
        return output.reshape(output.shape[0], -1)

    def forward(self, x, padding_mask=None, *_args, **_kwargs):
        features = self.forward_features(x, padding_mask=padding_mask)
        return self.projection(features)


class TimesNetASCAD(BaseTimesNetClassifier):
    def __init__(self, config: TimesNetConfig):
        super().__init__(config)
        hidden_1 = config.head_hidden_dim or 1024
        hidden_2 = config.head_hidden_dim_2 or 1024
        self.projection = nn.Sequential(
            nn.Linear(config.d_model * config.seq_len, hidden_1),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(hidden_2, config.num_class),
        )


class TimesNetASCADHead(TimesNetASCAD):
    pass


def build_timesnet_model(variant='baseline', **kwargs):
    config = TimesNetConfig(**kwargs)
    if variant == 'baseline':
        return BaseTimesNetClassifier(config)
    if variant == 'ascad_head':
        return TimesNetASCADHead(config)
    raise ValueError(f"Unknown TimesNet variant: {variant}")
