import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TimesNetConfig
from .layers.Conv_Blocks import Inception_Block_V1
from .layers.Embed import DataEmbedding


def _normalize_pool_mode(pool_mode: str | None) -> str:
    mode = (pool_mode or "avg").lower()
    if mode not in ("avg", "max"):
        raise ValueError(f"pool_mode must be 'avg' or 'max', got {pool_mode!r}")
    return mode


def temporal_pool(seq_features, padding_mask, pool_segments, segment_proj, pool_mode="avg"):
    """Pool over time: (B, L, d) -> (B, d) after optional K-bin adaptive pool + project."""
    mode = _normalize_pool_mode(pool_mode)
    if pool_segments and pool_segments > 0:
        x = seq_features.transpose(1, 2)
        if mode == "max":
            x = F.adaptive_max_pool1d(x, pool_segments)
        else:
            x = F.adaptive_avg_pool1d(x, pool_segments)
        x = x.reshape(x.size(0), -1)
        return segment_proj(x)
    if mode == "max":
        if padding_mask is None:
            return seq_features.max(dim=1).values
        fill = torch.finfo(seq_features.dtype).min
        masked = seq_features.masked_fill(~padding_mask.unsqueeze(-1).bool(), fill)
        return masked.max(dim=1).values
    if padding_mask is None:
        return seq_features.mean(dim=1)
    w = padding_mask.unsqueeze(-1).to(dtype=seq_features.dtype)
    denom = w.sum(dim=1).clamp_min(1.0)
    return (seq_features * w).sum(dim=1) / denom


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
        # Prefer actual sequence length (config.seq_len may lag if input is cropped).
        total_length = time_steps + self.pred_len
        for idx in range(self.k):
            period = int(period_list[idx])
            if period < 1:
                period = 1
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
        self.input_stride = max(1, int(config.input_stride or 1))
        self.pool_segments = int(config.pool_segments or 0)
        self.pool_mode = _normalize_pool_mode(config.pool_mode)
        self.model = nn.ModuleList([TimesBlock(config) for _ in range(config.e_layers)])
        self.enc_embedding = DataEmbedding(
            c_in=config.enc_in,
            d_model=config.d_model,
            dropout=config.dropout,
            kernel_size=config.embedding_kernel_size,
            max_len=max(config.seq_len, 5000),
        )
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.act = F.gelu
        self.dropout = nn.Dropout(config.dropout)
        in_dim = config.d_model * (self.pool_segments if self.pool_segments > 0 else config.seq_len)
        self.projection = nn.Linear(in_dim, config.num_class)

    def _maybe_stride(self, x, padding_mask=None):
        if self.input_stride <= 1:
            return x, padding_mask
        x = x[:, :: self.input_stride, :]
        if padding_mask is not None:
            padding_mask = padding_mask[:, :: self.input_stride]
        return x, padding_mask

    def forward_features(self, x, padding_mask=None):
        x, padding_mask = self._maybe_stride(x, padding_mask)
        enc_out = self.enc_embedding(x, None)
        for block in self.model:
            enc_out = self.layer_norm(block(enc_out))
        output = self.act(enc_out)
        output = self.dropout(output)
        if padding_mask is not None:
            output = output * padding_mask.unsqueeze(-1)
        if self.pool_segments > 0:
            if self.pool_mode == "max":
                pooled = F.adaptive_max_pool1d(output.transpose(1, 2), self.pool_segments).transpose(1, 2)
            else:
                pooled = F.adaptive_avg_pool1d(output.transpose(1, 2), self.pool_segments).transpose(1, 2)
            return pooled.reshape(pooled.shape[0], -1)
        return output.reshape(output.shape[0], -1)

    def forward(self, x, padding_mask=None, *_args, **_kwargs):
        features = self.forward_features(x, padding_mask=padding_mask)
        return self.projection(features)


class TimesNetASCAD(BaseTimesNetClassifier):
    def __init__(self, config: TimesNetConfig):
        super().__init__(config)
        hidden_1 = config.head_hidden_dim or 1024
        hidden_2 = config.head_hidden_dim_2 or 1024
        in_dim = config.d_model * (self.pool_segments if self.pool_segments > 0 else config.seq_len)
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_1),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(hidden_2, config.num_class),
        )


class TimesNetASCADHead(TimesNetASCAD):
    pass


class _BranchHead(nn.Module):
    """Lightweight MultiResNet-style branch (used when simple_heads=False)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1024)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(1024)
        self.out = nn.Linear(1024, out_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        return self.out(x)


class TimesNetMultiTask(nn.Module):
    """TimesNet encoder + temporal pool + multilabel heads for ASCADv2.

    Output keys match MultiResNet / MULTILABEL_KEYS so train/test recombination
    paths work unchanged.

    Optional:
      - input_stride: subsample time before encoder
      - pool_segments: adaptive pool to K bins (keeps coarse time structure)
      - pool_mode: "avg" (default) or "max"
    """

    def __init__(self, config: TimesNetConfig, use_dense_trunk: bool = False):
        super().__init__()
        self.config = config
        self.use_dense_trunk = use_dense_trunk
        self.without_permind = config.without_permind
        self.input_stride = max(1, int(config.input_stride or 1))
        self.pool_segments = int(config.pool_segments or 0)
        self.pool_mode = _normalize_pool_mode(config.pool_mode)

        self.model = nn.ModuleList([TimesBlock(config) for _ in range(config.e_layers)])
        self.enc_embedding = DataEmbedding(
            c_in=config.enc_in,
            d_model=config.d_model,
            dropout=config.dropout,
            kernel_size=config.embedding_kernel_size,
            max_len=max(config.seq_len, 5000),
        )
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.act = F.gelu
        self.dropout = nn.Dropout(config.dropout)

        if self.pool_segments > 0:
            self.segment_proj = nn.Linear(config.d_model * self.pool_segments, config.d_model)
        else:
            self.segment_proj = None

        feat_dim = config.d_model
        if use_dense_trunk:
            hidden_1 = config.head_hidden_dim or 1024
            hidden_2 = config.head_hidden_dim_2 or 1024
            self.trunk = nn.Sequential(
                nn.Linear(config.d_model, hidden_1),
                nn.ReLU(),
                nn.Dropout(config.head_dropout),
                nn.Linear(hidden_1, hidden_2),
                nn.ReLU(),
                nn.Dropout(config.head_dropout),
            )
            feat_dim = hidden_2
        else:
            self.trunk = nn.Identity()

        def make_head(out_dim):
            if config.simple_heads:
                return nn.Linear(feat_dim, out_dim)
            return _BranchHead(feat_dim, out_dim)

        self.alpha = make_head(256)
        self.beta = make_head(256)
        self.sbox = nn.ModuleList([make_head(256) for _ in range(16)])
        if not self.without_permind:
            self.permind = nn.ModuleList([make_head(16) for _ in range(16)])
        else:
            self.permind = None

    def _maybe_stride(self, x, padding_mask=None):
        if self.input_stride <= 1:
            return x, padding_mask
        x = x[:, :: self.input_stride, :]
        if padding_mask is not None:
            padding_mask = padding_mask[:, :: self.input_stride]
        return x, padding_mask

    def encode(self, x, padding_mask=None):
        """Return sequence features (B, L_eff, d_model)."""
        enc_out = self.enc_embedding(x, None)
        for block in self.model:
            enc_out = self.layer_norm(block(enc_out))
        output = self.act(enc_out)
        output = self.dropout(output)
        if padding_mask is not None:
            output = output * padding_mask.unsqueeze(-1)
        return output

    def pool_features(self, seq_features, padding_mask=None):
        """Pool over time -> (B, d_model)."""
        return temporal_pool(
            seq_features,
            padding_mask,
            self.pool_segments,
            self.segment_proj,
            self.pool_mode,
        )

    def forward(self, x, padding_mask=None, *_args, **_kwargs):
        x, padding_mask = self._maybe_stride(x, padding_mask)
        seq = self.encode(x, padding_mask=padding_mask)
        pooled = self.pool_features(seq, padding_mask=padding_mask)
        feat = self.trunk(pooled)
        outputs = {
            'alpha_output': self.alpha(feat),
            'beta_output': self.beta(feat),
        }
        for i in range(16):
            outputs[f'sbox_{i}_output'] = self.sbox[i](feat)
        if self.permind is not None:
            for i in range(16):
                outputs[f'permind_{i}_output'] = self.permind[i](feat)
        return outputs


class TimesNetASCADHeadMulti(TimesNetMultiTask):
    def __init__(self, config: TimesNetConfig):
        super().__init__(config, use_dense_trunk=True)


class TimesNetPooledClassifier(nn.Module):
    """Single-label TimesNet with temporal pool for long traces (e.g. ASCADv2 L=15k).

    Reuses the multi-task encoder/pool path but emits a single 256-way logit head.
    Input shape: ``(B, L, 1)``. Optional ``input_stride`` / ``pool_segments`` as in
    ``TimesNetMultiTask``.
    """

    def __init__(self, config: TimesNetConfig, use_dense_trunk: bool = False):
        super().__init__()
        self.config = config
        self.use_dense_trunk = use_dense_trunk
        self.input_stride = max(1, int(config.input_stride or 1))
        self.pool_segments = int(config.pool_segments or 0)
        self.pool_mode = _normalize_pool_mode(config.pool_mode)

        self.model = nn.ModuleList([TimesBlock(config) for _ in range(config.e_layers)])
        self.enc_embedding = DataEmbedding(
            c_in=config.enc_in,
            d_model=config.d_model,
            dropout=config.dropout,
            kernel_size=config.embedding_kernel_size,
            max_len=max(config.seq_len, 5000),
        )
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.act = F.gelu
        self.dropout = nn.Dropout(config.dropout)

        if self.pool_segments > 0:
            self.segment_proj = nn.Linear(config.d_model * self.pool_segments, config.d_model)
        else:
            self.segment_proj = None

        feat_dim = config.d_model
        if use_dense_trunk:
            hidden_1 = config.head_hidden_dim or 1024
            hidden_2 = config.head_hidden_dim_2 or 1024
            self.trunk = nn.Sequential(
                nn.Linear(config.d_model, hidden_1),
                nn.ReLU(),
                nn.Dropout(config.head_dropout),
                nn.Linear(hidden_1, hidden_2),
                nn.ReLU(),
                nn.Dropout(config.head_dropout),
            )
            feat_dim = hidden_2
        else:
            self.trunk = nn.Identity()

        self.classifier = nn.Linear(feat_dim, config.num_class)

    def _maybe_stride(self, x, padding_mask=None):
        if self.input_stride <= 1:
            return x, padding_mask
        x = x[:, :: self.input_stride, :]
        if padding_mask is not None:
            padding_mask = padding_mask[:, :: self.input_stride]
        return x, padding_mask

    def encode(self, x, padding_mask=None):
        enc_out = self.enc_embedding(x, None)
        for block in self.model:
            enc_out = self.layer_norm(block(enc_out))
        output = self.act(enc_out)
        output = self.dropout(output)
        if padding_mask is not None:
            output = output * padding_mask.unsqueeze(-1)
        return output

    def pool_features(self, seq_features, padding_mask=None):
        return temporal_pool(
            seq_features,
            padding_mask,
            self.pool_segments,
            self.segment_proj,
            self.pool_mode,
        )

    def forward(self, x, padding_mask=None, *_args, **_kwargs):
        x, padding_mask = self._maybe_stride(x, padding_mask)
        seq = self.encode(x, padding_mask=padding_mask)
        pooled = self.pool_features(seq, padding_mask=padding_mask)
        feat = self.trunk(pooled)
        return self.classifier(feat)


class TimesNetASCADHeadPooled(TimesNetPooledClassifier):
    def __init__(self, config: TimesNetConfig):
        super().__init__(config, use_dense_trunk=True)


def build_timesnet_model(variant='baseline', **kwargs):
    config = TimesNetConfig(**kwargs)
    if variant == 'baseline':
        return BaseTimesNetClassifier(config)
    if variant == 'ascad_head':
        return TimesNetASCADHead(config)
    if variant in ('pooled', 'timesnet_pooled'):
        return TimesNetPooledClassifier(config, use_dense_trunk=False)
    if variant in ('ascad_head_pooled', 'timesnet_ascad_head_pooled'):
        return TimesNetASCADHeadPooled(config)
    if variant in ('multi', 'timesnet_multi'):
        return TimesNetMultiTask(config, use_dense_trunk=False)
    if variant in ('ascad_head_multi', 'timesnet_ascad_head_multi'):
        return TimesNetASCADHeadMulti(config)
    raise ValueError(f"Unknown TimesNet variant: {variant}")
