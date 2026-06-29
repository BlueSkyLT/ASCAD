from dataclasses import dataclass


@dataclass
class TimesNetConfig:
    seq_len: int = 700
    enc_in: int = 1
    num_class: int = 256
    d_model: int = 32
    d_ff: int = 64
    e_layers: int = 2
    top_k: int = 5
    num_kernels: int = 6
    dropout: float = 0.1
    embedding_kernel_size: int = 3
    use_padding_mask: bool = True
    head_hidden_dim: int | None = None
    head_hidden_dim_2: int | None = None
    head_dropout: float = 0.2
