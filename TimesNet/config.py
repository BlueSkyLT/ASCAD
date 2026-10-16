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
    # Multilabel (ASCADv2): pool over time then branch heads.
    without_permind: bool = False
    # Simple linear heads on pooled features (True) vs MultiResNet-style BranchHead (False).
    simple_heads: bool = True
    # Subsample input along time before encoder (L_eff = ceil(L / stride)).
    input_stride: int = 1
    # 0 = global pool over time; K>0 = adaptive pool to K segments then project.
    pool_segments: int = 0
    # "avg" (default, checkpoint-compatible) or "max". Not a learned parameter.
    pool_mode: str = "avg"
