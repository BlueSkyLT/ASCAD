"""MultiResNet Model Implementation for Side-Channel Analysis on ASCADv2 (STM32F303).

Reference:
- ANSSI ASCADv2 paper: "ASCADv2: a benchmark for deep learning-based side-channel analysis" (Masure & Strullu, 2021)
- secAES-STM32 reference multi-head ResNet architecture.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn


class ResNetLayer(nn.Module):
    """1D Convolutional layer with optional BatchNorm and activation."""

    def __init__(
        self,
        in_filters: int,
        out_filters: int = 16,
        kernel_size: int = 11,
        strides: int = 1,
        activation: bool = True,
        batch_normalization: bool = True,
        conv_first: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_filters,
            out_filters,
            kernel_size=kernel_size,
            stride=strides,
            padding="same" if strides == 1 else kernel_size // 2,
            bias=True,
        )
        self.bn = nn.BatchNorm1d(out_filters) if batch_normalization else None
        self.act = nn.ReLU() if activation else None
        self.conv_first = conv_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv_first:
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            if self.act is not None:
                x = self.act(x)
            return x

        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return self.conv(x)


class ResidualBlock(nn.Module):
    """Residual unit with two ResNet layers and an optional projection shortcut."""

    def __init__(self, in_filters: int, out_filters: int, strides: int = 1):
        super().__init__()
        self.y1 = ResNetLayer(in_filters, out_filters, strides=strides)
        self.y2 = ResNetLayer(out_filters, out_filters, activation=False)
        self.shortcut = None
        if strides != 1 or in_filters != out_filters:
            self.shortcut = ResNetLayer(
                in_filters,
                out_filters,
                kernel_size=1,
                strides=strides,
                activation=False,
                batch_normalization=False,
            )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.y1(x)
        y = self.y2(y)
        identity = x if self.shortcut is None else self.shortcut(x)
        return self.relu(identity + y)


class BranchHead(nn.Module):
    """Dense classification head for each multi-task target (FC 1024 + BN + FC out)."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        return self.out(x)


class MultiResNet(nn.Module):
    """Multi-task ResNet for ASCADv2 (STM32F303).

    Predicts:
      - alpha_output: multiplicative mask alpha (256 classes)
      - beta_output: additive mask beta (256 classes)
      - sbox_{0..15}_output: 16 masked S-box outputs (256 classes each)
      - permind_{0..15}_output (optional): 16 shuffle indices (16 classes each)
    """

    def __init__(
        self,
        input_dim: int = 15000,
        depth: int = 19,
        without_permind: bool = False,
    ):
        super().__init__()
        if (depth - 1) % 18 != 0:
            raise ValueError(f"depth should be 18n+1 (e.g. 19, 37, 55...), got {depth}")

        num_filters = 16
        num_res_blocks = int((depth - 1) / 18)

        self.stem = ResNetLayer(1, 16)
        stacks = []
        in_filters = 16
        for stack_idx in range(9):
            block_list = []
            for res_block in range(num_res_blocks):
                strides = 1
                if stack_idx > 0 and res_block == 0:
                    strides = 2
                block = ResidualBlock(in_filters, num_filters, strides=strides)
                block_list.append(block)
                in_filters = num_filters
            stacks.append(nn.Sequential(*block_list))
            if num_filters < 256:
                num_filters *= 2
        self.stacks = nn.ModuleList(stacks)

        self.pool = nn.AvgPool1d(kernel_size=4)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            x = self.stem(dummy)
            for st in self.stacks:
                x = st(x)
            x = self.pool(x)
            flat_dim = x.view(1, -1).shape[1]

        self.flat_dim = flat_dim
        self.alpha = BranchHead(flat_dim, 256)
        self.beta = BranchHead(flat_dim, 256)
        self.sbox = nn.ModuleList([BranchHead(flat_dim, 256) for _ in range(16)])
        self.without_permind = without_permind
        if not without_permind:
            self.permind = nn.ModuleList([BranchHead(flat_dim, 16) for _ in range(16)])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[1] != 1 and x.shape[2] == 1:
            x = x.permute(0, 2, 1)

        x = self.stem(x)
        for st in self.stacks:
            x = st(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        outputs = {
            "alpha_output": self.alpha(x),
            "beta_output": self.beta(x),
        }
        for i in range(16):
            outputs[f"sbox_{i}_output"] = self.sbox[i](x)
        if not self.without_permind:
            for i in range(16):
                outputs[f"permind_{i}_output"] = self.permind[i](x)
        return outputs


class CompactMultiResNet(nn.Module):
    """Compact Multi-task ResNet for ASCADv2 (STM32F303).

    Compresses MultiResNet from ~137.4M down to ~15.09M params (~89% reduction)
    by placing a shared bottleneck layer (default 512-dim) between the 1D ResNet
    trunk and the 34 classification heads, and slimming the heads from
    hidden_dim=1024 down to hidden_dim=256.

    Retains the exact 7.53M 1D convolutional trunk and all multi-task targets:
      - alpha_output: multiplicative mask alpha (256 classes)
      - beta_output: additive mask beta (256 classes)
      - sbox_{0..15}_output: 16 masked S-box outputs (256 classes each)
      - permind_{0..15}_output (optional): 16 shuffle indices (16 classes each)
    """

    def __init__(
        self,
        input_dim: int = 15000,
        depth: int = 19,
        without_permind: bool = False,
        bottleneck_dim: Optional[int] = 512,
        head_hidden_dim: int = 256,
    ):
        super().__init__()
        if (depth - 1) % 18 != 0:
            raise ValueError(f"depth should be 18n+1 (e.g. 19, 37, 55...), got {depth}")

        num_filters = 16
        num_res_blocks = int((depth - 1) / 18)

        self.stem = ResNetLayer(1, 16)
        stacks = []
        in_filters = 16
        for stack_idx in range(9):
            block_list = []
            for res_block in range(num_res_blocks):
                strides = 1
                if stack_idx > 0 and res_block == 0:
                    strides = 2
                block = ResidualBlock(in_filters, num_filters, strides=strides)
                block_list.append(block)
                in_filters = num_filters
            stacks.append(nn.Sequential(*block_list))
            if num_filters < 256:
                num_filters *= 2
        self.stacks = nn.ModuleList(stacks)

        self.pool = nn.AvgPool1d(kernel_size=4)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            x = self.stem(dummy)
            for st in self.stacks:
                x = st(x)
            x = self.pool(x)
            flat_dim = x.view(1, -1).shape[1]

        self.flat_dim = flat_dim
        use_bottleneck = (bottleneck_dim is not None) and (bottleneck_dim > 0)
        self.bottleneck_dim = bottleneck_dim if use_bottleneck else None
        if use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(flat_dim, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(),
            )
            head_in_dim = bottleneck_dim
        else:
            self.bottleneck = None
            head_in_dim = flat_dim

        self.alpha = BranchHead(head_in_dim, 256, hidden_dim=head_hidden_dim)
        self.beta = BranchHead(head_in_dim, 256, hidden_dim=head_hidden_dim)
        self.sbox = nn.ModuleList([BranchHead(head_in_dim, 256, hidden_dim=head_hidden_dim) for _ in range(16)])
        self.without_permind = without_permind
        if not without_permind:
            self.permind = nn.ModuleList([BranchHead(head_in_dim, 16, hidden_dim=head_hidden_dim) for _ in range(16)])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[1] != 1 and x.shape[2] == 1:
            x = x.permute(0, 2, 1)

        x = self.stem(x)
        for st in self.stacks:
            x = st(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        if self.bottleneck is not None:
            x = self.bottleneck(x)

        # T-039 E6b: heads in float32 under AMP (set by train.py when a head has no CE label).
        # Unlabelled heads trained only through the decoder overflowed fp16 logits at 300k.
        if getattr(self, "heads_fp32", False):
            with torch.autocast(device_type=x.device.type, enabled=False):
                return self._heads(x.float())
        return self._heads(x)

    def _heads(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {
            "alpha_output": self.alpha(x),
            "beta_output": self.beta(x),
        }
        for i in range(16):
            outputs[f"sbox_{i}_output"] = self.sbox[i](x)
        if not self.without_permind:
            for i in range(16):
                outputs[f"permind_{i}_output"] = self.permind[i](x)
        return outputs


class SingleHeadResNet(nn.Module):
    """Single-head Depth-19 ResNet for direct unmasked S-box prediction on ASCADv2 (Arm 2A).

    Pure 1D Conv ResNet trunk (7.53M) + 1 single classification head (~0.25M with hidden_dim=64,
    or ~3.9M with hidden_dim=1024). Eliminates 70M dead parameters from unused multi-task heads.
    """

    def __init__(
        self,
        input_dim: int = 15000,
        depth: int = 19,
        num_classes: int = 256,
        head_hidden_dim: int = 64,
    ):
        super().__init__()
        if (depth - 1) % 18 != 0:
            raise ValueError(f"depth should be 18n+1 (e.g. 19, 37, 55...), got {depth}")

        num_filters = 16
        num_res_blocks = int((depth - 1) / 18)

        self.stem = ResNetLayer(1, 16)
        stacks = []
        in_filters = 16
        for stack_idx in range(9):
            block_list = []
            for res_block in range(num_res_blocks):
                strides = 1
                if stack_idx > 0 and res_block == 0:
                    strides = 2
                block = ResidualBlock(in_filters, num_filters, strides=strides)
                block_list.append(block)
                in_filters = num_filters
            stacks.append(nn.Sequential(*block_list))
            if num_filters < 256:
                num_filters *= 2
        self.stacks = nn.ModuleList(stacks)
        self.pool = nn.AvgPool1d(kernel_size=4)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            x = self.stem(dummy)
            for st in self.stacks:
                x = st(x)
            x = self.pool(x)
            flat_dim = x.view(1, -1).shape[1]

        self.flat_dim = flat_dim
        self.head = BranchHead(flat_dim, num_classes, hidden_dim=head_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[1] != 1 and x.shape[2] == 1:
            x = x.permute(0, 2, 1)

        x = self.stem(x)
        for st in self.stacks:
            x = st(x)
        x = self.pool(x)
        feat = torch.flatten(x, 1)
        return self.head(feat)


class SlotUnshuffledResNet(nn.Module):
    """16-Head Slot-Unshuffled ResNet for non-worst-case (black-box) attack on ASCADv2 (Arm 2B).

    Neutralizes temporal shuffling by assigning 16 dedicated unshared slim heads to the
    16 physical execution slots in the trace. Each head s predicts the unmasked S-box value
    Y_{i, s} = AES_Sbox[p_{i, pi(i, s)} ^ k_{i, pi(i, s)}] computed at physical slot s.

    Contains NO mask labels (alpha, beta omitted from loss and architecture).
    Total parameters: ~11.46M.
    """

    def __init__(
        self,
        input_dim: int = 15000,
        depth: int = 19,
        head_hidden_dim: int = 64,
        num_classes: int = 256,
    ):
        super().__init__()
        if (depth - 1) % 18 != 0:
            raise ValueError(f"depth should be 18n+1 (e.g. 19, 37, 55...), got {depth}")

        num_filters = 16
        num_res_blocks = int((depth - 1) / 18)

        self.stem = ResNetLayer(1, 16)
        stacks = []
        in_filters = 16
        for stack_idx in range(9):
            block_list = []
            for res_block in range(num_res_blocks):
                strides = 1
                if stack_idx > 0 and res_block == 0:
                    strides = 2
                block = ResidualBlock(in_filters, num_filters, strides=strides)
                block_list.append(block)
                in_filters = num_filters
            stacks.append(nn.Sequential(*block_list))
            if num_filters < 256:
                num_filters *= 2
        self.stacks = nn.ModuleList(stacks)
        self.pool = nn.AvgPool1d(kernel_size=4)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            x = self.stem(dummy)
            for st in self.stacks:
                x = st(x)
            x = self.pool(x)
            flat_dim = x.view(1, -1).shape[1]

        self.flat_dim = flat_dim
        self.slot_heads = nn.ModuleList([
            BranchHead(flat_dim, num_classes, hidden_dim=head_hidden_dim)
            for _ in range(16)
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[1] != 1 and x.shape[2] == 1:
            x = x.permute(0, 2, 1)

        x = self.stem(x)
        for st in self.stacks:
            x = st(x)
        x = self.pool(x)
        feat = torch.flatten(x, 1)

        return {f"slot_{i}_output": self.slot_heads[i](feat) for i in range(16)}

