import torch
import torch.nn as nn


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, kernel_sizes=None, init_weight=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        if kernel_sizes is None:
            kernel_sizes = [2 * i + 1 for i in range(num_kernels)]
        self.kernel_sizes = kernel_sizes
        kernels = []
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        outputs = [kernel(x) for kernel in self.kernels]
        return torch.stack(outputs, dim=-1).mean(-1)
