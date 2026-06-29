import os
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class AscadPaths:
    profiling: str
    attack: str


class AscadTraceDataset(Dataset):
    def __init__(self, h5_path, split='profiling', target_byte=2, normalize=False, dtype=torch.float32):
        self.h5_path = os.path.abspath(h5_path)
        self.split = split
        self.target_byte = target_byte
        self.normalize = normalize
        self.dtype = dtype

        if split not in {'profiling', 'attack'}:
            raise ValueError("split must be 'profiling' or 'attack'")

        with h5py.File(self.h5_path, 'r') as handle:
            traces_key = 'Profiling_traces/traces' if split == 'profiling' else 'Attack_traces/traces'
            labels_key = 'Profiling_traces/labels' if split == 'profiling' else 'Attack_traces/labels'
            metadata_key = 'Profiling_traces/metadata' if split == 'profiling' else 'Attack_traces/metadata'

            self.length = handle[traces_key].shape[0]
            self.seq_len = handle[traces_key].shape[1]
            self.has_labels = labels_key in handle
            self.has_metadata = metadata_key in handle

    def __len__(self):
        return self.length

    def _read_trace(self, handle, index, split):
        traces_key = 'Profiling_traces/traces' if split == 'profiling' else 'Attack_traces/traces'
        trace = handle[traces_key][index]
        trace = np.asarray(trace, dtype=np.float32)
        if self.normalize:
            mean = trace.mean()
            std = trace.std()
            trace = (trace - mean) / (std + 1e-6)
        trace = torch.as_tensor(trace, dtype=self.dtype).unsqueeze(-1)
        return trace

    def _read_label(self, handle, index, split):
        labels_key = 'Profiling_traces/labels' if split == 'profiling' else 'Attack_traces/labels'
        if labels_key not in handle:
            return None
        label = handle[labels_key][index]
        return torch.as_tensor(int(label), dtype=torch.long)

    def __getitem__(self, index):
        with h5py.File(self.h5_path, 'r') as handle:
            trace = self._read_trace(handle, index, self.split)
            label = self._read_label(handle, index, self.split)
        if label is None:
            return trace
        return trace, label


def ascad_collate_fn(batch):
    traces, labels = zip(*batch)
    x = torch.stack(traces, dim=0)
    y = torch.stack(labels, dim=0)
    padding_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool)
    return x, y, padding_mask


def build_fixed_length_ascad_loaders(profiling_path, attack_path=None, batch_size=64, normalize=False, num_workers=0):
    profiling_dataset = AscadTraceDataset(profiling_path, split='profiling', normalize=normalize)
    profiling_loader = torch.utils.data.DataLoader(
        profiling_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ascad_collate_fn,
        drop_last=False,
    )

    attack_loader = None
    attack_dataset = None
    if attack_path is not None:
        attack_dataset = AscadTraceDataset(attack_path, split='attack', normalize=normalize)
        attack_loader = torch.utils.data.DataLoader(
            attack_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=ascad_collate_fn,
            drop_last=False,
        )

    return profiling_dataset, profiling_loader, attack_dataset, attack_loader
