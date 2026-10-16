import numpy as np
import os
import torch
from torch.utils.data import DataLoader

from .utils import HAS_PYG, ensure_graph_dependencies

if HAS_PYG:
    from torch_geometric.data import Batch, Data
else:
    Batch = None
    Data = None


def build_fully_connected_edge_index(num_nodes):
    src = np.repeat(np.arange(num_nodes, dtype=np.int64), num_nodes)
    dst = np.tile(np.arange(num_nodes, dtype=np.int64), num_nodes)
    edge_index = np.stack([src, dst], axis=0)
    return torch.as_tensor(edge_index, dtype=torch.long)


class TraceGraphDataset(torch.utils.data.Dataset):
    def __init__(self, traces, labels):
        self.traces = traces
        self.labels = labels

    def __len__(self):
        return self.traces.shape[0]

    def __getitem__(self, idx):
        return self.traces[idx], self.labels[idx]


def build_graph_batch_from_traces(trace_batch, edge_index):
    ensure_graph_dependencies('gcn')
    if edge_index is None:
        if len(trace_batch) == 0:
            raise ValueError('empty trace batch provided for graph conversion')
        edge_index = build_fully_connected_edge_index(trace_batch[0].numel())

    data_list = []
    for trace in trace_batch:
        x = trace.reshape(-1, 1)
        data_list.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(data_list)


class _GPUResidentGraphLoader:
    """Zero-copy GPU-resident loader for graph models.

    Because the graph topology is identical for every trace (fixed complete graph
    on 700 nodes), we can keep the entire feature matrix on the GPU. This
    eliminates DataLoader, worker processes, collate, and PCIe transfers on every
    batch/epoch. Slicing on GPU is extremely fast and lets us drive much higher
    sustained utilization with large batches (directly addressing "high GPU util
    but low memory usage").
    """
    def __init__(self, X, y, batch_size, shuffle, device):
        self.X = torch.as_tensor(X, dtype=torch.float32, device=device)
        self.y = torch.as_tensor(y, dtype=torch.long, device=device)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.n = self.X.shape[0]
        self.device = device

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.n, device=self.device)
            X = self.X[perm]
            y = self.y[perm]
        else:
            X = self.X
            y = self.y
        for start in range(0, self.n, self.batch_size):
            end = min(start + self.batch_size, self.n)
            yield X[start:end], y[start:end]

    def __len__(self):
        import math
        return math.ceil(self.n / self.batch_size)


def make_graph_loader(X, y, batch_size, shuffle, device=None, mask=None):
    """Create a loader for graph traces.

    Sample-as-node baseline: X shape (n, 700).
    Segment-as-node: X shape (n, N, F) with optional mask (n, N).

    If device is a CUDA device we preload the full profiling (or attack) data
    onto the GPU once and serve pure device-to-device slices. This is the
    highest throughput path when memory usage reports as "low".
    """
    X = np.asarray(X) if not isinstance(X, torch.Tensor) else X
    if isinstance(X, torch.Tensor):
        x_ndim = X.dim()
        n_nodes = X.shape[1]
    else:
        x_ndim = X.ndim
        n_nodes = X.shape[1]

    if x_ndim == 2:
        if n_nodes != 700:
            raise ValueError('sample-graph mode expects 700 nodes per trace, got %d' % n_nodes)
        if device is not None and getattr(device, 'type', None) == 'cuda':
            return _GPUResidentGraphLoader(X, y, batch_size, shuffle, device), None
    elif x_ndim == 3:
        # Segment features (n, N, F) — optionally with mask
        if device is not None and getattr(device, 'type', None) == 'cuda':
            return _GPUResidentSegmentLoader(X, y, batch_size, shuffle, device, mask=mask), None
    else:
        raise ValueError('graph mode expects X with shape (n, 700) or (n, N, F), got ndim=%s' % x_ndim)

    # Fallback host DataLoader path (pin_memory + workers)
    ds = TraceGraphDataset(X, y) if mask is None else SegmentGraphDataset(X, y, mask)

    def _collate(batch):
        if mask is None:
            traces = [item[0] for item in batch]
            if len(traces) > 0 and isinstance(traces[0], np.ndarray):
                traces_t = torch.from_numpy(np.stack(traces)).float()
            else:
                traces_t = torch.stack([
                    t.float() if isinstance(t, torch.Tensor) else torch.from_numpy(np.asarray(t)).float()
                    for t in traces
                ])
            labels = torch.stack([
                item[1] if isinstance(item[1], torch.Tensor) else torch.as_tensor(item[1])
                for item in batch
            ])
            return traces_t, labels
        xs, ys, ms = [], [], []
        for item in batch:
            x_i, y_i, m_i = item
            xs.append(x_i if isinstance(x_i, torch.Tensor) else torch.from_numpy(np.asarray(x_i)))
            ys.append(y_i if isinstance(y_i, torch.Tensor) else torch.as_tensor(y_i))
            ms.append(m_i if isinstance(m_i, torch.Tensor) else torch.from_numpy(np.asarray(m_i)))
        return (torch.stack(xs).float(), torch.stack(ms).bool()), torch.stack(ys)

    num_workers = min(8, os.cpu_count() or 1)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": _collate,
    }
    if num_workers > 0:
        loader_kwargs.update({
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        })
    return DataLoader(ds, **loader_kwargs), None


class SegmentGraphDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, mask):
        self.features = features
        self.labels = labels
        self.mask = mask

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.mask[idx]


class _GPUResidentSegmentLoader:
    """GPU-resident loader for segment-node features (B, N, F) + mask."""

    def __init__(self, X, y, batch_size, shuffle, device, mask=None):
        self.X = torch.as_tensor(X, dtype=torch.float32, device=device)
        self.y = torch.as_tensor(y, dtype=torch.long, device=device)
        if mask is None:
            self.mask = torch.ones(self.X.shape[0], self.X.shape[1], dtype=torch.bool, device=device)
        else:
            self.mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.n = self.X.shape[0]
        self.device = device

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.n, device=self.device)
            X = self.X[perm]
            y = self.y[perm]
            m = self.mask[perm]
        else:
            X, y, m = self.X, self.y, self.mask
        for start in range(0, self.n, self.batch_size):
            end = min(self.n, start + self.batch_size)
            yield (X[start:end], m[start:end]), y[start:end]

    def __len__(self):
        import math
        return math.ceil(self.n / self.batch_size)
