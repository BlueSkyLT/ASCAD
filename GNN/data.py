import numpy as np
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


def make_graph_loader(X, y, batch_size, shuffle):
    if X.ndim != 2:
        raise ValueError('graph mode expects traces with shape [n_samples, n_nodes]')
    num_nodes = X.shape[1]
    if num_nodes != 700:
        raise ValueError('graph mode baseline expects 700 nodes per trace, got %d' % num_nodes)

    edge_index = build_fully_connected_edge_index(num_nodes)
    ds = TraceGraphDataset(X, y)

    def _collate(batch):
        traces = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch])
        graph_batch = build_graph_batch_from_traces(traces, edge_index)
        return graph_batch, labels

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate), None
