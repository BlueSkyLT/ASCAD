import json
import math
import os
import os.path
import sys
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from torch.amp import autocast, GradScaler
    from contextlib import nullcontext
    HAS_AMP = True
except Exception:
    HAS_AMP = False
    autocast = None
    GradScaler = None
    nullcontext = None


class _NoopContext:
    def __enter__(self):
        return None
    def __exit__(self, *args):
        return False

from GNN.classifiers import (
    GATClassifier,
    GCNClassifier,
    GINClassifier,
    GraphSAGEClassifier,
    SegmentGraphSAGEClassifier,
)
from GNN.data import make_graph_loader
from GNN.utils import (
    ensure_graph_dependencies,
    is_graph_network,
    is_segment_graph_network,
)
from GNN.segment_graph import (
    DEFAULT_EDGE_MODE,
    DEFAULT_NUM_NODES,
    DEFAULT_OUT_DIM,
    DEFAULT_SEGMENTER,
    DEFAULT_THRESHOLD,
    cache_path_for,
    load_or_build_segment_features,
    standardize_segment_features,
)
from utils.snr import discover_snr_windows
from TimesNet import build_timesnet_model
from leakage_localizer.model import build_poi_searcher, is_poi_searcher_network


# Keep output ordering stable for multilabel mode.
MULTILABEL_KEYS = [
    'alpha_output',
    'beta_output',
] + [f'sbox_{i}_output' for i in range(16)] + [f'permind_{i}_output' for i in range(16)]

MULTILABEL_WITHOUT_PERMIND_KEYS = [
    'alpha_output',
    'beta_output',
] + [f'sbox_{i}_output' for i in range(16)]


class MLPBest(nn.Module):
    def __init__(self, node=200, layer_nb=6, input_dim=1400):
        super().__init__()
        layers = [nn.Linear(input_dim, node), nn.ReLU()]
        for _ in range(layer_nb - 2):
            layers.extend([nn.Linear(node, node), nn.ReLU()])
        layers.append(nn.Linear(node, 256))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNNBest(nn.Module):
    def __init__(self, classes=256, input_dim=700, first_stride=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=first_stride, padding='same'),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(64, 128, kernel_size=11, padding='same'),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(128, 256, kernel_size=11, padding='same'),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(256, 512, kernel_size=11, padding='same'),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(512, 512, kernel_size=11, padding='same'),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            flat_dim = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResNetLayer(nn.Module):
    def __init__(self, in_filters, out_filters=16, kernel_size=11, strides=1,
                 activation=True, batch_normalization=True, conv_first=True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_filters,
            out_filters,
            kernel_size=kernel_size,
            stride=strides,
            padding='same' if strides == 1 else kernel_size // 2,
            bias=True,
        )
        self.bn = nn.BatchNorm1d(out_filters) if batch_normalization else None
        self.act = nn.ReLU() if activation else None
        self.conv_first = conv_first

    def forward(self, x):
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
    def __init__(self, in_filters, out_filters, strides=1):
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

    def forward(self, x):
        y = self.y1(x)
        y = self.y2(y)
        identity = x if self.shortcut is None else self.shortcut(x)
        return self.relu(identity + y)


class BranchHead(nn.Module):
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


class MultiResNet(nn.Module):
    def __init__(self, input_dim=15000, depth=19, without_permind=False):
        super().__init__()
        if (depth - 1) % 18 != 0:
            raise ValueError('depth should be 18n+1 (eg 19, 37, 55 ...)')

        num_filters = 16
        num_res_blocks = int((depth - 1) / 18)

        self.stem = ResNetLayer(1, 16)
        stacks = []
        in_filters = 16
        for _stack in range(9):
            block_list = []
            for res_block in range(num_res_blocks):
                strides = 1
                if _stack > 0 and res_block == 0:
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

        self.alpha = BranchHead(flat_dim, 256)
        self.beta = BranchHead(flat_dim, 256)
        self.sbox = nn.ModuleList([BranchHead(flat_dim, 256) for _ in range(16)])
        self.without_permind = without_permind
        if not without_permind:
            self.permind = nn.ModuleList([BranchHead(flat_dim, 16) for _ in range(16)])

    def forward(self, x):
        x = self.stem(x)
        for st in self.stacks:
            x = st(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        outputs = {
            'alpha_output': self.alpha(x),
            'beta_output': self.beta(x),
        }
        for i in range(16):
            outputs[f'sbox_{i}_output'] = self.sbox[i](x)
        if not self.without_permind:
            for i in range(16):
                outputs[f'permind_{i}_output'] = self.permind[i](x)
        return outputs


class MultiTest(nn.Module):
    def __init__(self, input_dim=1400):
        super().__init__()
        self.conv = nn.Conv1d(1, 3, kernel_size=11, stride=100, padding='same')
        self.relu = nn.ReLU()

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            flat_dim = self.relu(self.conv(dummy)).view(1, -1).shape[1]

        self.alpha = BranchHead(flat_dim, 256)
        self.beta = BranchHead(flat_dim, 256)
        self.sbox = nn.ModuleList([BranchHead(flat_dim, 256) for _ in range(16)])
        self.permind = nn.ModuleList([BranchHead(flat_dim, 16) for _ in range(16)])

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = torch.flatten(x, 1)
        outputs = {
            'alpha_output': self.alpha(x),
            'beta_output': self.beta(x),
        }
        for i in range(16):
            outputs[f'sbox_{i}_output'] = self.sbox[i](x)
        for i in range(16):
            outputs[f'permind_{i}_output'] = self.permind[i](x)
        return outputs


def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path) is False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


def strip_torch_compile_prefix(state_dict):
    """Remove torch.compile `_orig_mod.` prefix from checkpoint keys if present."""
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if keys and all(k.startswith("_orig_mod.") for k in keys):
        return {k[len("_orig_mod.") :]: v for k, v in state_dict.items()}
    if keys and any(k.startswith("_orig_mod.") for k in keys):
        return {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}
    return state_dict


def load_sca_model(model_file, network_type, input_dim, device, timesnet_config=None):
    check_file_exists(model_file)
    ensure_graph_dependencies(network_type)
    if is_timesnet_network(network_type):
        model = build_timesnet_for_network(network_type, input_dim, timesnet_config=timesnet_config)
    elif network_type == 'mlp':
        model = MLPBest(input_dim=input_dim)
    elif network_type == 'cnn':
        model = CNNBest(input_dim=input_dim)
    elif network_type == 'cnn2':
        model = CNNBest(input_dim=input_dim, first_stride=2)
    elif network_type == 'multi_test':
        model = MultiTest(input_dim=input_dim)
    elif network_type == 'multi_resnet':
        model = MultiResNet(input_dim=input_dim, depth=19, without_permind=False)
    elif network_type == 'multi_resnet_without_permind':
        model = MultiResNet(input_dim=input_dim, depth=19, without_permind=True)
    elif network_type == 'gcn':
        model = GCNClassifier(input_dim=1)
    elif network_type == 'gat':
        model = GATClassifier(input_dim=1)
    elif network_type == 'gin':
        model = GINClassifier(input_dim=1)
    elif network_type == 'graphsage':
        model = GraphSAGEClassifier(input_dim=1)
    elif network_type == 'seg_graphsage':
        # Prefer dimensions from checkpoint sidecar when available.
        model = SegmentGraphSAGEClassifier(input_dim=input_dim if input_dim and input_dim > 1 else DEFAULT_OUT_DIM)
    elif is_poi_searcher_network(network_type):
        model = build_poi_searcher(input_dim, None)
    else:
        print("Error: no topology found for network '%s' ..." % network_type)
        sys.exit(-1)

    try:
        state = torch.load(model_file, map_location=device)
        state = strip_torch_compile_prefix(state)
        model.load_state_dict(state)
    except Exception:
        print("Error: can't load PyTorch model file '%s'" % model_file)
        sys.exit(-1)

    model.to(device)
    model.eval()
    return model


# Loads the profiling and attack datasets from the ASCAD database.
def load_ascad(ascad_database_file, load_metadata=False, train_len=0, load_attack=True):
    """Load ASCAD HDF5 splits.

    train_len>0 slices profiling traces/labels inside HDF5 (int8) so large v2
    DBs do not need a full float32 materialization of 500k traces.
    """
    check_file_exists(ascad_database_file)
    try:
        in_file = h5py.File(ascad_database_file, "r")
    except Exception:
        print(
            "Error: can't open HDF5 file '%s' for reading (it might be malformed) ..."
            % ascad_database_file
        )
        sys.exit(-1)

    n_prof = in_file['Profiling_traces/traces'].shape[0]
    if train_len and train_len > 0:
        n_take = min(int(train_len), n_prof)
        X_profiling = np.array(in_file['Profiling_traces/traces'][:n_take], dtype=np.int8)
        Y_profiling = np.array(in_file['Profiling_traces/labels'][:n_take])
    else:
        X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
        Y_profiling = np.array(in_file['Profiling_traces/labels'])

    if load_attack:
        X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
        Y_attack = np.array(in_file['Attack_traces/labels'])
    else:
        X_attack, Y_attack = None, None

    if load_metadata is False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)

    meta_p = in_file['Profiling_traces/metadata']
    meta_a = in_file['Attack_traces/metadata'] if load_attack else None
    if train_len and train_len > 0:
        meta_p = meta_p[: min(int(train_len), n_prof)]
    return (
        (X_profiling, Y_profiling),
        (X_attack, Y_attack),
        (meta_p, meta_a),
    )


def build_multilabel_targets(Y, without_permind=False):
    y = {
        'alpha_output': torch.as_tensor(Y['alpha_mask'], dtype=torch.long).reshape(-1),
        'beta_output': torch.as_tensor(Y['beta_mask'], dtype=torch.long).reshape(-1),
    }
    sbox_field = 'sbox_masked_with_perm' if without_permind else 'sbox_masked'
    for i in range(16):
        y[f'sbox_{i}_output'] = torch.as_tensor(Y[sbox_field][:, i], dtype=torch.long)
    if not without_permind:
        for i in range(16):
            y[f'permind_{i}_output'] = torch.as_tensor(Y['perm_index'][:, i], dtype=torch.long)
    return y


def is_timesnet_network(network_type):
    return network_type in (
        'timesnet',
        'timesnet_ascad_head',
        'timesnet_multi',
        'timesnet_ascad_head_multi',
    )


def is_timesnet_multilabel_network(network_type):
    return network_type in ('timesnet_multi', 'timesnet_ascad_head_multi')


def _effective_timesnet_seq_len(raw_len, timesnet_config):
    """seq_len after optional input_stride (matches x[:, ::stride])."""
    cfg = timesnet_config or {}
    stride = max(1, int(cfg.get('input_stride', 1) or 1))
    if stride <= 1:
        return int(raw_len)
    return (int(raw_len) + stride - 1) // stride


def build_timesnet_for_network(network_type, input_dim, timesnet_config=None):
    """Factory for single-label and multilabel TimesNet variants."""
    cfg = dict(timesnet_config) if timesnet_config else {}
    raw_len = int(cfg.get('seq_len', input_dim) or input_dim)
    cfg['seq_len'] = _effective_timesnet_seq_len(raw_len, cfg)
    if network_type == 'timesnet':
        return build_timesnet_model(variant='baseline', **cfg)
    if network_type == 'timesnet_ascad_head':
        cfg.setdefault('embedding_kernel_size', 11)
        return build_timesnet_model(variant='ascad_head', **cfg)
    if network_type == 'timesnet_multi':
        cfg.setdefault('embedding_kernel_size', 3)
        cfg.setdefault('simple_heads', True)
        return build_timesnet_model(variant='multi', **cfg)
    if network_type == 'timesnet_ascad_head_multi':
        cfg.setdefault('embedding_kernel_size', 11)
        cfg.setdefault('simple_heads', True)
        return build_timesnet_model(variant='ascad_head_multi', **cfg)
    raise ValueError(f"Not a TimesNet network_type: {network_type}")


def save_checkpoint_sidecar(save_file_name, payload):
    """Write {ckpt}.config.json next to a checkpoint for reproducible eval."""
    path = save_file_name + ".config.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        print(f"Warning: could not write sidecar {path}: {exc}")


def reshape_inputs_for_network(X, network_type):
    x = torch.as_tensor(X, dtype=torch.float32)
    if network_type in ('mlp',) or is_poi_searcher_network(network_type):
        return x
    if is_graph_network(network_type):
        return x
    if is_timesnet_network(network_type):
        return x.unsqueeze(-1)  # (B, L) -> (B, L, 1) for TimesNet
    return x.unsqueeze(1)  # (B, L) -> (B, 1, L) for CNN


def split_train_validation(X, y, validation_split):
    n_samples = X.shape[0]
    val_count = int(n_samples * validation_split)
    if val_count == 0:
        return (X, y), (None, None)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    def _select_tensor_or_dict(data, idx):
        if isinstance(data, dict):
            return {k: v[idx] for k, v in data.items()}
        return data[idx]

    X_train = X[train_indices]
    y_train = _select_tensor_or_dict(y, train_indices)
    X_val = X[val_indices]
    y_val = _select_tensor_or_dict(y, val_indices)
    return (X_train, y_train), (X_val, y_val)


def make_loader(X, y, batch_size, shuffle, network_type, device=None, graph_mask=None):
    if is_graph_network(network_type):
        # Pass device so we can use the zero-overhead GPU-resident loader
        # when on CUDA (keeps the fixed-topology graph features on device).
        return make_graph_loader(
            X, y, batch_size, shuffle, device=device, mask=graph_mask
        )

    if isinstance(y, dict):
        keys = list(y.keys())
        tensors = [X] + [y[k] for k in keys]
        ds = TensorDataset(*tensors)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle), keys

    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle), None


def head_loss_weight(key, c_weight):
    """Loss weight for one head: the 16 masked S-box (C) terms scale, everything else stays at 1."""
    return c_weight if key.startswith("sbox_") else 1.0


def compute_multilabel_loss(outputs, target_dict, criterion, keys, c_weight=1.0):
    """Weighted mean over heads. At c_weight=1 this is exactly the plain mean the trainer used
    before. The denominator is the SUM OF WEIGHTS rather than the head count, so down-weighting C
    does not silently rescale the reported loss — which matters because val loss is what selects
    the `best` checkpoint (T-031 N4b showed checkpoint choice is load-bearing)."""
    total_loss = 0.0
    total_weight = 0.0
    for k in keys:
        if k not in outputs or k not in target_dict:
            continue
        w = head_loss_weight(k, c_weight)
        total_loss = total_loss + w * criterion(outputs[k], target_dict[k])
        total_weight += w
    if total_weight == 0.0:
        raise RuntimeError("compute_multilabel_loss: no valid head keys")
    return total_loss / total_weight


def resolve_multilabel_keys(multilabel, train_heads=None):
    if multilabel == 1:
        keys = list(MULTILABEL_KEYS)
    else:
        keys = list(MULTILABEL_WITHOUT_PERMIND_KEYS)
    if train_heads:
        allow = set(train_heads)
        keys = [k for k in keys if k in allow]
        if not keys:
            raise ValueError(f"train_heads={train_heads} matched no multilabel keys")
    return keys


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    multilabel=0,
    use_amp=False,
    scaler=None,
    grad_clip=0.0,
    train_heads=None,
    return_head_acc=False,
    c_weight=1.0,
):
    """One pass over `loader`. Returns the mean loss, or (loss, {head: top-1}) when
    `return_head_acc` is set on a multilabel run (T-030 S3 needs the per-epoch alpha curve)."""
    training = optimizer is not None
    criterion = nn.CrossEntropyLoss()
    epoch_loss = 0.0
    sample_count = 0
    skipped_nonfinite = 0
    head_correct = {}
    head_total = 0

    if training:
        model.train()
    else:
        model.eval()

    if use_amp and HAS_AMP and device.type == 'cuda':
        amp_ctx = autocast('cuda')
    else:
        amp_ctx = nullcontext() if nullcontext is not None else _NoopContext()

    multilabel_keys = None
    if multilabel != 0:
        multilabel_keys = resolve_multilabel_keys(multilabel, train_heads=train_heads)

    for batch in loader:
        if multilabel == 0:
            xb, yb = batch
            # Segment-graph loaders yield ((features, mask), labels)
            if isinstance(xb, (tuple, list)):
                x_feat, x_mask = xb[0], xb[1]
                if x_feat.device != device:
                    x_feat = x_feat.to(device, non_blocking=True)
                if x_mask.device != device:
                    x_mask = x_mask.to(device, non_blocking=True)
                xb = (x_feat, x_mask)
            elif xb.device != device:
                xb = xb.to(device, non_blocking=True)
            if yb.device != device:
                yb = yb.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                if isinstance(xb, (tuple, list)):
                    logits = model(xb[0], xb[1])
                else:
                    logits = model(xb)
                loss = criterion(logits, yb)
            batch_n = yb.shape[0]
        else:
            xb = batch[0]
            batch_targets = batch[1:]
            if xb.device != device:
                xb = xb.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                outputs = model(xb)
                # Full key order must match TensorDataset column order from build_multilabel_targets
                full_keys = (
                    MULTILABEL_KEYS if multilabel == 1 else MULTILABEL_WITHOUT_PERMIND_KEYS
                )
                target_dict = {
                    k: (
                        batch_targets[idx].to(device, non_blocking=True)
                        if batch_targets[idx].device != device
                        else batch_targets[idx]
                    )
                    for idx, k in enumerate(full_keys)
                }
                loss = compute_multilabel_loss(outputs, target_dict, criterion, multilabel_keys,
                                               c_weight=c_weight)
            batch_n = xb.shape[0]
            if return_head_acc:
                for k in multilabel_keys:
                    head_correct[k] = head_correct.get(k, 0) + int(
                        (outputs[k].argmax(dim=1) == target_dict[k]).sum().item())
                head_total += batch_n

        loss_value = float(loss.detach().item()) if torch.isfinite(loss).item() else float('nan')
        if training:
            if not math.isfinite(loss_value):
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            if use_amp and HAS_AMP and scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        if math.isfinite(loss_value):
            epoch_loss += loss_value * batch_n
            sample_count += batch_n

    if skipped_nonfinite:
        print(f"  skipped_nonfinite_batches={skipped_nonfinite}")
    mean_loss = epoch_loss / float(sample_count) if sample_count else float('nan')
    if return_head_acc:
        accs = {k: v / float(head_total) for k, v in head_correct.items()} if head_total else {}
        return mean_loss, accs
    return mean_loss


# Training high level function.
def resolve_optimizer_config(network_type, model=None, learning_rate=None, optimizer_name=None):
    """Resolve optimizer name + LR for a network family.

    Defaults (when args are None) match historical recipes:
      - TimesNet: RAdam @ 1e-3 (benchmark profiles usually override LR to 1e-4)
      - MLPBest / CNNBest: RMSprop @ 1e-5
      - Graph (GCN/GAT/GIN/GraphSAGE) and other: Adam @ 1e-3

    ``learning_rate`` and ``optimizer_name`` override those defaults for all families
    (including GNN — previously LR was ignored outside the TimesNet branch).
    """
    opt = (optimizer_name or "").strip().lower() if optimizer_name else None
    if is_timesnet_network(network_type):
        name = opt or "radam"
        lr = 1e-3 if learning_rate is None else float(learning_rate)
    elif model is not None and isinstance(model, (MLPBest, CNNBest)):
        name = opt or "rmsprop"
        lr = 1e-5 if learning_rate is None else float(learning_rate)
    else:
        # Graph networks and any remaining models (default Adam path).
        name = opt or "adam"
        lr = 1e-3 if learning_rate is None else float(learning_rate)
    return name, lr


def build_optimizer(model, network_type, learning_rate=None, optimizer_name=None):
    """Create a torch optimizer and return (optimizer, name, lr)."""
    name, lr = resolve_optimizer_config(
        network_type,
        model=model,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
    )
    if name == "radam":
        optimizer = optim.RAdam(model.parameters(), lr=lr)
    elif name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(
            f"Unknown optimizer_name={name!r}. Supported: adam, radam, rmsprop."
        )
    return optimizer, name, lr


def train_model(
    X_profiling,
    Y_profiling,
    model,
    save_file_name,
    network_type,
    device,
    epochs=150,
    batch_size=100,
    multilabel=0,
    validation_split=0,
    early_stopping=0,
    learning_rate=None,
    optimizer_name=None,
    grad_clip=None,
    use_amp=None,
    train_heads=None,
    save_last=None,
    save_epochs=False,
    sidecar_meta=None,
    early_stopping_min_delta=0.0,
    early_stopping_patience=10,
    resume_checkpoint=None,
    resume_best_metric=None,
    graph_mask=None,
    c_loss_weight=1.0,
):
    out_dir = os.path.dirname(save_file_name)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if is_graph_network(network_type) and multilabel != 0:
        print("Error: multilabel modes are not supported for graph networks in this baseline implementation.")
        sys.exit(-1)

    if resume_checkpoint:
        check_file_exists(resume_checkpoint)
        try:
            state = torch.load(resume_checkpoint, map_location="cpu")
            state = strip_torch_compile_prefix(state)
            model.load_state_dict(state)
            print(f"Resumed weights from {resume_checkpoint}")
        except Exception as exc:
            print(f"Error: failed to load resume_checkpoint '{resume_checkpoint}': {exc}")
            sys.exit(-1)

    X = reshape_inputs_for_network(X_profiling, network_type)
    mask_t = None
    if graph_mask is not None:
        mask_t = torch.as_tensor(graph_mask, dtype=torch.bool)
    if multilabel == 1:
        y = build_multilabel_targets(Y_profiling, without_permind=False)
    elif multilabel == 2:
        y = build_multilabel_targets(Y_profiling, without_permind=True)
    else:
        y = torch.as_tensor(Y_profiling, dtype=torch.long)

    if early_stopping != 0 and validation_split == 0:
        validation_split = 0.1

    (X_train, y_train), (X_val, y_val) = split_train_validation(X, y, validation_split)
    mask_train = mask_val = None
    if mask_t is not None:
        # Re-split mask with same random indices: re-run split on indices via X shape.
        # split_train_validation already shuffled; recover by matching sizes is wrong.
        # Instead split mask jointly below.
        pass

    # Joint split for features + optional segment mask (re-do if mask present).
    if mask_t is not None:
        n_samples = X.shape[0]
        val_count = int(n_samples * validation_split)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        if val_count == 0:
            X_train, y_train = X, y
            X_val, y_val = None, None
            mask_train, mask_val = mask_t, None
        else:
            val_indices = indices[:val_count]
            train_indices = indices[val_count:]
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
            mask_train = mask_t[train_indices]
            mask_val = mask_t[val_indices]

    train_loader, _ = make_loader(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        network_type=network_type,
        device=device,
        graph_mask=mask_train,
    )
    val_loader = None
    if X_val is not None:
        val_loader, _ = make_loader(
            X_val,
            y_val,
            batch_size=batch_size,
            shuffle=False,
            network_type=network_type,
            device=device,
            graph_mask=mask_val,
        )

    model.to(device)

    # When using the GPU-resident path for fixed-topology graphs and memory
    # headroom exists, compiling can reduce overhead and fuse the dense ops.
    # Skip compile for segment graphs: path+mask kernels are small and compile
    # prefixes checkpoint keys with `_orig_mod.` (painful for eval).
    if is_graph_network(network_type) and not is_segment_graph_network(network_type) and device.type == 'cuda':
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except Exception:
            pass  # compile is best-effort (depends on PyTorch version / model)
    optimizer, opt_name, lr = build_optimizer(
        model,
        network_type,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
    )

    # Defaults tuned for TimesNet multilabel stability.
    if grad_clip is None:
        grad_clip = 1.0 if is_timesnet_multilabel_network(network_type) else 0.0
    else:
        grad_clip = float(grad_clip)
    if use_amp is None:
        amp_enabled = (device.type == 'cuda') and HAS_AMP
    else:
        amp_enabled = bool(use_amp) and (device.type == 'cuda') and HAS_AMP
    if save_last is None:
        save_last = is_timesnet_multilabel_network(network_type)

    print(
        f"Optimizer: {opt_name}  learning_rate: {lr}  grad_clip: {grad_clip}  "
        f"amp: {amp_enabled}  early_stopping: {early_stopping}"
    )
    if train_heads:
        print(f"train_heads: {train_heads}")

    scaler = GradScaler('cuda') if amp_enabled else None

    best_metric = float('inf')
    if resume_best_metric is not None:
        try:
            best_metric = float(resume_best_metric)
            print(f"resume_best_metric={best_metric} (only save best if val/train improves further)")
        except (TypeError, ValueError):
            best_metric = float('inf')
    elif resume_checkpoint:
        # Try sidecar metric from the resume file so we don't overwrite with worse CE.
        side_path = resume_checkpoint + ".config.json"
        if os.path.exists(side_path):
            try:
                with open(side_path, "r", encoding="utf-8") as f:
                    side = json.load(f)
                if side.get("metric") is not None:
                    best_metric = float(side["metric"])
                    print(f"resume_best_metric from sidecar={best_metric}")
            except Exception:
                pass
    patience = int(early_stopping_patience) if early_stopping_patience else 10
    min_delta = float(early_stopping_min_delta or 0.0)
    patience_counter = 0
    history = {
        'loss': [],
        'val_loss': [],
        'val_head_acc': [],
        'optimizer': opt_name,
        'learning_rate': lr,
        'grad_clip': grad_clip,
        'use_amp': amp_enabled,
        'resume_checkpoint': resume_checkpoint,
    }

    def _write_history():
        """Per-epoch curve on disk as it happens, so a stopped run still has the alpha history."""
        try:
            with open(save_file_name + ".history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except OSError as exc:
            print(f"Warning: could not write training history: {exc}")

    def _write_ckpt(path, metric_value, kind):
        torch.save(model.state_dict(), path)
        meta = {
            'kind': kind,
            'network_type': network_type,
            'multilabel': multilabel,
            'metric': metric_value if metric_value is None or math.isfinite(metric_value) else None,
            'learning_rate': lr,
            'optimizer': opt_name,
            'grad_clip': grad_clip,
            'use_amp': amp_enabled,
            'train_heads': train_heads,
        }
        if sidecar_meta:
            meta.update(sidecar_meta)
        save_checkpoint_sidecar(path, meta)

    for epoch in range(epochs):
        train_loss = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            multilabel=multilabel,
            use_amp=amp_enabled,
            scaler=scaler,
            grad_clip=grad_clip,
            train_heads=train_heads,
            c_weight=c_loss_weight,
        )
        history['loss'].append(train_loss)

        if val_loader is not None:
            want_acc = multilabel != 0
            with torch.no_grad():
                res = run_epoch(
                    model,
                    val_loader,
                    device,
                    optimizer=None,
                    multilabel=multilabel,
                    use_amp=amp_enabled,
                    scaler=None,
                    grad_clip=0.0,
                    train_heads=train_heads,
                    return_head_acc=want_acc,
                    c_weight=c_loss_weight,
                )
            val_loss, val_acc = res if want_acc else (res, {})
            history['val_loss'].append(val_loss)
            history['val_head_acc'].append(val_acc)
            metric = val_loss
            print(
                f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.6f} - val_loss: {val_loss:.6f}"
            )
            if val_acc:
                c_accs = [v for k, v in val_acc.items() if k.startswith('sbox_')]
                p_accs = [v for k, v in val_acc.items() if k.startswith('permind_')]
                print(
                    "  val head top-1: "
                    f"alpha={val_acc.get('alpha_output', float('nan')) * 100:.2f}% "
                    f"beta={val_acc.get('beta_output', float('nan')) * 100:.2f}%"
                    + (f" C_mean={np.mean(c_accs) * 100:.2f}% C_max={np.max(c_accs) * 100:.2f}%"
                       if c_accs else "")
                    + (f" pi_mean={np.mean(p_accs) * 100:.2f}%" if p_accs else "")
                )
            _write_history()
        else:
            metric = train_loss
            print(f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.6f}")
            _write_history()

        improved = False
        if math.isfinite(metric) and (metric < best_metric - min_delta):
            best_metric = metric
            _write_ckpt(save_file_name, metric, 'best')
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1

        if not improved and not math.isfinite(metric):
            print(f"  non-finite metric at epoch {epoch + 1}; best ckpt preserved")

        if save_epochs:
            root, ext = os.path.splitext(save_file_name)
            ep_path = f"{root}_ep{epoch + 1}{ext or '.pth'}"
            _write_ckpt(ep_path, metric, f'epoch_{epoch + 1}')

        if early_stopping != 0 and val_loader is not None and patience_counter >= patience:
            print("Early stopping triggered.")
            break

    if save_last:
        root, ext = os.path.splitext(save_file_name)
        last_path = f"{root}_last{ext or '.pth'}"
        last_metric = history['val_loss'][-1] if history['val_loss'] else history['loss'][-1]
        _write_ckpt(last_path, last_metric, 'last')
        print(f"Saved last checkpoint: {last_path}")

    return history


def read_parameters_from_file(param_filename):
    # read parameters for the train_model and load_ascad functions
    # TODO: sanity checks on parameters
    param_file = open(param_filename, "r")

    # TODO: replace eval() by ast.literal_eval()
    my_parameters = eval(param_file.read())

    ascad_database = my_parameters["ascad_database"]
    training_model = my_parameters["training_model"]
    network_type = my_parameters["network_type"]
    epochs = my_parameters["epochs"]
    batch_size = my_parameters["batch_size"]
    train_len = 0
    if "train_len" in my_parameters:
        train_len = my_parameters["train_len"]
    validation_split = 0
    if "validation_split" in my_parameters:
        validation_split = my_parameters["validation_split"]
    multilabel = 0
    if "multilabel" in my_parameters:
        multilabel = my_parameters["multilabel"]
    early_stopping = 0
    if "early_stopping" in my_parameters:
        early_stopping = my_parameters["early_stopping"]
    graph_hidden_dim = 128
    if "graph_hidden_dim" in my_parameters:
        graph_hidden_dim = my_parameters["graph_hidden_dim"]
    graph_num_layers = 2
    if "graph_num_layers" in my_parameters:
        graph_num_layers = my_parameters["graph_num_layers"]
    graph_dropout = 0.2
    if "graph_dropout" in my_parameters:
        graph_dropout = my_parameters["graph_dropout"]
    graph_heads = 4
    if "graph_heads" in my_parameters:
        graph_heads = my_parameters["graph_heads"]
    learning_rate = None
    if "learning_rate" in my_parameters:
        learning_rate = my_parameters["learning_rate"]
    optimizer_name = None
    if "optimizer_name" in my_parameters:
        optimizer_name = my_parameters["optimizer_name"]
    timesnet_config = None
    if "timesnet_config" in my_parameters:
        timesnet_config = my_parameters["timesnet_config"]
    grad_clip = None
    if "grad_clip" in my_parameters:
        grad_clip = my_parameters["grad_clip"]
    use_amp = None
    if "use_amp" in my_parameters:
        use_amp = my_parameters["use_amp"]
    train_heads = None
    if "train_heads" in my_parameters:
        train_heads = my_parameters["train_heads"]
    save_last = None
    if "save_last" in my_parameters:
        save_last = my_parameters["save_last"]
    early_stopping_min_delta = 0.0
    if "early_stopping_min_delta" in my_parameters:
        early_stopping_min_delta = my_parameters["early_stopping_min_delta"]
    early_stopping_patience = 10
    if "early_stopping_patience" in my_parameters:
        early_stopping_patience = my_parameters["early_stopping_patience"]
    resume_checkpoint = None
    if "resume_checkpoint" in my_parameters:
        resume_checkpoint = my_parameters["resume_checkpoint"]
    resume_best_metric = None
    if "resume_best_metric" in my_parameters:
        resume_best_metric = my_parameters["resume_best_metric"]
    segment_config = None
    if "segment_config" in my_parameters:
        segment_config = my_parameters["segment_config"]
    return (
        ascad_database,
        training_model,
        network_type,
        epochs,
        batch_size,
        train_len,
        validation_split,
        multilabel,
        early_stopping,
        graph_hidden_dim,
        graph_num_layers,
        graph_dropout,
        graph_heads,
        learning_rate,
        optimizer_name,
        timesnet_config,
        grad_clip,
        use_amp,
        train_heads,
        save_last,
        early_stopping_min_delta,
        early_stopping_patience,
        resume_checkpoint,
        resume_best_metric,
        segment_config,
    )


if __name__ == "__main__":
    if len(sys.argv) == 2:
        with open(sys.argv[1], "r", encoding="utf-8") as _pf:
            _raw_params = eval(_pf.read())
        if is_poi_searcher_network(_raw_params.get("network_type")):
            from leakage_localizer.train import train_from_params

            train_from_params(_raw_params)
            sys.exit(0)
    if len(sys.argv) != 2:
        ascad_database = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
        network_type = "mlp"
        training_model = (
            "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/"
            "ASCAD_trained_models/my_mlp_best_desync0_epochs75_batchsize200.pth"
        )
        validation_split = 0
        multilabel = 0
        train_len = 0
        epochs = 75
        batch_size = 200
        early_stopping = 0
        graph_hidden_dim = 128
        graph_num_layers = 2
        graph_dropout = 0.2
        graph_heads = 4
        learning_rate = None
        optimizer_name = None
        timesnet_config = None
        grad_clip = None
        use_amp = None
        train_heads = None
        save_last = None
        early_stopping_min_delta = 0.0
        early_stopping_patience = 10
        resume_checkpoint = None
        resume_best_metric = None
        segment_config = None
    else:
        (
            ascad_database,
            training_model,
            network_type,
            epochs,
            batch_size,
            train_len,
            validation_split,
            multilabel,
            early_stopping,
            graph_hidden_dim,
            graph_num_layers,
            graph_dropout,
            graph_heads,
            learning_rate,
            optimizer_name,
            timesnet_config,
            grad_clip,
            use_amp,
            train_heads,
            save_last,
            early_stopping_min_delta,
            early_stopping_patience,
            resume_checkpoint,
            resume_best_metric,
            segment_config,
        ) = read_parameters_from_file(sys.argv[1])

    # Optional per-run seed and C-head loss weight (T-031). Both are additive: the defaults
    # reproduce the previous behaviour exactly. Seeding happens here, before load_ascad, before
    # split_train_validation (which shuffles with the global numpy RNG) and before model init, so
    # two runs given the same seed share initialisation, data order and validation split.
    c_loss_weight = float(_raw_params.get("c_loss_weight", 1.0))
    if not (0.0 <= c_loss_weight <= 1.0):
        raise SystemExit(f"c_loss_weight must be in [0, 1], got {c_loss_weight}")
    if c_loss_weight != 1.0:
        print(f"c_loss_weight = {c_loss_weight}: the 16 sbox_* loss terms scale, "
              "alpha/beta/permind stay at weight 1; the loss stays a weighted MEAN so val_loss "
              "remains comparable across arms")
    seed = _raw_params.get("seed", None)
    if seed is not None:
        import random
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Seed {seed} applied to python/numpy/torch before data load, split and model init")

    # Slice profiling inside HDF5 when train_len is set (keeps host RAM sane on v2).
    load_meta = (multilabel == 0)
    res = load_ascad(
        ascad_database,
        load_metadata=load_meta,
        train_len=train_len if train_len else 0,
        load_attack=False,
    )
    if load_meta:
        (X_profiling, Y_profiling), _, (meta_p, _) = res
    else:
        (X_profiling, Y_profiling), _ = res

    target_byte = int(_raw_params.get("target_byte", 2))
    target_field = _raw_params.get("target_field", None)
    if multilabel == 0:
        if getattr(Y_profiling.dtype, "names", None) is not None or target_field == "sbox":
            from ASCAD_generate import AES_Sbox
            pt = np.asarray(meta_p["plaintext"][:, target_byte], dtype=np.uint8)
            k = np.asarray(meta_p["key"][:, target_byte], dtype=np.uint8)
            Y_profiling = AES_Sbox[pt ^ k].astype(np.int64)
            print(f"Single-label S-box mode: target_byte={target_byte}, classes={len(np.unique(Y_profiling))}")
        elif target_field and getattr(Y_profiling.dtype, "names", None) is not None:
            Y_profiling = np.asarray(Y_profiling[target_field][:, target_byte], dtype=np.int64)
            print(f"Single-label field mode: field={target_field}, target_byte={target_byte}")

    sub_avg = bool(_raw_params.get("sub_avg", False) or _raw_params.get("subtract_mean_waveform", False))
    mean_wf_path = None
    if sub_avg:
        print("Subtracting average waveform across profiling traces...")
        mean_wf = np.mean(X_profiling, axis=0, keepdims=True, dtype=np.float32)
        X_profiling = (X_profiling.astype(np.float32) - mean_wf)
        mean_wf_path = training_model + ".mean_wf.npy"
        out_dir = os.path.dirname(training_model)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.save(mean_wf_path, mean_wf.squeeze(0))
        print(f"Saved mean waveform to {mean_wf_path}")
    print(
        f"Loaded profiling: X={X_profiling.shape} labels={getattr(Y_profiling, 'shape', type(Y_profiling))} "
        f"network={network_type} multilabel={multilabel} sub_avg={sub_avg}"
    )

    ensure_graph_dependencies(network_type)

    # Align multilabel head layout for multi TimesNet / multi-ResNet.
    if timesnet_config is None:
        timesnet_config = {}
    else:
        timesnet_config = dict(timesnet_config)
    if is_timesnet_multilabel_network(network_type):
        timesnet_config['without_permind'] = (multilabel == 2)
        if multilabel not in (1, 2):
            print("Error: timesnet multi variants require multilabel=1 or 2")
            sys.exit(-1)

    if segment_config is None:
        segment_config = {}
    else:
        segment_config = dict(segment_config)

    graph_mask = None
    input_dim = len(X_profiling[0])
    if is_segment_graph_network(network_type):
        seg_f = int(segment_config.get("out_dim", DEFAULT_OUT_DIM))
        seg_th = int(segment_config.get("threshold", DEFAULT_THRESHOLD))
        seg_kind = segment_config.get("segmenter", DEFAULT_SEGMENTER)
        seg_center = bool(segment_config.get("center", False))
        edge_mode = segment_config.get("edge_mode", DEFAULT_EDGE_MODE)
        segment_config["edge_mode"] = edge_mode
        cache_dir = segment_config.get("cache_dir")
        if not cache_dir:
            cache_dir = os.path.join(
                os.path.dirname(training_model) or ".",
                "segment_feature_cache",
            )

        fixed_windows = segment_config.get("fixed_windows")
        snr_extra = ""
        if seg_kind == "snr":
            half_w = int(segment_config.get("snr_half_width", 12))
            max_peaks = int(segment_config.get("snr_max_peaks", 16))
            min_dist = int(segment_config.get("snr_min_distance", seg_th))
            if not fixed_windows:
                print(
                    f"Discovering SNR peaks on profiling "
                    f"(half_width={half_w}, max_peaks={max_peaks}, min_distance={min_dist})..."
                )
                disc = discover_snr_windows(
                    np.asarray(X_profiling, dtype=np.float32),
                    np.asarray(Y_profiling).ravel(),
                    half_width=half_w,
                    min_distance=min_dist,
                    max_peaks=max_peaks,
                )
                fixed_windows = disc["windows"]
                segment_config["fixed_windows"] = fixed_windows
                segment_config["snr_peaks"] = disc["peaks"].tolist()
                segment_config["snr_max"] = disc["snr_max"]
                segment_config["snr_mean"] = disc["snr_mean"]
                print(
                    f"  SNR peaks ({disc['n_peaks']}): {disc['peaks'].tolist()} "
                    f"max_snr={disc['snr_max']:.4f}"
                )
            else:
                fixed_windows = [(int(s), int(e)) for s, e in fixed_windows]
                segment_config["fixed_windows"] = fixed_windows
            # Prefer N = #windows unless user overrode num_nodes larger
            n_win = max(1, len(fixed_windows))
            if "num_nodes" not in segment_config:
                segment_config["num_nodes"] = n_win
            else:
                segment_config["num_nodes"] = max(int(segment_config["num_nodes"]), n_win)
            snr_extra = "snr_p%s_w%d" % (
                "-".join(str(p) for p in segment_config.get("snr_peaks", [])),
                half_w,
            )

        seg_n = int(segment_config.get("num_nodes", DEFAULT_NUM_NODES))
        cpath = cache_path_for(
            cache_dir,
            ascad_database,
            split="profiling" + (f"_t{train_len}" if train_len else ""),
            num_nodes=seg_n,
            out_dim=seg_f,
            segmenter=seg_kind,
            threshold=seg_th,
            center=seg_center,
            n_traces=X_profiling.shape[0],
            trace_len=X_profiling.shape[1],
            extra=snr_extra,
        )
        seg_std = bool(segment_config.get("standardize", True))
        build_kwargs = {}
        if seg_kind == "snr":
            build_kwargs["fixed_windows"] = fixed_windows
        X_profiling, graph_mask = load_or_build_segment_features(
            X_profiling,
            cache_path=cpath,
            num_nodes=seg_n,
            out_dim=seg_f,
            segmenter=seg_kind,
            threshold=seg_th,
            center=seg_center,
            standardize=seg_std,
            **build_kwargs,
        )
        if seg_std and getattr(load_or_build_segment_features, "last_stats", None) is not None:
            mu, std = load_or_build_segment_features.last_stats
            segment_config["feature_mu"] = mu.tolist()
            segment_config["feature_std"] = std.tolist()
        input_dim = seg_f
        print(
            f"Segment graph: N={seg_n} F={seg_f} segmenter={seg_kind} edge_mode={edge_mode} "
            f"mask_frac_real={float(graph_mask.mean()):.3f} standardize={seg_std}"
        )

    if is_timesnet_network(network_type):
        # Keep raw length in sidecar; factory applies stride for internal seq_len.
        timesnet_config.setdefault('seq_len', input_dim)
        best_model = build_timesnet_for_network(
            network_type, input_dim, timesnet_config=timesnet_config
        )
        print(
            f"TimesNet config: stride={timesnet_config.get('input_stride', 1)} "
            f"pool_segments={timesnet_config.get('pool_segments', 0)} "
            f"pool_mode={timesnet_config.get('pool_mode', 'avg')} "
            f"eff_seq_len={_effective_timesnet_seq_len(input_dim, timesnet_config)}"
        )
    elif network_type == "mlp":
        best_model = MLPBest(input_dim=input_dim)
    elif network_type == "cnn":
        best_model = CNNBest(input_dim=input_dim)
    elif network_type == "cnn2":
        best_model = CNNBest(input_dim=input_dim, first_stride=2)
    elif network_type == "multi_test":
        best_model = MultiTest(input_dim=input_dim)
    elif network_type == "multi_resnet":
        best_model = MultiResNet(input_dim=input_dim, depth=19, without_permind=False)
    elif network_type == "multi_resnet_without_permind":
        best_model = MultiResNet(input_dim=input_dim, depth=19, without_permind=True)
    elif network_type == "gcn":
        best_model = GCNClassifier(
            input_dim=1,
            hidden_dim=graph_hidden_dim,
            num_layers=graph_num_layers,
            dropout=graph_dropout,
        )
    elif network_type == "gat":
        best_model = GATClassifier(
            input_dim=1,
            hidden_dim=graph_hidden_dim,
            num_layers=graph_num_layers,
            dropout=graph_dropout,
            heads=graph_heads,
        )
    elif network_type == "gin":
        best_model = GINClassifier(
            input_dim=1,
            hidden_dim=graph_hidden_dim,
            num_layers=graph_num_layers,
            dropout=graph_dropout,
        )
    elif network_type == "graphsage":
        best_model = GraphSAGEClassifier(
            input_dim=1,
            hidden_dim=graph_hidden_dim,
            num_layers=graph_num_layers,
            dropout=graph_dropout,
        )
    elif network_type == "seg_graphsage":
        best_model = SegmentGraphSAGEClassifier(
            input_dim=int(segment_config.get("out_dim", DEFAULT_OUT_DIM)),
            hidden_dim=graph_hidden_dim,
            num_layers=graph_num_layers,
            dropout=graph_dropout,
            num_nodes=int(segment_config.get("num_nodes", DEFAULT_NUM_NODES)),
            edge_mode=segment_config.get("edge_mode", DEFAULT_EDGE_MODE),
        )
    else:
        print("Error: no topology found for network '%s' ..." % network_type)
        sys.exit(-1)

    nparams = sum(p.numel() for p in best_model.parameters())
    print(f"Model params: {nparams / 1e6:.2f}M")

    out_dir = os.path.dirname(training_model)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    sidecar = {
        'ascad_database': ascad_database,
        'train_len': train_len,
        'epochs': epochs,
        'batch_size': batch_size,
        'validation_split': validation_split,
        'timesnet_config': timesnet_config if is_timesnet_network(network_type) else None,
        'segment_config': segment_config if is_segment_graph_network(network_type) else None,
        'input_dim': input_dim,
        'resume_checkpoint': resume_checkpoint,
        'sub_avg': sub_avg,
        'mean_wf_path': mean_wf_path,
        'target_byte': target_byte if multilabel == 0 else None,
        'c_loss_weight': c_loss_weight,
        'seed': seed,
        'note': ('seed=None means the trainer set no RNG seed, so run-to-run spread is unmeasured; '
                 'and split_train_validation is seeded only through the global numpy RNG'),
    }

    train_model(
        X_profiling,
        Y_profiling,
        best_model,
        training_model,
        network_type,
        device,
        epochs,
        batch_size,
        multilabel,
        validation_split,
        early_stopping,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        grad_clip=grad_clip,
        use_amp=use_amp,
        train_heads=train_heads,
        save_last=save_last,
        save_epochs=bool(_raw_params.get("save_epochs", False)),
        sidecar_meta=sidecar,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_patience=early_stopping_patience,
        resume_checkpoint=resume_checkpoint,
        resume_best_metric=resume_best_metric,
        graph_mask=graph_mask,
        c_loss_weight=c_loss_weight,
    )
