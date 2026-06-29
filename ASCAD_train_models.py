import os
import os.path
import sys
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from GNN.classifiers import GATClassifier, GCNClassifier, GINClassifier, GraphSAGEClassifier
from GNN.data import make_graph_loader
from GNN.utils import ensure_graph_dependencies, is_graph_network
from TimesNet import build_timesnet_model


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


def load_sca_model(model_file, network_type, input_dim, device):
    check_file_exists(model_file)
    ensure_graph_dependencies(network_type)
    if network_type == 'mlp':
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
    else:
        print("Error: no topology found for network '%s' ..." % network_type)
        sys.exit(-1)

    try:
        state = torch.load(model_file, map_location=device)
        model.load_state_dict(state)
    except Exception:
        print("Error: can't load PyTorch model file '%s'" % model_file)
        sys.exit(-1)

    model.to(device)
    model.eval()
    return model


# Loads the profiling and attack datasets from the ASCAD database.
def load_ascad(ascad_database_file, load_metadata=False):
    check_file_exists(ascad_database_file)
    try:
        in_file = h5py.File(ascad_database_file, "r")
    except Exception:
        print(
            "Error: can't open HDF5 file '%s' for reading (it might be malformed) ..."
            % ascad_database_file
        )
        sys.exit(-1)

    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
    Y_profiling = np.array(in_file['Profiling_traces/labels'])
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
    Y_attack = np.array(in_file['Attack_traces/labels'])
    if load_metadata is False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    return (
        (X_profiling, Y_profiling),
        (X_attack, Y_attack),
        (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata']),
    )


def build_multilabel_targets(Y, without_permind=False):
    y = {
        'alpha_output': torch.as_tensor(Y['alpha_mask'], dtype=torch.long),
        'beta_output': torch.as_tensor(Y['beta_mask'], dtype=torch.long),
    }
    sbox_field = 'sbox_masked_with_perm' if without_permind else 'sbox_masked'
    for i in range(16):
        y[f'sbox_{i}_output'] = torch.as_tensor(Y[sbox_field][:, i], dtype=torch.long)
    if not without_permind:
        for i in range(16):
            y[f'permind_{i}_output'] = torch.as_tensor(Y['perm_index'][:, i], dtype=torch.long)
    return y


def is_timesnet_network(network_type):
    return network_type in ('timesnet', 'timesnet_ascad_head')


def reshape_inputs_for_network(X, network_type):
    x = torch.as_tensor(X, dtype=torch.float32)
    if network_type in ('mlp',):
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


def make_loader(X, y, batch_size, shuffle, network_type):
    if is_graph_network(network_type):
        return make_graph_loader(X, y, batch_size, shuffle)

    if isinstance(y, dict):
        keys = list(y.keys())
        tensors = [X] + [y[k] for k in keys]
        ds = TensorDataset(*tensors)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle), keys

    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle), None


def compute_multilabel_loss(outputs, target_dict, criterion, keys):
    total_loss = 0.0
    for k in keys:
        total_loss = total_loss + criterion(outputs[k], target_dict[k])
    return total_loss / float(len(keys))


def run_epoch(model, loader, device, optimizer=None, multilabel=0):
    training = optimizer is not None
    criterion = nn.CrossEntropyLoss()
    epoch_loss = 0.0
    sample_count = 0

    if training:
        model.train()
    else:
        model.eval()

    for batch in loader:
        if multilabel == 0:
            xb, yb = batch
            xb = xb.to(device)
            yb = yb.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            batch_n = yb.shape[0]
        else:
            xb = batch[0].to(device)
            batch_targets = batch[1:]
            if training:
                optimizer.zero_grad()
            outputs = model(xb)
            keys = (
                MULTILABEL_KEYS if multilabel == 1 else MULTILABEL_WITHOUT_PERMIND_KEYS
            )
            target_dict = {
                k: batch_targets[idx].to(device)
                for idx, k in enumerate(keys)
            }
            loss = compute_multilabel_loss(outputs, target_dict, criterion, keys)
            batch_n = xb.shape[0]

        if training:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.detach().item() * batch_n
        sample_count += batch_n

    if sample_count == 0:
        return 0.0
    return epoch_loss / float(sample_count)


# Training high level function.
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
):
    check_file_exists(os.path.dirname(save_file_name))

    if is_graph_network(network_type) and multilabel != 0:
        print("Error: multilabel modes are not supported for graph networks in this baseline implementation.")
        sys.exit(-1)

    X = reshape_inputs_for_network(X_profiling, network_type)
    if multilabel == 1:
        y = build_multilabel_targets(Y_profiling, without_permind=False)
    elif multilabel == 2:
        y = build_multilabel_targets(Y_profiling, without_permind=True)
    else:
        y = torch.as_tensor(Y_profiling, dtype=torch.long)

    if early_stopping != 0 and validation_split == 0:
        validation_split = 0.1

    (X_train, y_train), (X_val, y_val) = split_train_validation(X, y, validation_split)

    train_loader, _ = make_loader(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        network_type=network_type,
    )
    val_loader = None
    if X_val is not None:
        val_loader, _ = make_loader(
            X_val,
            y_val,
            batch_size=batch_size,
            shuffle=False,
            network_type=network_type,
        )

    model.to(device)
    if is_timesnet_network(network_type):
        optimizer = optim.RAdam(model.parameters(), lr=1e-3)
    elif isinstance(model, (MLPBest, CNNBest)):
        optimizer = optim.RMSprop(model.parameters(), lr=1e-5)
    else:
        optimizer = optim.Adam(model.parameters())

    best_metric = float('inf')
    patience = 10
    patience_counter = 0
    history = {'loss': [], 'val_loss': []}

    for epoch in range(epochs):
        train_loss = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            multilabel=multilabel,
        )
        history['loss'].append(train_loss)

        if val_loader is not None:
            with torch.no_grad():
                val_loss = run_epoch(
                    model,
                    val_loader,
                    device,
                    optimizer=None,
                    multilabel=multilabel,
                )
            history['val_loss'].append(val_loss)
            metric = val_loss
            print(
                f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.6f} - val_loss: {val_loss:.6f}"
            )
        else:
            metric = train_loss
            print(f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.6f}")

        if metric < best_metric:
            best_metric = metric
            torch.save(model.state_dict(), save_file_name)
            patience_counter = 0
        else:
            patience_counter += 1

        if early_stopping != 0 and val_loader is not None and patience_counter >= patience:
            print("Early stopping triggered.")
            break

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
    graph_num_layers = 3
    if "graph_num_layers" in my_parameters:
        graph_num_layers = my_parameters["graph_num_layers"]
    graph_dropout = 0.2
    if "graph_dropout" in my_parameters:
        graph_dropout = my_parameters["graph_dropout"]
    graph_heads = 4
    if "graph_heads" in my_parameters:
        graph_heads = my_parameters["graph_heads"]
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
    )


if __name__ == "__main__":
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
        graph_num_layers = 3
        graph_dropout = 0.2
        graph_heads = 4
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
        ) = read_parameters_from_file(sys.argv[1])

    (X_profiling, Y_profiling), _ = load_ascad(ascad_database)

    ensure_graph_dependencies(network_type)

    if network_type == "mlp":
        best_model = MLPBest(input_dim=len(X_profiling[0]))
    elif network_type == "cnn":
        best_model = CNNBest(input_dim=len(X_profiling[0]))
    elif network_type == "cnn2":
        best_model = CNNBest(input_dim=len(X_profiling[0]), first_stride=2)
    elif network_type == "multi_test":
        best_model = MultiTest(input_dim=len(X_profiling[0]))
    elif network_type == "multi_resnet":
        best_model = MultiResNet(input_dim=len(X_profiling[0]), depth=19, without_permind=False)
    elif network_type == "multi_resnet_without_permind":
        best_model = MultiResNet(input_dim=len(X_profiling[0]), depth=19, without_permind=True)
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
    else:
        print("Error: no topology found for network '%s' ..." % network_type)
        sys.exit(-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if train_len == 0:
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
        )
    else:
        train_model(
            X_profiling[:train_len],
            Y_profiling[:train_len],
            best_model,
            training_model,
            network_type,
            device,
            epochs,
            batch_size,
            multilabel,
            validation_split,
            early_stopping,
        )
