import os
import matplotlib as mpl

# Take care of case where no graphical display is available.
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
else:
    mpl.use('TkAgg')

import os.path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch

from GNN.classifiers import GATClassifier, GCNClassifier, GINClassifier, GraphSAGEClassifier
from GNN.data import build_graph_batch_from_traces
from GNN.utils import ensure_graph_dependencies, is_graph_network

from ASCAD_train_models import (
    MLPBest,
    CNNBest,
    MultiResNet,
    MultiTest,
    is_timesnet_network,
)
from TimesNet import build_timesnet_model

# The AES SBox that we will use to compute the rank.
AES_Sbox = np.array([
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
        ])

# Two tables to process a field multiplication over GF(256):
# a*b = alog(log(a) + log(b) mod 255).
log_table = [0, 0, 25, 1, 50, 2, 26, 198, 75, 199, 27, 104, 51, 238, 223, 3,
    100, 4, 224, 14, 52, 141, 129, 239, 76, 113, 8, 200, 248, 105, 28, 193,
    125, 194, 29, 181, 249, 185, 39, 106, 77, 228, 166, 114, 154, 201, 9, 120,
    101, 47, 138, 5, 33, 15, 225, 36, 18, 240, 130, 69, 53, 147, 218, 142,
    150, 143, 219, 189, 54, 208, 206, 148, 19, 92, 210, 241, 64, 70, 131, 56,
    102, 221, 253, 48, 191, 6, 139, 98, 179, 37, 226, 152, 34, 136, 145, 16,
    126, 110, 72, 195, 163, 182, 30, 66, 58, 107, 40, 84, 250, 133, 61, 186,
    43, 121, 10, 21, 155, 159, 94, 202, 78, 212, 172, 229, 243, 115, 167, 87,
    175, 88, 168, 80, 244, 234, 214, 116, 79, 174, 233, 213, 231, 230, 173, 232,
    44, 215, 117, 122, 235, 22, 11, 245, 89, 203, 95, 176, 156, 169, 81, 160,
    127, 12, 246, 111, 23, 196, 73, 236, 216, 67, 31, 45, 164, 118, 123, 183,
    204, 187, 62, 90, 251, 96, 177, 134, 59, 82, 161, 108, 170, 85, 41, 157,
    151, 178, 135, 144, 97, 190, 220, 252, 188, 149, 207, 205, 55, 63, 91, 209,
    83, 57, 132, 60, 65, 162, 109, 71, 20, 42, 158, 93, 86, 242, 211, 171,
    68, 17, 146, 217, 35, 32, 46, 137, 180, 124, 184, 38, 119, 153, 227, 165,
    103, 74, 237, 222, 197, 49, 254, 24, 13, 99, 140, 128, 192, 247, 112, 7]

alog_table = [1, 3, 5, 15, 17, 51, 85, 255, 26, 46, 114, 150, 161, 248, 19, 53,
    95, 225, 56, 72, 216, 115, 149, 164, 247, 2, 6, 10, 30, 34, 102, 170,
    229, 52, 92, 228, 55, 89, 235, 38, 106, 190, 217, 112, 144, 171, 230, 49,
    83, 245, 4, 12, 20, 60, 68, 204, 79, 209, 104, 184, 211, 110, 178, 205,
    76, 212, 103, 169, 224, 59, 77, 215, 98, 166, 241, 8, 24, 40, 120, 136,
    131, 158, 185, 208, 107, 189, 220, 127, 129, 152, 179, 206, 73, 219, 118, 154,
    181, 196, 87, 249, 16, 48, 80, 240, 11, 29, 39, 105, 187, 214, 97, 163,
    254, 25, 43, 125, 135, 146, 173, 236, 47, 113, 147, 174, 233, 32, 96, 160,
    251, 22, 58, 78, 210, 109, 183, 194, 93, 231, 50, 86, 250, 21, 63, 65,
    195, 94, 226, 61, 71, 201, 64, 192, 91, 237, 44, 116, 156, 191, 218, 117,
    159, 186, 213, 100, 172, 239, 42, 126, 130, 157, 188, 223, 122, 142, 137, 128,
    155, 182, 193, 88, 232, 35, 101, 175, 234, 37, 111, 177, 200, 67, 197, 84,
    252, 31, 33, 99, 165, 244, 7, 9, 27, 45, 119, 153, 176, 203, 70, 202,
    69, 207, 74, 222, 121, 139, 134, 145, 168, 227, 62, 66, 198, 81, 243, 14,
    18, 54, 90, 238, 41, 123, 141, 140, 143, 138, 133, 148, 167, 242, 13, 23,
    57, 75, 221, 124, 132, 151, 162, 253, 28, 36, 108, 180, 199, 82, 246, 1]


def multGF256(a, b):
    if (a == 0) or (b == 0):
        return 0
    return alog_table[(log_table[a] + log_table[b]) % 255]


def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path) is False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)


def build_model(network_type, input_dim, graph_hidden_dim=128, graph_num_layers=3, graph_dropout=0.2, graph_heads=4, timesnet_config=None):
    if network_type in ('timesnet', 'timesnet_ascad_head'):
        variant = 'ascad_head' if network_type == 'timesnet_ascad_head' else 'baseline'
        cfg_kwargs = dict(timesnet_config) if timesnet_config else {}
        cfg_kwargs.setdefault('seq_len', input_dim)
        return build_timesnet_model(variant=variant, **cfg_kwargs)
    ensure_graph_dependencies(network_type)
    if network_type == 'mlp':
        return MLPBest(input_dim=input_dim)
    if network_type == 'cnn':
        return CNNBest(input_dim=input_dim)
    if network_type == 'cnn2':
        return CNNBest(input_dim=input_dim, first_stride=2)
    if network_type == 'multi_test':
        return MultiTest(input_dim=input_dim)
    if network_type == 'multi_resnet':
        return MultiResNet(input_dim=input_dim, depth=19, without_permind=False)
    if network_type == 'multi_resnet_without_permind':
        return MultiResNet(input_dim=input_dim, depth=19, without_permind=True)
    if network_type == 'gcn':
        return GCNClassifier(
            input_dim=1,
            hidden_dim=graph_hidden_dim,
            num_layers=graph_num_layers,
            dropout=graph_dropout,
        )
    if network_type == 'gat':
        return GATClassifier(
            input_dim=1,
            hidden_dim=graph_hidden_dim,
            num_layers=graph_num_layers,
            dropout=graph_dropout,
            heads=graph_heads,
        )
    if network_type == 'gin':
        return GINClassifier(
            input_dim=1,
            hidden_dim=graph_hidden_dim,
            num_layers=graph_num_layers,
            dropout=graph_dropout,
        )
    if network_type == 'graphsage':
        return GraphSAGEClassifier(
            input_dim=1,
            hidden_dim=graph_hidden_dim,
            num_layers=graph_num_layers,
            dropout=graph_dropout,
        )
    print("Error: no topology found for network '%s' ..." % network_type)
    sys.exit(-1)


def infer_network_type_from_filename(model_file, multilabel):
    lower = os.path.basename(model_file).lower()
    if multilabel == 1:
        return 'multi_resnet'
    if multilabel == 2:
        return 'multi_resnet_without_permind'
    if 'cnn2' in lower:
        return 'cnn2'
    if 'timesnet_ascad_head' in lower:
        return 'timesnet_ascad_head'
    if 'timesnet' in lower:
        return 'timesnet'
    if 'cnn' in lower:
        return 'cnn'
    if 'mlp' in lower:
        return 'mlp'
    if 'gcn' in lower:
        return 'gcn'
    if 'gat' in lower:
        return 'gat'
    if 'gin' in lower:
        return 'gin'
    if 'graphsage' in lower:
        return 'graphsage'
    return 'cnn'


def load_sca_model(
    model_file,
    network_type,
    input_dim,
    device,
    graph_hidden_dim=128,
    graph_num_layers=3,
    graph_dropout=0.2,
    graph_heads=4,
    timesnet_config=None,
):
    check_file_exists(model_file)
    model = build_model(
        network_type,
        input_dim,
        graph_hidden_dim=graph_hidden_dim,
        graph_num_layers=graph_num_layers,
        graph_dropout=graph_dropout,
        graph_heads=graph_heads,
        timesnet_config=timesnet_config,
    )
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
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        sys.exit(-1)

    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
    Y_profiling = np.array(in_file['Profiling_traces/labels'])
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
    Y_attack = np.array(in_file['Attack_traces/labels'])
    if load_metadata is False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])


# Compute the rank of the real key for a given set of predictions.
def rank(predictions, metadata, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba, target_byte, simulated_key):
    if len(last_key_bytes_proba) == 0:
        key_bytes_proba = np.zeros(256)
    else:
        key_bytes_proba = last_key_bytes_proba

    for p in range(0, max_trace_idx - min_trace_idx):
        plaintext = metadata[min_trace_idx + p]['plaintext'][target_byte]
        key = metadata[min_trace_idx + p]['key'][target_byte]
        for i in range(0, 256):
            if simulated_key != 1:
                proba = predictions[p][AES_Sbox[plaintext ^ i]]
            else:
                proba = predictions[p][AES_Sbox[plaintext ^ key ^ i]]
            if proba != 0:
                key_bytes_proba[i] += np.log(proba)
            else:
                min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
                if len(min_proba_predictions) == 0:
                    print("Error: got a prediction with only zeroes ... this should not happen!")
                    sys.exit(-1)
                min_proba = min(min_proba_predictions)
                key_bytes_proba[i] += np.log(min_proba ** 2)

    sorted_proba = np.array(list(map(lambda a: key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
    return real_key_rank, key_bytes_proba


def full_ranks(predictions, dataset, metadata, min_trace_idx, max_trace_idx, rank_step, target_byte, simulated_key):
    print("Computing rank for targeted byte {}".format(target_byte))
    if simulated_key != 1:
        real_key = metadata[0]['key'][target_byte]
    else:
        real_key = 0

    if max_trace_idx > dataset.shape[0]:
        print("Error: asked trace index %d overflows the total traces number %d" % (max_trace_idx, dataset.shape[0]))
        sys.exit(-1)

    index = np.arange(min_trace_idx + rank_step, max_trace_idx, rank_step)
    f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
    key_bytes_proba = []
    for t, i in zip(index, range(0, len(index))):
        real_key_rank, key_bytes_proba = rank(
            predictions[t - rank_step:t],
            metadata,
            real_key,
            t - rank_step,
            t,
            key_bytes_proba,
            target_byte,
            simulated_key,
        )
        f_ranks[i] = [t - min_trace_idx, real_key_rank]
    return f_ranks


# Compute Pr(Sbox(p^k)*alpha|t)
def proba_dissect_beta(proba_sboxmuladd, proba_beta):
    proba = np.zeros(proba_sboxmuladd.shape)
    for j in range(proba_beta.shape[1]):
        proba_sboxdeadd = proba_sboxmuladd[:, [(beta ^ j) for beta in range(256)]]
        proba[:, j] = np.sum(proba_sboxdeadd * proba_beta, axis=1)
    return proba


# Compute Pr(Sbox(p^k)|t)
def proba_dissect_alpha(proba_sboxmul, proba_alpha):
    proba = np.zeros(proba_sboxmul.shape)
    for j in range(proba_alpha.shape[1]):
        proba_sboxdemul = proba_sboxmul[:, [multGF256(alpha, j) for alpha in range(256)]]
        proba[:, j] = np.sum(proba_sboxdemul * proba_alpha, axis=1)
    return proba


# Compute Pr(Sbox(p[permind]^k[permind])|t)
def proba_dissect_permind(proba_x, proba_permind, j):
    proba = np.zeros((proba_x.shape[0], proba_x.shape[2]))
    for s in range(proba_x.shape[2]):
        proba_1 = proba_x[:, :, s]
        proba_2 = proba_permind[:, :, j]
        proba[:, s] = np.sum(proba_1 * proba_2, axis=1)
    return proba


# Compute Pr(Sbox(p^k)|t) by recombination of guessed probabilities.
def multilabel_predict(predictions):
    predictions_alpha = predictions[0]
    predictions_beta = predictions[1]
    predictions_unshuffledsboxmuladd = []
    predictions_permind = []
    for i in range(16):
        predictions_unshuffledsboxmuladd.append(predictions[2 + i])
        predictions_permind.append(predictions[2 + 16 + i])

    predictions_unshuffledsboxmul = []
    print("Computing multiplicative masked sbox probas with shuffle...")
    for i in range(16):
        predictions_unshuffledsboxmul.append(proba_dissect_beta(predictions_unshuffledsboxmuladd[i], predictions_beta))

    print("Computing sbox probas with shuffle...")
    predictions_unshuffledsbox = []
    for i in range(16):
        predictions_unshuffledsbox.append(proba_dissect_alpha(predictions_unshuffledsboxmul[i], predictions_alpha))

    predictions_unshuffledsbox_v = np.array(predictions_unshuffledsbox)
    predictions_permind_v = np.array(predictions_permind)
    predictions_unshuffledsbox_v = np.moveaxis(predictions_unshuffledsbox_v, [0, 1, 2], [1, 0, 2])
    predictions_permind_v = np.moveaxis(predictions_permind_v, [0, 1, 2], [1, 0, 2])
    predictions_sbox = []
    print("Computing sbox probas...")
    for i in range(16):
        predictions_sbox.append(proba_dissect_permind(predictions_unshuffledsbox_v, predictions_permind_v, i))

    return predictions_sbox


# Compute Pr(Sbox(p^k)|t) by recombination without shuffling permutation.
def multilabel_without_permind_predict(predictions):
    predictions_alpha = predictions[0]
    predictions_beta = predictions[1]
    predictions_sboxmuladd = []
    for i in range(16):
        predictions_sboxmuladd.append(predictions[2 + i])

    predictions_sboxmul = []
    print("Computing multiplicative masked sbox...")
    for i in range(16):
        predictions_sboxmul.append(proba_dissect_beta(predictions_sboxmuladd[i], predictions_beta))

    print("Computing sbox probas...")
    predictions_sbox = []
    for i in range(16):
        predictions_sbox.append(proba_dissect_alpha(predictions_sboxmul[i], predictions_alpha))

    return predictions_sbox


def prepare_input_tensor(X_attack, num_traces, network_type, device):
    input_data = X_attack[:num_traces, :].astype(np.float32)
    tensor = torch.from_numpy(input_data)
    if is_graph_network(network_type):
        if tensor.shape[1] != 700:
            print("Error: graph mode baseline expects 700 nodes per trace, got %d." % tensor.shape[1])
            sys.exit(-1)
        graph_batch = build_graph_batch_from_traces([tensor[i] for i in range(tensor.shape[0])], None)
        return graph_batch.to(device)
    if is_timesnet_network(network_type):
        return tensor.unsqueeze(-1).to(device)  # (B, L) -> (B, L, 1)
    if network_type != 'mlp':
        tensor = tensor.unsqueeze(1)
    return tensor.to(device)


def to_prediction_array(model_output, multilabel):
    if multilabel == 0:
        probs = torch.softmax(model_output, dim=1)
        return probs.detach().cpu().numpy()

    if multilabel == 1:
        keys = ['alpha_output', 'beta_output'] + [f'sbox_{i}_output' for i in range(16)] + [f'permind_{i}_output' for i in range(16)]
    else:
        keys = ['alpha_output', 'beta_output'] + [f'sbox_{i}_output' for i in range(16)]

    preds = []
    for k in keys:
        preds.append(torch.softmax(model_output[k], dim=1).detach().cpu().numpy())
    return preds


# Check a saved model against one of the ASCAD databases Attack traces.
def check_model(
    model_file,
    ascad_database,
    num_traces=2000,
    target_byte=2,
    multilabel=0,
    simulated_key=0,
    save_file="",
    network_type=None,
    graph_hidden_dim=128,
    graph_num_layers=3,
    graph_dropout=0.2,
    graph_heads=4,
    timesnet_config=None,
):
    check_file_exists(model_file)
    check_file_exists(ascad_database)

    (_, _), (X_attack, _), (_, Metadata_attack) = load_ascad(ascad_database, load_metadata=True)

    if network_type is None:
        network_type = infer_network_type_from_filename(model_file, multilabel)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_sca_model(
        model_file,
        network_type,
        len(X_attack[0, :]),
        device,
        graph_hidden_dim=graph_hidden_dim,
        graph_num_layers=graph_num_layers,
        graph_dropout=graph_dropout,
        graph_heads=graph_heads,
        timesnet_config=timesnet_config,
    )

    input_tensor = prepare_input_tensor(X_attack, num_traces, network_type, device)

    with torch.no_grad():
        model_output = model(input_tensor)
    predictions = to_prediction_array(model_output, multilabel)

    if multilabel != 0:
        if multilabel == 1:
            predictions_sbox = multilabel_predict(predictions)
        else:
            predictions_sbox = multilabel_without_permind_predict(predictions)

        for byte_idx in range(16):
            ranks_i = full_ranks(predictions_sbox[byte_idx], X_attack, Metadata_attack, 0, num_traces, 10, byte_idx, simulated_key)
            x_i = [ranks_i[i][0] for i in range(0, ranks_i.shape[0])]
            y_i = [ranks_i[i][1] for i in range(0, ranks_i.shape[0])]
            plt.plot(x_i, y_i, label="key_" + str(byte_idx))

        plt.title('Performance of ' + model_file + ' against ' + ascad_database)
        plt.xlabel('number of traces')
        plt.ylabel('rank')
        plt.grid(True)
        plt.legend(loc='upper right')
        if save_file != "":
            plt.savefig(save_file)
        else:
            plt.show(block=False)
    else:
        ranks = full_ranks(predictions, X_attack, Metadata_attack, 0, num_traces, 10, target_byte, simulated_key)
        x = [ranks[i][0] for i in range(0, ranks.shape[0])]
        y = [ranks[i][1] for i in range(0, ranks.shape[0])]
        plt.title('Performance of ' + model_file + ' against ' + ascad_database)
        plt.xlabel('number of traces')
        plt.ylabel('rank')
        plt.grid(True)
        plt.plot(x, y)
        if save_file != "":
            plt.savefig(save_file)
        else:
            plt.show(block=False)


def read_parameters_from_file(param_filename):
    # TODO: sanity checks on parameters
    param_file = open(param_filename, "r")

    # FIXME: replace eval() by ast.literal_eval()
    my_parameters = eval(param_file.read())

    model_file = my_parameters["model_file"]
    ascad_database = my_parameters["ascad_database"]
    num_traces = my_parameters["num_traces"]
    target_byte = 2
    if "target_byte" in my_parameters:
        target_byte = my_parameters["target_byte"]
    multilabel = 0
    if "multilabel" in my_parameters:
        multilabel = my_parameters["multilabel"]
    simulated_key = 0
    if "simulated_key" in my_parameters:
        simulated_key = my_parameters["simulated_key"]
    save_file = ""
    if "save_file" in my_parameters:
        save_file = my_parameters["save_file"]
    network_type = None
    if "network_type" in my_parameters:
        network_type = my_parameters["network_type"]
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
        model_file,
        ascad_database,
        num_traces,
        target_byte,
        multilabel,
        simulated_key,
        save_file,
        network_type,
        graph_hidden_dim,
        graph_num_layers,
        graph_dropout,
        graph_heads,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        model_file = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/cnn_best_ascad_desync0_epochs75_classes256_batchsize200.pth"
        ascad_database = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
        num_traces = 2000
        target_byte = 2
        multilabel = 0
        simulated_key = 0
        save_file = ""
        network_type = None
        graph_hidden_dim = 128
        graph_num_layers = 3
        graph_dropout = 0.2
        graph_heads = 4
    else:
        (
            model_file,
            ascad_database,
            num_traces,
            target_byte,
            multilabel,
            simulated_key,
            save_file,
            network_type,
            graph_hidden_dim,
            graph_num_layers,
            graph_dropout,
            graph_heads,
        ) = read_parameters_from_file(sys.argv[1])

    check_model(
        model_file,
        ascad_database,
        num_traces,
        target_byte,
        multilabel,
        simulated_key,
        save_file,
        network_type,
        graph_hidden_dim,
        graph_num_layers,
        graph_dropout,
        graph_heads,
    )

    try:
        input("Press enter to exit ...")
    except SyntaxError:
        pass
