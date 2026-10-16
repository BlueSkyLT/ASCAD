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

from GNN.classifiers import (
    GATClassifier,
    GCNClassifier,
    GINClassifier,
    GraphSAGEClassifier,
    SegmentGraphSAGEClassifier,
)
from GNN.utils import ensure_graph_dependencies, is_graph_network, is_segment_graph_network
from GNN.segment_graph import (
    DEFAULT_NUM_NODES,
    DEFAULT_OUT_DIM,
    DEFAULT_SEGMENTER,
    DEFAULT_THRESHOLD,
    cache_path_for,
    load_or_build_segment_features,
    standardize_segment_features,
)
# Note: build_graph_batch_from_traces (in GNN.data) is retained only for backward compat.
# Sample-node graphs pass (B, 700); segment-node graphs pass (B, N, F) + mask.

from ASCAD_train_models import (
    MLPBest,
    CNNBest,
    MultiResNet,
    MultiTest,
    is_timesnet_network,
    is_timesnet_multilabel_network,
    build_timesnet_for_network,
    strip_torch_compile_prefix,
)
from TimesNet import build_timesnet_model
from leakage_localizer.model import build_poi_searcher, is_poi_searcher_network

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


def build_model(
    network_type,
    input_dim,
    graph_hidden_dim=128,
    graph_num_layers=2,
    graph_dropout=0.2,
    graph_heads=4,
    timesnet_config=None,
    multilabel=0,
    segment_config=None,
):
    if is_timesnet_network(network_type):
        cfg_kwargs = dict(timesnet_config) if timesnet_config else {}
        cfg_kwargs.setdefault('seq_len', input_dim)
        if is_timesnet_multilabel_network(network_type):
            # multilabel=2 => without_permind heads; multilabel=1 => with permind
            cfg_kwargs['without_permind'] = (multilabel == 2)
        return build_timesnet_for_network(network_type, input_dim, timesnet_config=cfg_kwargs)
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
    if network_type == 'seg_graphsage':
        seg = dict(segment_config or {})
        feat_dim = int(seg.get("out_dim", input_dim if input_dim and input_dim != 700 else DEFAULT_OUT_DIM))
        num_nodes = int(seg.get("num_nodes", DEFAULT_NUM_NODES))
        return SegmentGraphSAGEClassifier(
            input_dim=feat_dim,
            hidden_dim=graph_hidden_dim,
            num_layers=graph_num_layers,
            dropout=graph_dropout,
            num_nodes=num_nodes,
            edge_mode=seg.get("edge_mode", "path"),
        )
    if is_poi_searcher_network(network_type):
        poi_cfg = None
        if timesnet_config and isinstance(timesnet_config, dict) and timesnet_config.get("poi_config"):
            poi_cfg = timesnet_config.get("poi_config")
        return build_poi_searcher(input_dim, poi_cfg)
    print("Error: no topology found for network '%s' ..." % network_type)
    sys.exit(-1)


def infer_network_type_from_filename(model_file, multilabel):
    lower = os.path.basename(model_file).lower()
    if 'timesnet_ascad_head_multi' in lower:
        return 'timesnet_ascad_head_multi'
    if 'timesnet_multi' in lower:
        return 'timesnet_multi'
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
    if 'poi_searcher' in lower:
        return 'poi_searcher'
    if 'seg_graphsage' in lower:
        return 'seg_graphsage'
    if 'gcn' in lower:
        return 'gcn'
    if 'gat' in lower:
        return 'gat'
    if 'gin' in lower:
        return 'gin'
    if 'graphsage' in lower:
        return 'graphsage'
    return 'cnn'


def load_timesnet_config_sidecar(model_file):
    """Load optional {model}.config.json written at train time."""
    path = model_file + ".config.json"
    if not os.path.exists(path):
        return None
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Warning: could not read sidecar {path}: {exc}")
        return None


def load_sca_model(
    model_file,
    network_type,
    input_dim,
    device,
    graph_hidden_dim=128,
    graph_num_layers=2,
    graph_dropout=0.2,
    graph_heads=4,
    timesnet_config=None,
    multilabel=0,
    segment_config=None,
):
    check_file_exists(model_file)
    sidecar = load_timesnet_config_sidecar(model_file)
    if sidecar:
        if timesnet_config is None and sidecar.get("timesnet_config"):
            timesnet_config = sidecar["timesnet_config"]
        if segment_config is None and sidecar.get("segment_config"):
            segment_config = sidecar["segment_config"]
        if multilabel == 0 and sidecar.get("multilabel") is not None:
            multilabel = int(sidecar["multilabel"])
        if sidecar.get("network_type") and network_type in (None,):
            network_type = sidecar["network_type"]
        if sidecar.get("input_dim") is not None and is_segment_graph_network(network_type or ""):
            input_dim = int(sidecar["input_dim"])
        if sidecar.get("network_type") and is_poi_searcher_network(sidecar.get("network_type")):
            network_type = sidecar["network_type"]
        if sidecar.get("input_dim") is not None and is_poi_searcher_network(network_type or ""):
            input_dim = int(sidecar["input_dim"])
    if is_poi_searcher_network(network_type):
        poi_cfg = (sidecar or {}).get("poi_config")
        model = build_poi_searcher(input_dim, poi_cfg)
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
    model = build_model(
        network_type,
        input_dim,
        graph_hidden_dim=graph_hidden_dim,
        graph_num_layers=graph_num_layers,
        graph_dropout=graph_dropout,
        graph_heads=graph_heads,
        timesnet_config=timesnet_config,
        multilabel=multilabel,
        segment_config=segment_config,
    )
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
#
# tie_break controls what happens when other key hypotheses score exactly equal to the true
# key. "worst" (default) reports the LAST position among equals, so a completely flat score
# vector returns 255 rather than 0. "best" reproduces the historical behaviour of this
# function (`np.where(sorted == true)[0][0]`, the FIRST position among equals) and is kept
# for comparing against published numbers, which assume it.
#
# This was changed because a tie is not evidence: with the old convention, replacing every
# head of a model with a uniform distribution reported rank 0 on all 16 bytes, i.e. total
# ignorance scored as a perfect break. See docs/tasks/T028 §"Phase 0".
def rank(predictions, metadata, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba,
         target_byte, simulated_key, tie_break="worst"):
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

    real_key_score = key_bytes_proba[real_key]
    strictly_better = int(np.sum(key_bytes_proba > real_key_score))
    tied_with_real = int(np.sum(key_bytes_proba == real_key_score))
    if tie_break == "best":
        real_key_rank = strictly_better
    elif tie_break == "worst":
        # last position among equals: ties are charged against us
        real_key_rank = strictly_better + max(tied_with_real - 1, 0)
    else:
        raise ValueError(f"tie_break must be 'best' or 'worst', got {tie_break!r}")
    return real_key_rank, key_bytes_proba


def full_ranks(predictions, dataset, metadata, min_trace_idx, max_trace_idx, rank_step,
               target_byte, simulated_key, tie_break="worst"):
    print("Computing rank for targeted byte {} (tie_break={})".format(target_byte, tie_break))
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
            tie_break,
        )
        f_ranks[i] = [t - min_trace_idx, real_key_rank]
    return f_ranks


def _extract_attack_plaintexts_and_key(metadata, num_traces, target_byte, simulated_key=0):
    """Pull plaintext bytes and the reference key byte from ASCAD metadata.

    Supports h5py datasets and numpy structured arrays.
    """
    if num_traces <= 0:
        raise ValueError("num_traces must be positive")

    # Prefer bulk slice (fast for h5py / structured arrays).
    try:
        meta_slice = metadata[:num_traces]
        plaintexts = np.asarray(meta_slice["plaintext"][:, target_byte], dtype=np.uint8)
        if simulated_key == 1:
            real_key = 0
        else:
            real_key = int(np.asarray(meta_slice["key"][0, target_byte]))
        return plaintexts, real_key
    except Exception:
        plaintexts = np.empty(num_traces, dtype=np.uint8)
        for i in range(num_traces):
            plaintexts[i] = int(metadata[i]["plaintext"][target_byte])
        if simulated_key == 1:
            real_key = 0
        else:
            real_key = int(metadata[0]["key"][target_byte])
        return plaintexts, real_key


def log_key_scores_from_predictions(predictions, plaintexts, simulated_key=0, keys=None):
    """Per-trace log-likelihood of each key-byte hypothesis.

    Parameters
    ----------
    predictions : (N, 256) array
        Model probabilities over S-box outputs (or equivalent labels).
    plaintexts : (N,) uint8
        Target-byte plaintexts for each attack trace.
    simulated_key : int
        If 1, use ASCAD simulated constant-key rewriting (variable-key sets).
    keys : (N,) uint8 or None
        Per-trace true key bytes; required only when simulated_key == 1.

    Returns
    -------
    log_scores : (N, 256) float64
        log_scores[t, k] = log Pr(label | hyp key k) for trace t.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    plaintexts = np.asarray(plaintexts, dtype=np.uint8)
    if predictions.ndim != 2 or predictions.shape[1] != 256:
        raise ValueError(f"predictions must have shape (N, 256), got {predictions.shape}")
    if plaintexts.shape[0] != predictions.shape[0]:
        raise ValueError("plaintexts length must match predictions")

    n = predictions.shape[0]
    hyps = np.arange(256, dtype=np.uint8)[None, :]  # (1, 256)
    pt = plaintexts[:, None]  # (N, 1)

    if simulated_key == 1:
        if keys is None:
            raise ValueError("keys is required when simulated_key == 1")
        keys = np.asarray(keys, dtype=np.uint8)[:, None]
        # Same rewriting as rank(): AES_Sbox[plaintext ^ key ^ i]
        sbox_idx = AES_Sbox[np.bitwise_xor(np.bitwise_xor(pt, keys), hyps)]
    else:
        sbox_idx = AES_Sbox[np.bitwise_xor(pt, hyps)]

    # probs[t, k] = predictions[t, sbox_idx[t, k]]
    probs = predictions[np.arange(n)[:, None], sbox_idx]
    # Softmax outputs should be > 0; clip for numerical safety (matches common GE practice).
    return np.log(np.clip(probs, 1e-45, 1.0))


def ranks_from_log_scores(log_scores, real_key, rank_step, tie_break="worst"):
    """Cumulative key rank curve for one ordered attack sequence.

    Parameters
    ----------
    log_scores : (N, 256)
        Per-trace log scores already ordered as the attack will use them.
    real_key : int
        True key byte (0..255).
    rank_step : int
        Evaluate rank every ``rank_step`` traces.
    tie_break : str
        ``worst`` (default) charges tied hypotheses against us, matching ``rank()``;
        ``best`` reproduces this function's historical first-among-equals convention,
        which reports rank 0 for a perfectly flat score vector.

    Returns
    -------
    ranks : (n_steps, 2) uint32
        ranks[i] = [trace_count, rank] with 0-based rank (0 = best).
    """
    log_scores = np.asarray(log_scores, dtype=np.float64)
    n = log_scores.shape[0]
    if n <= 0:
        return np.zeros((0, 2), dtype=np.uint32)
    if rank_step <= 0:
        raise ValueError("rank_step must be positive")

    cum = np.cumsum(log_scores, axis=0)
    # Same step grid as full_ranks: rank_step, 2*rank_step, ... (< n if n not multiple)
    # full_ranks uses arange(min + step, max, step) with values t - min = step, 2step, ...
    # and max is exclusive end index, so last multi-step below max.
    # Additionally many callers want the endpoint; full_ranks may miss exact N if N % step != 0
    # or if max is exclusive. Mirror full_ranks for the grid, then always include N if missing.
    steps = list(range(rank_step, n, rank_step))
    if n not in steps:
        steps.append(n)
    if not steps:
        steps = [n]

    out = np.zeros((len(steps), 2), dtype=np.uint32)
    real_key = int(real_key)
    for i, n_tr in enumerate(steps):
        scores = cum[n_tr - 1]
        true_score = scores[real_key]
        strictly_better = int(np.sum(scores > true_score))
        if tie_break == "best":
            rank = strictly_better
        elif tie_break == "worst":
            rank = strictly_better + int(np.sum(scores == true_score)) - 1
        else:
            raise ValueError(f"tie_break must be 'best' or 'worst', got {tie_break!r}")
        out[i, 0] = n_tr
        out[i, 1] = rank
    return out


def guessing_entropy(
    predictions,
    metadata,
    num_traces,
    rank_step,
    target_byte,
    n_attacks=100,
    rng=None,
    simulated_key=0,
    return_curve=False,
    ge_report_points=None,
):
    """Guessing entropy via multi-attack averaging of key rank.

    Classical GE(n) = E[rank of correct key after n attack traces], estimated by
    averaging over ``n_attacks`` independent random orderings (permutations without
    replacement) of the attack set.

    Notes
    -----
    - This is **not** the same as averaging ``rank_final`` over training seeds.
    - Final scores after using the *same full multiset* of traces are
      **order-invariant** (log-likelihood sums commute). So ``ge_final`` at
      ``num_traces`` equals the single full-set key rank; multi-attack averaging
      mainly affects the **curve** GE(n) for n < num_traces, the distribution of
      traces-to-rank-0, and intermediate success rates.
    - ``ge_traces_to_0`` is the smallest n with GE(n) == 0 (all attacks rank-0).
    - ``mean_traces_to_rank0`` averages each attack's first rank-0 time (MTD).

    Parameters
    ----------
    predictions : (N, 256) array
        Model probabilities for the first ``num_traces`` attack traces (or more).
    metadata : ASCAD attack metadata (h5py dataset or structured array)
    num_traces : int
        Number of attack traces to use (N).
    rank_step : int
        Step for the GE curve.
    target_byte : int
        Key / plaintext byte index.
    n_attacks : int
        Number of independent random attack orderings. Must be >= 1.
    rng : numpy Generator, int, or None
        RNG for permutations. If None, uses numpy default_rng().
    simulated_key : int
        Passed through for variable-key simulated ranking.
    return_curve : bool
        If True, also return the full GE curve array.
    ge_report_points : sequence of int or None
        Trace counts at which to snapshot GE into ``ge_at``. Defaults to a
        sparse set up to ``num_traces``.

    Returns
    -------
    result : dict
        ge_final, ge_traces_to_0, mean_traces_to_rank0, median_traces_to_rank0,
        frac_reached_rank0, success_rate_final, n_attacks, ge_at, ...
    """
    if n_attacks is None or int(n_attacks) <= 0:
        raise ValueError("n_attacks must be >= 1")
    n_attacks = int(n_attacks)
    num_traces = int(num_traces)
    rank_step = int(rank_step)

    predictions = np.asarray(predictions[:num_traces], dtype=np.float64)
    if predictions.shape[0] < num_traces:
        raise ValueError(
            f"predictions has {predictions.shape[0]} rows but num_traces={num_traces}"
        )

    plaintexts, real_key = _extract_attack_plaintexts_and_key(
        metadata, num_traces, target_byte, simulated_key=simulated_key
    )
    keys = None
    if simulated_key == 1:
        try:
            meta_slice = metadata[:num_traces]
            keys = np.asarray(meta_slice["key"][:, target_byte], dtype=np.uint8)
        except Exception:
            keys = np.array(
                [int(metadata[i]["key"][target_byte]) for i in range(num_traces)],
                dtype=np.uint8,
            )

    log_scores = log_key_scores_from_predictions(
        predictions, plaintexts, simulated_key=simulated_key, keys=keys
    )

    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))

    # Probe step grid from identity order (same grid for every permutation length).
    probe = ranks_from_log_scores(log_scores, real_key, rank_step)
    n_steps = probe.shape[0]
    trace_counts = probe[:, 0].astype(np.int64)
    rank_sum = np.zeros(n_steps, dtype=np.float64)
    success_final = 0
    mtd_list = []

    for _ in range(n_attacks):
        order = rng.permutation(num_traces)
        ranks = ranks_from_log_scores(log_scores[order], real_key, rank_step)
        rank_sum += ranks[:, 1].astype(np.float64)
        if int(ranks[-1, 1]) == 0:
            success_final += 1
        mtd = None
        for n_tr, rv in ranks:
            if int(rv) == 0:
                mtd = int(n_tr)
                break
        mtd_list.append(mtd)

    ge_curve_vals = rank_sum / float(n_attacks)
    ge_final = float(ge_curve_vals[-1])

    ge_traces_to_0 = None
    for n_tr, ge_v in zip(trace_counts, ge_curve_vals):
        # Exact 0 only if every attack had rank 0 at that point.
        if ge_v == 0.0:
            ge_traces_to_0 = int(n_tr)
            break

    mtds_reached = [m for m in mtd_list if m is not None]
    if mtds_reached:
        mean_traces_to_rank0 = float(np.mean(mtds_reached))
        median_traces_to_rank0 = float(np.median(mtds_reached))
    else:
        mean_traces_to_rank0 = None
        median_traces_to_rank0 = None
    frac_reached_rank0 = float(len(mtds_reached)) / float(n_attacks)

    if ge_report_points is None:
        candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, num_traces]
        ge_report_points = sorted({int(p) for p in candidates if 0 < int(p) <= num_traces})

    # Map requested points to nearest available step >= p (or last).
    ge_at = {}
    for p in ge_report_points:
        p = int(p)
        idx = None
        for j, n_tr in enumerate(trace_counts):
            if int(n_tr) >= p:
                idx = j
                break
        if idx is None:
            idx = len(trace_counts) - 1
        ge_at[str(int(trace_counts[idx]))] = float(ge_curve_vals[idx])

    result = {
        "ge_final": ge_final,
        "ge_traces_to_0": ge_traces_to_0,
        "mean_traces_to_rank0": mean_traces_to_rank0,
        "median_traces_to_rank0": median_traces_to_rank0,
        "frac_reached_rank0": frac_reached_rank0,
        "success_rate_final": float(success_final) / float(n_attacks),
        "n_attacks": n_attacks,
        "real_key": int(real_key),
        "target_byte": int(target_byte),
        "ge_at": ge_at,
    }
    if return_curve:
        curve = np.zeros((n_steps, 2), dtype=np.float64)
        curve[:, 0] = trace_counts
        curve[:, 1] = ge_curve_vals
        result["ge_curve"] = curve
    return result


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
    # T-041: route with exact permutation marginals (ASCAD_PI_ROUTING=raw restores the old routing)
    from perm_routing import apply_pi_routing
    predictions_permind_v = apply_pi_routing(predictions_permind_v)
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
        # Fast path: classifiers now accept raw (B, 700) tensor directly via dense
        # complete-graph implementation (avoids O(N^2) edges, high VRAM, CPU bottleneck).
        return tensor.to(device)
    if is_timesnet_network(network_type):
        return tensor.unsqueeze(-1).to(device)  # (B, L) -> (B, L, 1)
    if is_poi_searcher_network(network_type) or network_type == 'mlp':
        return tensor.to(device)
    if network_type != 'mlp':
        tensor = tensor.unsqueeze(1)
    return tensor.to(device)


def to_prediction_array(model_output, multilabel):
    if isinstance(model_output, dict) and "p_s" in model_output:
        return model_output["p_s"].detach().cpu().numpy()
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


def _batched_predict(model, X_attack, num_traces, network_type, device, batch_size, multilabel):
    """Perform model inference in chunks for memory-heavy networks (TimesNet, graphs).
    Returns the same format as to_prediction_array (probs array or list of arrays).
    """
    if batch_size <= 0:
        batch_size = num_traces

    if multilabel != 0:
        # Multilabel: accumulate per-head softmax probs across batches.
        if multilabel == 1:
            keys = ['alpha_output', 'beta_output'] + [f'sbox_{i}_output' for i in range(16)] + [f'permind_{i}_output' for i in range(16)]
        else:
            keys = ['alpha_output', 'beta_output'] + [f'sbox_{i}_output' for i in range(16)]
        head_chunks = [[] for _ in keys]
        for start in range(0, num_traces, batch_size):
            stop = min(num_traces, start + batch_size)
            sub = X_attack[start:stop].astype(np.float32)
            t = torch.from_numpy(sub)
            if is_timesnet_network(network_type):
                t = t.unsqueeze(-1)
            elif network_type != 'mlp' and not is_graph_network(network_type):
                t = t.unsqueeze(1)
            t = t.to(device)
            with torch.no_grad():
                model_output = model(t)
            for i, k in enumerate(keys):
                head_chunks[i].append(torch.softmax(model_output[k], dim=1).detach().cpu().numpy())
        return [np.concatenate(chunks, axis=0) for chunks in head_chunks]

    all_probs = []
    for start in range(0, num_traces, batch_size):
        stop = min(num_traces, start + batch_size)
        sub = X_attack[start:stop]
        if is_segment_graph_network(network_type):
            # X_attack already converted to (n, N, F); optional parallel mask in attr
            t = torch.from_numpy(sub.astype(np.float32)).to(device)
            m = None
            if getattr(_batched_predict, "_seg_mask", None) is not None:
                m = torch.from_numpy(
                    _batched_predict._seg_mask[start:stop].astype(np.bool_)
                ).to(device)
            with torch.no_grad():
                logits = model(t, m) if m is not None else model(t)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                all_probs.append(probs)
        elif is_graph_network(network_type):
            if sub.ndim != 2 or sub.shape[1] != 700:
                print("Error: sample-graph mode expects (B, 700), got %s." % (sub.shape,))
                sys.exit(-1)
            # Fast path: direct tensor, no graph construction.
            t = torch.from_numpy(sub.astype(np.float32)).to(device)
            with torch.no_grad():
                logits = model(t)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                all_probs.append(probs)
        elif is_timesnet_network(network_type):
            t = torch.from_numpy(sub.astype(np.float32)).unsqueeze(-1).to(device)
            padding_mask = torch.ones(t.shape[0], t.shape[1], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = model(t, padding_mask)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                all_probs.append(probs)
        elif is_poi_searcher_network(network_type):
            t = torch.from_numpy(sub.astype(np.float32)).to(device)
            with torch.no_grad():
                out = model(t)
                all_probs.append(out["p_s"].detach().cpu().numpy())
        else:
            # fallback for other nets (rare)
            t = torch.from_numpy(sub.astype(np.float32))
            if network_type != 'mlp':
                t = t.unsqueeze(1)
            t = t.to(device)
            with torch.no_grad():
                logits = model(t)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                all_probs.append(probs)
    if len(all_probs) == 0:
        return np.zeros((0, 256), dtype=np.float32)
    return np.concatenate(all_probs, axis=0)


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
    timesnet_eval_batch_size=256,
    graph_eval_batch_size=8,
    segment_config=None,
):
    check_file_exists(model_file)
    check_file_exists(ascad_database)

    (_, _), (X_attack, _), (_, Metadata_attack) = load_ascad(ascad_database, load_metadata=True)

    if network_type is None:
        network_type = infer_network_type_from_filename(model_file, multilabel)

    # Sidecar may carry segment_config / input_dim for segment graphs.
    sidecar = load_timesnet_config_sidecar(model_file)
    if sidecar and segment_config is None and sidecar.get("segment_config"):
        segment_config = sidecar["segment_config"]

    if sidecar and sidecar.get("sub_avg"):
        mean_path = sidecar.get("mean_wf_path") or (model_file + ".mean_wf.npy")
        if os.path.exists(mean_path):
            mean_wf = np.load(mean_path)
            print(f"Applying average waveform subtraction to attack traces (from {mean_path})...")
            X_attack = (X_attack.astype(np.float32) - mean_wf)
        else:
            print(f"Warning: sub_avg is True in sidecar, but mean waveform file not found at {mean_path}")

    if sidecar and sidecar.get("target_byte") is not None:
        target_byte = int(sidecar["target_byte"])
        print(f"Loaded target_byte={target_byte} from sidecar")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_input_dim = len(X_attack[0, :])
    if is_segment_graph_network(network_type):
        seg = dict(segment_config or {})
        seg_n = int(seg.get("num_nodes", DEFAULT_NUM_NODES))
        seg_f = int(seg.get("out_dim", DEFAULT_OUT_DIM))
        seg_th = int(seg.get("threshold", DEFAULT_THRESHOLD))
        seg_kind = seg.get("segmenter", DEFAULT_SEGMENTER)
        seg_center = bool(seg.get("center", False))
        cache_dir = seg.get("cache_dir") or os.path.join(
            os.path.dirname(model_file) or ".", "segment_feature_cache"
        )
        snr_extra = ""
        fixed_windows = seg.get("fixed_windows")
        if seg_kind == "snr":
            if not fixed_windows:
                print("Error: segmenter=snr requires fixed_windows in segment_config/sidecar (profiling peaks).")
                sys.exit(-1)
            half_w = int(seg.get("snr_half_width", 12))
            peaks = seg.get("snr_peaks", [])
            snr_extra = "snr_p%s_w%d" % ("-".join(str(p) for p in peaks), half_w)
            fixed_windows = [(int(s), int(e)) for s, e in fixed_windows]
        cpath = cache_path_for(
            cache_dir,
            ascad_database,
            split="attack",
            num_nodes=seg_n,
            out_dim=seg_f,
            segmenter=seg_kind,
            threshold=seg_th,
            center=seg_center,
            n_traces=X_attack.shape[0],
            trace_len=X_attack.shape[1],
            extra=snr_extra,
        )
        seg_std = bool(seg.get("standardize", True))
        stats = None
        if seg_std and seg.get("feature_mu") is not None and seg.get("feature_std") is not None:
            stats = (
                np.asarray(seg["feature_mu"], dtype=np.float32),
                np.asarray(seg["feature_std"], dtype=np.float32),
            )
        build_kwargs = {}
        if seg_kind == "snr":
            build_kwargs["fixed_windows"] = fixed_windows
        X_attack_seg, attack_mask = load_or_build_segment_features(
            X_attack,
            cache_path=cpath,
            num_nodes=seg_n,
            out_dim=seg_f,
            segmenter=seg_kind,
            threshold=seg_th,
            center=seg_center,
            standardize=False,  # apply profiling stats below if needed
            **build_kwargs,
        )
        if seg_std:
            if stats is None:
                # Fall back to attack-set stats (suboptimal but avoids crash)
                X_attack_seg, _, _ = standardize_segment_features(X_attack_seg, attack_mask)
            else:
                X_attack_seg, _, _ = standardize_segment_features(
                    X_attack_seg, attack_mask, mu=stats[0], std=stats[1]
                )
        X_attack_for_pred = X_attack_seg
        _batched_predict._seg_mask = attack_mask
        model_input_dim = seg_f
    else:
        X_attack_for_pred = X_attack
        _batched_predict._seg_mask = None

    model = load_sca_model(
        model_file,
        network_type,
        model_input_dim,
        device,
        graph_hidden_dim=graph_hidden_dim,
        graph_num_layers=graph_num_layers,
        graph_dropout=graph_dropout,
        graph_heads=graph_heads,
        timesnet_config=timesnet_config,
        multilabel=multilabel,
        segment_config=segment_config,
    )

    use_batched = (
        is_timesnet_network(network_type)
        or is_graph_network(network_type)
        or is_poi_searcher_network(network_type)
        or multilabel != 0
    )
    if use_batched:
        if is_timesnet_network(network_type):
            bs = timesnet_eval_batch_size if timesnet_eval_batch_size and timesnet_eval_batch_size > 0 else 8
        elif is_graph_network(network_type):
            bs = graph_eval_batch_size if graph_eval_batch_size and graph_eval_batch_size > 0 else num_traces
            if is_segment_graph_network(network_type) and (not bs or bs < 32):
                bs = max(bs or 0, 256)
        else:
            # Multilabel multi-ResNet etc.: still safer to micro-batch on long traces.
            bs = timesnet_eval_batch_size if timesnet_eval_batch_size and timesnet_eval_batch_size > 0 else 32
        predictions = _batched_predict(
            model, X_attack_for_pred, num_traces, network_type, device, bs, multilabel
        )
    else:
        input_tensor = prepare_input_tensor(X_attack_for_pred, num_traces, network_type, device)
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
    graph_num_layers = 2
    if "graph_num_layers" in my_parameters:
        graph_num_layers = my_parameters["graph_num_layers"]
    graph_dropout = 0.2
    if "graph_dropout" in my_parameters:
        graph_dropout = my_parameters["graph_dropout"]
    graph_heads = 4
    if "graph_heads" in my_parameters:
        graph_heads = my_parameters["graph_heads"]
    timesnet_config = None
    if "timesnet_config" in my_parameters:
        timesnet_config = my_parameters["timesnet_config"]
    timesnet_eval_batch_size = 256
    if "timesnet_eval_batch_size" in my_parameters:
        timesnet_eval_batch_size = my_parameters["timesnet_eval_batch_size"]
    graph_eval_batch_size = 8
    if "graph_eval_batch_size" in my_parameters:
        graph_eval_batch_size = my_parameters["graph_eval_batch_size"]
    segment_config = None
    if "segment_config" in my_parameters:
        segment_config = my_parameters["segment_config"]

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
        timesnet_config,
        timesnet_eval_batch_size,
        graph_eval_batch_size,
        segment_config,
    )


if __name__ == "__main__":
    if len(sys.argv) == 2:
        with open(sys.argv[1], "r", encoding="utf-8") as _pf:
            _raw_params = eval(_pf.read())
        _nt = _raw_params.get("network_type")
        _mf = str(_raw_params.get("model_file", ""))
        if is_poi_searcher_network(_nt) or "poi_searcher" in _mf.lower():
            from leakage_localizer.eval_ge import eval_from_params

            eval_from_params(_raw_params)
            sys.exit(0)
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
        graph_num_layers = 2
        graph_dropout = 0.2
        graph_heads = 4
        timesnet_config = None
        timesnet_eval_batch_size = 256
        graph_eval_batch_size = 8
        segment_config = None
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
            timesnet_config,
            timesnet_eval_batch_size,
            graph_eval_batch_size,
            segment_config,
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
        timesnet_config=timesnet_config,
        timesnet_eval_batch_size=timesnet_eval_batch_size,
        graph_eval_batch_size=graph_eval_batch_size,
        segment_config=segment_config,
    )

    # Non-interactive / nohup-safe exit (no stdin prompt).
    if sys.stdin is not None and sys.stdin.isatty():
        try:
            input("Press enter to exit ...")
        except (EOFError, SyntaxError):
            pass
