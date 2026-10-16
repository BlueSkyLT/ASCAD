import sys

try:
    from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    GATConv = None
    GCNConv = None
    GINConv = None
    SAGEConv = None
    global_mean_pool = None

GRAPH_NETWORK_TYPES = ('gcn', 'gat', 'gin', 'graphsage', 'seg_graphsage')
SAMPLE_GRAPH_NETWORK_TYPES = ('gcn', 'gat', 'gin', 'graphsage')
SEGMENT_GRAPH_NETWORK_TYPES = ('seg_graphsage',)


def is_graph_network(network_type):
    return network_type in GRAPH_NETWORK_TYPES


def is_sample_graph_network(network_type):
    return network_type in SAMPLE_GRAPH_NETWORK_TYPES


def is_segment_graph_network(network_type):
    return network_type in SEGMENT_GRAPH_NETWORK_TYPES


def ensure_graph_dependencies(network_type):
    if is_graph_network(network_type) and not HAS_PYG:
        print(
            "Error: graph network '%s' requires torch-geometric. Install it before running graph mode." % network_type
        )
        print("Hint: pip install torch-geometric")
        sys.exit(-1)
