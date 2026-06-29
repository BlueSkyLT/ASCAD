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

GRAPH_NETWORK_TYPES = ('gcn', 'gat', 'gin', 'graphsage')


def is_graph_network(network_type):
    return network_type in GRAPH_NETWORK_TYPES


def ensure_graph_dependencies(network_type):
    if is_graph_network(network_type) and not HAS_PYG:
        print(
            "Error: graph network '%s' requires torch-geometric. Install it before running graph mode." % network_type
        )
        print("Hint: pip install torch-geometric")
        sys.exit(-1)
