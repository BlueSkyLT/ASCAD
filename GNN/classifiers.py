import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    GATConv,
    GCNConv,
    GINConv,
    SAGEConv,
    ensure_graph_dependencies,
    global_mean_pool,
)


class GCNClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=3, dropout=0.2, classes=256):
        super().__init__()
        ensure_graph_dependencies('gcn')
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


class GATClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=3, dropout=0.2, heads=4, classes=256):
        super().__init__()
        ensure_graph_dependencies('gat')
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=False))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=False))
        self.classifier = nn.Linear(hidden_dim, classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


class GINClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=3, dropout=0.2, classes=256):
        super().__init__()
        ensure_graph_dependencies('gin')
        self.dropout = dropout
        self.convs = nn.ModuleList()
        mlp_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINConv(mlp_in))
        for _ in range(num_layers - 1):
            mlp_h = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp_h))
        self.classifier = nn.Linear(hidden_dim, classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


class GraphSAGEClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=3, dropout=0.2, classes=256):
        super().__init__()
        ensure_graph_dependencies('graphsage')
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.classifier(x)
