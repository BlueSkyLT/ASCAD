"""ASCADv1 leakage localizer (T-014).

Current method: TwoOrderWindows + loss_2o (profiled 2O CPA).
CE-searcher variants remain in this package but are not the claimed model.
"""

from .affine_moments import centered_product3, m3_table
from .combine import xor_combine
from .losses import independence_loss, total_loss
from .model import (
    AttentionBooleanSCA,
    BandAttentionProposer,
    DirectWindowSHead,
    GlobalPoolSHead,
    GlobalThetaSCA,
    SoftWindowBooleanSCA,
    ThreeOrderWindows,
    WindowProposer,
    build_poi_searcher,
    is_poi_searcher_network,
)

NETWORK_TYPE = "poi_searcher"
NETWORK_TYPES = ("poi_searcher", "poi_searcher_v1", "leakage_localizer")

__all__ = [
    "NETWORK_TYPE",
    "NETWORK_TYPES",
    "xor_combine",
    "independence_loss",
    "total_loss",
    "WindowProposer",
    "SoftWindowBooleanSCA",
    "DirectWindowSHead",
    "GlobalPoolSHead",
    "GlobalThetaSCA",
    "ThreeOrderWindows",
    "m3_table",
    "centered_product3",
    "build_poi_searcher",
    "is_poi_searcher_network",
]
