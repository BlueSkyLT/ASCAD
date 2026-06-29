"""
ASCAD Trace Visualization Module.

Core visualization helpers for inspecting individual traces, class means,
heatmaps, and basic leakage metrics.
"""

from statistics import NormalDist
from typing import Dict, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")


def _sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute standard error of the mean."""
    sample_size = values.shape[axis]
    return np.std(values, axis=axis, ddof=1) / np.sqrt(sample_size)


def _entropy_from_probabilities(probabilities: np.ndarray) -> float:
    """Compute base-2 entropy from a probability vector."""
    valid = probabilities[probabilities > 0]
    if valid.size == 0:
        return 0.0
    return float(-np.sum(valid * np.log2(valid)))


def _welch_t_statistic(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute the absolute Welch t-statistic for two samples."""
    mean_diff = group_a.mean() - group_b.mean()
    var_a = group_a.var(ddof=1)
    var_b = group_b.var(ddof=1)
    denom = np.sqrt(var_a / len(group_a) + var_b / len(group_b) + 1e-12)
    return float(abs(mean_diff / denom))


class TraceVisualizer:
    """Visualization suite for ASCAD power traces."""

    def __init__(self, figsize: Tuple[int, int] = (14, 6), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi

    def plot_individual_traces(
        self,
        traces: np.ndarray,
        labels: np.ndarray,
        traces_per_class: int = 20,
        num_classes: Optional[int] = None,
        save_path: Optional[str] = None,
    ):
        """Plot overlaid traces grouped by label class."""
        if num_classes is None:
            num_classes = len(np.unique(labels))

        unique_labels = np.unique(labels)[:num_classes]
        n_cols = 4
        n_rows = (len(unique_labels) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), dpi=self.dpi)
        axes = np.atleast_1d(axes).flatten()

        for idx, label in enumerate(unique_labels):
            ax = axes[idx]
            class_indices = np.where(labels == label)[0]
            if len(class_indices) > traces_per_class:
                selected_indices = np.random.choice(class_indices, traces_per_class, replace=False)
            else:
                selected_indices = class_indices

            for trace_idx in selected_indices:
                ax.plot(traces[trace_idx], alpha=0.35, linewidth=0.7, color="#2E86AB")

            mean_trace = traces[class_indices].mean(axis=0)
            ax.plot(mean_trace, color="#A23B72", linewidth=2, label="Mean", zorder=10)
            ax.set_title(f"Label {label} (n={len(class_indices)})", fontsize=10, fontweight="bold")
            ax.set_xlabel("Sample", fontsize=9)
            ax.set_ylabel("Power", fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8)

        for idx in range(len(unique_labels), len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle(f"Individual Traces Overlay ({traces_per_class} traces/class)", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig, axes[: len(unique_labels)]

    def plot_mean_traces(
        self,
        traces: np.ndarray,
        labels: np.ndarray,
        num_classes: Optional[int] = None,
        confidence: float = 0.95,
        save_path: Optional[str] = None,
    ):
        """Plot class means with confidence intervals."""
        unique_labels = np.unique(labels)
        if num_classes is not None:
            unique_labels = unique_labels[:num_classes]

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        colors = plt.cm.tab20c(np.linspace(0, 1, len(unique_labels)))

        for idx, label in enumerate(unique_labels):
            class_traces = traces[labels == label]
            mean_trace = class_traces.mean(axis=0)
            sem = _sem(class_traces, axis=0)
            t_val = NormalDist().inv_cdf((1 + confidence) / 2)
            ci = t_val * sem
            samples = np.arange(len(mean_trace))
            color = colors[idx] if idx < len(colors) else "C0"
            ax.plot(samples, mean_trace, linewidth=1.2, color=color, alpha=0.8)
            ax.fill_between(samples, mean_trace - ci, mean_trace + ci, alpha=0.12, color=color)
            if idx % 10 == 0:
                ax.plot([], [], color=color, linewidth=2, label=f"Label {label}")

        ax.set_xlabel("Sample Index", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Power Consumption", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Mean Power Traces by Sbox Output ({confidence * 100:.0f}% CI)",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=9, loc="best", ncol=2)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_heatmap_mean_power(
        self,
        traces: np.ndarray,
        labels: np.ndarray,
        num_classes: int = 256,
        cmap: str = "RdYlBu_r",
        save_path: Optional[str] = None,
    ):
        """Plot class-by-time heatmap of mean trace power."""
        unique_labels = np.unique(labels)
        n_classes = min(len(unique_labels), num_classes)
        heatmap_data = np.zeros((n_classes, traces.shape[1]))
        for idx, label in enumerate(unique_labels[:n_classes]):
            heatmap_data[idx] = traces[labels == label].mean(axis=0)

        fig, ax = plt.subplots(figsize=(16, 8), dpi=self.dpi)
        im = ax.imshow(heatmap_data, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_ylabel("Sbox Output Class", fontsize=11, fontweight="bold")
        ax.set_xlabel("Sample Index", fontsize=11, fontweight="bold")
        ax.set_title("Mean Power Heatmap: Class x Time", fontsize=12, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mean Power (a.u.)", fontsize=10, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig, ax

    def compute_signal_quality_metrics(
        self,
        traces: np.ndarray,
        labels: np.ndarray,
        sample_indices: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute SNR, correlation, Welch t-statistic, and mutual information."""
        n_samples = traces.shape[1]
        if sample_indices is None:
            sample_indices = np.arange(n_samples)

        snr = np.zeros(n_samples)
        correlation = np.zeros(n_samples)
        welch_tstat = np.zeros(n_samples)
        mutual_info = np.zeros(n_samples)
        unique_labels = np.unique(labels)

        for sample_idx in sample_indices:
            sample_data = traces[:, sample_idx]
            between_var = np.var([sample_data[labels == label].mean() for label in unique_labels])
            within_var = np.mean([sample_data[labels == label].var() for label in unique_labels])
            snr[sample_idx] = between_var / (within_var + 1e-10)
            correlation[sample_idx] = np.corrcoef(sample_data, labels)[0, 1]

            label_groups = [sample_data[labels == label] for label in unique_labels]
            if len(label_groups) >= 2:
                means = np.array([group.mean() for group in label_groups])
                idx1, idx2 = np.argsort(means)[[0, -1]]
                welch_tstat[sample_idx] = _welch_t_statistic(label_groups[idx1], label_groups[idx2])

            sample_bins = np.digitize(sample_data, np.linspace(sample_data.min(), sample_data.max(), 10))
            joint_entropy = 0.0
            for sample_bin in np.unique(sample_bins):
                for label in unique_labels:
                    p_joint = np.mean((sample_bins == sample_bin) & (labels == label))
                    if p_joint > 0:
                        joint_entropy -= p_joint * np.log2(p_joint)
            h_sample = _entropy_from_probabilities(np.bincount(sample_bins, minlength=11) / len(sample_bins))
            h_label = _entropy_from_probabilities(np.bincount(labels, minlength=256) / len(labels))
            mutual_info[sample_idx] = h_sample + h_label - joint_entropy

        return {
            "snr": snr,
            "correlation": correlation,
            "welch_tstat": welch_tstat,
            "mutual_info": mutual_info,
        }

    def plot_signal_quality(
        self,
        traces: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[str] = None,
        compute_all: bool = True,
    ):
        """Plot the leakage metrics in a 2x2 figure."""
        sample_indices = np.arange(traces.shape[1])
        if not compute_all and traces.shape[1] > 500:
            sample_indices = np.linspace(0, traces.shape[1] - 1, 500, dtype=int)

        metrics = self.compute_signal_quality_metrics(traces, labels, sample_indices)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)
        axes = axes.flatten()

        plot_specs = [
            ("snr", "#2E86AB", "SNR (Signal-to-Noise Ratio)", "SNR"),
            ("correlation", "#A23B72", "|Correlation| with Sbox Output", "|Correlation|"),
            ("welch_tstat", "#F18F01", "Welch's t-statistic", "t-statistic"),
            ("mutual_info", "#06A77D", "Mutual Information with Label", "MI (bits)"),
        ]

        for ax, (metric_key, color, title, ylabel) in zip(axes, plot_specs):
            values = np.abs(metrics[metric_key]) if metric_key == "correlation" else metrics[metric_key]
            ax.plot(values, linewidth=1.5, color=color)
            ax.fill_between(np.arange(len(values)), values, alpha=0.3, color=color)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("Sample Index", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Signal Quality Metrics: Leakage Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig
