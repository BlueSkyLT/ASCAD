"""
ASCAD Comparative Analysis Module.

Helpers for comparing synchronized and desynchronized ASCAD variants.
"""

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ascad_loader import ASCADLoader


class ComparativeAnalyzer:
    """Compare trace behavior across ASCAD dataset variants."""

    def __init__(self, figsize: Tuple[int, int] = (14, 6), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        self.loader = ASCADLoader(verbose=False)

    def compare_sync_vs_desync(
        self,
        file_sync: str,
        file_desync_50: str,
        file_desync_100: str,
        num_classes: int = 256,
        save_path: Optional[str] = None,
    ):
        """Overlay class means for synchronized and desynchronized datasets."""
        traces_sync, labels_sync, _ = self.loader.load_ascad_file(file_sync, split="profiling")
        traces_d50, labels_d50, _ = self.loader.load_ascad_file(file_desync_50, split="profiling")
        traces_d100, labels_d100, _ = self.loader.load_ascad_file(file_desync_100, split="profiling")

        unique_labels = np.unique(labels_sync)[:num_classes]
        n_plots = min(len(unique_labels), 9)
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), dpi=self.dpi)
        axes = axes.flatten()
        colors = ["#2E86AB", "#F18F01", "#C73E1D"]
        legends = ["Synchronized", "Desync ±50", "Desync ±100"]

        for plot_idx, label in enumerate(unique_labels[:n_plots]):
            ax = axes[plot_idx]
            mean_sync = traces_sync[labels_sync == label].mean(axis=0)
            mean_d50 = traces_d50[labels_d50 == label].mean(axis=0)
            mean_d100 = traces_d100[labels_d100 == label].mean(axis=0)
            samples = np.arange(len(mean_sync))
            ax.plot(samples, mean_sync, linewidth=2, color=colors[0], label=legends[0])
            ax.plot(samples, mean_d50, linewidth=2, color=colors[1], label=legends[1], alpha=0.8)
            ax.plot(samples, mean_d100, linewidth=2, color=colors[2], label=legends[2], alpha=0.7)
            ax.set_title(f"Label {label}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Sample Index", fontsize=9)
            ax.set_ylabel("Mean Power", fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8)

        for plot_idx in range(n_plots, len(axes)):
            fig.delaxes(axes[plot_idx])

        fig.suptitle("Effect of Desynchronization on Trace Leakage", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def _compute_snr_vector(self, traces: np.ndarray, labels: np.ndarray) -> np.ndarray:
        snr_values = np.zeros(traces.shape[1])
        unique_labels = np.unique(labels)
        for sample_idx in range(traces.shape[1]):
            sample_data = traces[:, sample_idx]
            between_var = np.var([sample_data[labels == label].mean() for label in unique_labels])
            within_var = np.mean([sample_data[labels == label].var() for label in unique_labels])
            snr_values[sample_idx] = between_var / (within_var + 1e-10)
        return snr_values

    def compute_leakage_reduction(
        self,
        file_sync: str,
        file_desync_50: str,
        file_desync_100: str,
    ) -> Dict[str, float]:
        """Quantify SNR reduction due to desynchronization."""
        traces_sync, labels_sync, _ = self.loader.load_ascad_file(file_sync, split="profiling")
        traces_d50, _, _ = self.loader.load_ascad_file(file_desync_50, split="profiling")
        traces_d100, _, _ = self.loader.load_ascad_file(file_desync_100, split="profiling")

        snr_sync = self._compute_snr_vector(traces_sync, labels_sync)
        snr_d50 = self._compute_snr_vector(traces_d50, labels_sync)
        snr_d100 = self._compute_snr_vector(traces_d100, labels_sync)

        return {
            "snr_sync_max": float(snr_sync.max()),
            "snr_sync_mean": float(snr_sync.mean()),
            "snr_desync50_max": float(snr_d50.max()),
            "snr_desync50_mean": float(snr_d50.mean()),
            "snr_desync100_max": float(snr_d100.max()),
            "snr_desync100_mean": float(snr_d100.mean()),
            "reduction_50_max": float((1 - snr_d50.max() / (snr_sync.max() + 1e-10)) * 100),
            "reduction_100_max": float((1 - snr_d100.max() / (snr_sync.max() + 1e-10)) * 100),
        }

    def plot_leakage_comparison(
        self,
        file_sync: str,
        file_desync_50: str,
        file_desync_100: str,
        save_path: Optional[str] = None,
    ):
        """Plot SNR curves for synchronized and desynchronized datasets."""
        traces_sync, labels_sync, _ = self.loader.load_ascad_file(file_sync, split="profiling")
        traces_d50, _, _ = self.loader.load_ascad_file(file_desync_50, split="profiling")
        traces_d100, _, _ = self.loader.load_ascad_file(file_desync_100, split="profiling")

        snr_sync = self._compute_snr_vector(traces_sync, labels_sync)
        snr_d50 = self._compute_snr_vector(traces_d50, labels_sync)
        snr_d100 = self._compute_snr_vector(traces_d100, labels_sync)
        samples = np.arange(len(snr_sync))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=self.dpi)
        ax1.plot(samples, snr_sync, linewidth=2, label="Synchronized", color="#2E86AB")
        ax1.plot(samples, snr_d50, linewidth=2, label="Desync ±50", color="#F18F01")
        ax1.plot(samples, snr_d100, linewidth=2, label="Desync ±100", color="#C73E1D")
        ax1.set_ylabel("SNR", fontsize=11, fontweight="bold")
        ax1.set_title("SNR Comparison Across Synchronization Variants", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        reduction_50 = (1 - snr_d50 / (snr_sync + 1e-10)) * 100
        reduction_100 = (1 - snr_d100 / (snr_sync + 1e-10)) * 100
        ax2.plot(samples, reduction_50, linewidth=2, label="Desync ±50", color="#F18F01")
        ax2.plot(samples, reduction_100, linewidth=2, label="Desync ±100", color="#C73E1D")
        ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax2.set_xlabel("Sample Index", fontsize=11, fontweight="bold")
        ax2.set_ylabel("SNR Reduction (%)", fontsize=11, fontweight="bold")
        ax2.set_title("Leakage Reduction Due to Desynchronization", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def compare_raw_vs_poi(
        self,
        file_raw: str,
        file_poi: str,
        sample_range: Tuple[int, int] = (300, 1000),
        num_examples: int = 5,
        save_path: Optional[str] = None,
    ):
        """Compare a window from the raw trace file with extracted POI traces."""
        traces_raw, _ = self.loader.load_raw_traces(file_raw, sample_range=sample_range, max_traces=num_examples)
        traces_poi, _, _ = self.loader.load_ascad_file(file_poi, split="profiling")
        traces_poi = traces_poi[:num_examples]

        fig, axes = plt.subplots(num_examples, 1, figsize=(14, 3 * num_examples), dpi=self.dpi)
        axes = np.atleast_1d(axes)
        for idx in range(num_examples):
            ax = axes[idx]
            raw_norm = (traces_raw[idx] - traces_raw[idx].mean()) / (traces_raw[idx].std() + 1e-10)
            poi_norm = (traces_poi[idx] - traces_poi[idx].mean()) / (traces_poi[idx].std() + 1e-10)
            ax.plot(raw_norm, linewidth=1, label="Raw window", color="#2E86AB", alpha=0.7)
            ax.plot(poi_norm, linewidth=1.5, label="POI-extracted", color="#A23B72", alpha=0.8)
            ax.set_title(f"Example {idx}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Sample Index", fontsize=9)
            ax.set_ylabel("Normalized Power", fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=9)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig


def print_desync_summary(metrics: Dict[str, float]) -> None:
    """Print a short summary of desynchronization impact."""
    print("\n" + "=" * 70)
    print("DESYNCHRONIZATION IMPACT SUMMARY")
    print("=" * 70)
    print(f"\nMax SNR:\n  Synchronized:    {metrics['snr_sync_max']:.4f}")
    print(f"  Desync ±50:      {metrics['snr_desync50_max']:.4f} ({metrics['reduction_50_max']:+.1f}%)")
    print(f"  Desync ±100:     {metrics['snr_desync100_max']:.4f} ({metrics['reduction_100_max']:+.1f}%)")
    print(f"\nMean SNR:\n  Synchronized:    {metrics['snr_sync_mean']:.4f}")
    print(f"  Desync ±50:      {metrics['snr_desync50_mean']:.4f}")
    print(f"  Desync ±100:     {metrics['snr_desync100_mean']:.4f}")
    print("=" * 70 + "\n")
