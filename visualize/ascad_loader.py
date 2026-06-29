"""
ASCAD Data Loader Utility

Provides efficient loading and inspection of ASCAD HDF5 files.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np


class ASCADLoader:
    """Utility class for loading ASCAD HDF5 files."""

    def __init__(self, cache_dir: Optional[Path] = None, verbose: bool = True):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.verbose = verbose
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _load_metadata_fields(self, metadata_obj, num_traces: Optional[int] = None) -> Dict[str, Any]:
        """Load metadata from either a group of datasets or a compound dataset."""
        metadata: Dict[str, Any] = {}

        if isinstance(metadata_obj, h5py.Group):
            for field_name in metadata_obj.keys():
                field_value = np.array(metadata_obj[field_name])
                if num_traces is not None:
                    field_value = field_value[:num_traces]
                metadata[field_name] = field_value
            return metadata

        if isinstance(metadata_obj, h5py.Dataset):
            dtype_names = metadata_obj.dtype.names or ()
            for field_name in dtype_names:
                field_value = np.array(metadata_obj[field_name])
                if num_traces is not None:
                    field_value = field_value[:num_traces]
                metadata[field_name] = field_value
            return metadata

        return metadata

    def load_ascad_file(
        self,
        file_path: str,
        split: str = "profiling",
        max_traces: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load a processed ASCAD HDF5 file."""
        if split not in ["profiling", "attack"]:
            raise ValueError(f"split must be 'profiling' or 'attack', got {split}")

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"ASCAD file not found: {file_path_obj}")

        self._log(f"Loading ASCAD {split} set from {file_path_obj}")
        split_key = "Profiling_traces" if split == "profiling" else "Attack_traces"

        with h5py.File(file_path_obj, "r") as hf:
            traces_key = f"{split_key}/traces"
            labels_key = f"{split_key}/labels"
            if traces_key not in hf or labels_key not in hf:
                raise ValueError(f"Required keys {traces_key} or {labels_key} not found in HDF5 file")

            traces = np.array(hf[traces_key], dtype=np.float32)
            labels = np.array(hf[labels_key], dtype=np.uint8)
            if max_traces is not None:
                traces = traces[:max_traces]
                labels = labels[:max_traces]

            self._log(f"  Traces shape: {traces.shape}, dtype: {traces.dtype}")
            self._log(f"  Labels shape: {labels.shape}, unique classes: {len(np.unique(labels))}")

            metadata: Dict[str, Any] = {}
            metadata_key = f"{split_key}/metadata"
            if metadata_key in hf:
                metadata = self._load_metadata_fields(hf[metadata_key], max_traces)
                if metadata:
                    self._log(f"  Metadata keys loaded: {list(metadata.keys())}")

        return traces, labels, metadata

    def load_raw_traces(
        self,
        file_path: str,
        sample_range: Tuple[int, int] = (0, 700),
        max_traces: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load a slice of the raw ASCAD trace file."""
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Raw traces file not found: {file_path_obj}")

        start_sample, end_sample = sample_range
        self._log(f"Loading raw traces from {file_path_obj}")
        self._log(f"  Sample range: [{start_sample}, {end_sample}]")

        with h5py.File(file_path_obj, "r") as hf:
            if "traces" not in hf:
                raise ValueError("Key 'traces' not found in HDF5 file")

            num_traces = hf["traces"].shape[0]
            if max_traces is not None:
                num_traces = min(num_traces, max_traces)

            traces = np.array(hf["traces"][:num_traces, start_sample:end_sample], dtype=np.float32)
            self._log(f"  Traces shape: {traces.shape}, dtype: {traces.dtype}")

            metadata: Dict[str, Any] = {}
            if "metadata" in hf:
                metadata = self._load_metadata_fields(hf["metadata"], num_traces)
                if metadata:
                    self._log(f"  Metadata keys loaded: {list(metadata.keys())}")

        return traces, metadata

    def get_dataset_info(self, file_path: str) -> Dict[str, Any]:
        """Inspect an ASCAD HDF5 file without loading all arrays."""
        file_path_obj = Path(file_path)
        info: Dict[str, Any] = {}

        with h5py.File(file_path_obj, "r") as hf:
            def collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    info[name] = {
                        "shape": obj.shape,
                        "dtype": str(obj.dtype),
                        "size_mb": obj.nbytes / (1024 * 1024),
                    }

            hf.visititems(collect)

        return info


def load_ascad(
    file_path: str,
    split: str = "profiling",
    max_traces: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Convenience wrapper for loading processed ASCAD data."""
    loader = ASCADLoader(verbose=False)
    return loader.load_ascad_file(file_path, split=split, max_traces=max_traces)


def load_raw(
    file_path: str,
    sample_range: Tuple[int, int] = (0, 700),
    max_traces: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convenience wrapper for loading raw ASCAD trace slices."""
    loader = ASCADLoader(verbose=False)
    return loader.load_raw_traces(file_path, sample_range=sample_range, max_traces=max_traces)
