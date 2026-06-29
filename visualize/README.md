# ASCAD Visualization Toolkit

Utilities for exploring ASCAD power traces in /home/lantian/ASCAD/visualize.

## Files

- ascad_loader.py: Load processed ASCAD datasets and raw trace windows.
- trace_visualizer.py: Plot individual traces, mean traces, heatmaps, and leakage metrics.
- comparative_analysis.py: Compare synchronized and desynchronized datasets.
- explore_traces.ipynb: Notebook for interactive inspection.

## Environment

Use the repository Python environment described in the project instructions:

```bash
conda activate py10
```

Dependencies used by these scripts are already expected in that environment:

- numpy
- scipy
- h5py
- matplotlib
- jupyter

## Notebook usage

Open the notebook from the repository root or from the visualize directory:

```bash
cd /home/lantian/ASCAD/visualize
jupyter notebook explore_traces.ipynb
```

The notebook walks through:

1. Loading ASCAD.h5.
2. Printing one trace and its metadata.
3. Plotting the selected trace with label and metadata in the title.
4. Plotting 20 traces from the same class on one figure.
5. Plotting 20 traces from two different classes on the same axes.

## Script usage

Example from a Python session:

```python
from ascad_loader import load_ascad
from trace_visualizer import TraceVisualizer
from comparative_analysis import ComparativeAnalyzer

traces, labels, metadata = load_ascad(
    "../ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
)

viz = TraceVisualizer()
viz.plot_individual_traces(traces, labels, traces_per_class=20, num_classes=16)
viz.plot_mean_traces(traces, labels, num_classes=32)
viz.plot_heatmap_mean_power(traces, labels)
viz.plot_signal_quality(traces, labels, compute_all=False)

analyzer = ComparativeAnalyzer()
analyzer.compare_sync_vs_desync(
    "../ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5",
    "../ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD_desync50.h5",
    "../ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD_desync100.h5",
)
```

## Notes

- Processed traces are 700 samples wide.
- The processed profiling split contains 50,000 traces and 256 labels.
- Metadata can be stored either as separate HDF5 datasets or as a compound dataset; ascad_loader.py handles both.
- Raw trace inspection is much heavier than processed-file inspection.
