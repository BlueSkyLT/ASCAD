# TimesNet for ASCAD

This folder contains a self-contained ASCAD-focused TimesNet implementation.

Scope:
- Read-only HDF5 loading for the three ATMEGA fixed-key datasets
- TimesNet baseline classifier
- ASCAD-tuned TimesNet classifier head variant
- Local data utilities and reusable embedding / Inception blocks
- Optional frozen GPT-2/Llama-3 trace classifier (`RouteATimeLLMClassifier`)

The original ASCAD datasets are not modified.

## Frozen LLM route

`RouteATimeLLMClassifier` adapts the time-series reprogramming approach to
ASCADv2: traces are split into overlapping patches, prefixed with a domain
prompt, and passed through a frozen Transformer.  The patch projector and
classification head remain trainable, so an ASCADv2 classifier can be built
without fine-tuning the language model:

```python
from TimesNet import RouteATimeLLMClassifier

model = RouteATimeLLMClassifier(
    trace_len=15000,
    num_classes=256,
    patch_size=256,
    stride=128,
    model_type="gpt2",  # use "llama3" for Meta-Llama-3-8B
)
```

Install `transformers` only when using this route.  Llama-3 access requires
the Hugging Face model permission and authentication.  For long ASCADv2
traces, choose patch settings that keep prompt plus patch count below the
backbone context length.  The backbone stays in evaluation mode and frozen;
unlike a `no_grad` block, the forward pass still propagates gradients to the
trainable patch projector.
