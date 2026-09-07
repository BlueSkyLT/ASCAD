"""Frozen language-model adapters for ASCAD trace classification.

The adapter treats a power trace as a sequence of non-overlapping/overlapping
patches and uses the final hidden states as a learned representation.  Only
the patch projection and classification head are trainable.
"""

import torch
import torch.nn as nn


class RouteATimeLLMClassifier(nn.Module):
    """Classify ASCAD traces with a frozen GPT-2 or Llama-3 backbone.

    ``transformers`` is imported lazily so the existing CNN/TimesNet models do
    not require an LLM installation.  Inputs can be ``(batch, length)`` or
    ``(batch, length, 1)``; the latter is the format returned by the ASCAD
    dataset utilities.
    """

    MODEL_NAMES = {
        "gpt2": ("gpt2", 768),
        "llama3": ("meta-llama/Meta-Llama-3-8B", 4096),
    }

    def __init__(
        self,
        trace_len=700,
        num_classes=256,
        patch_size=16,
        stride=8,
        model_type="gpt2",
        prompt_text="Classify the side-channel trace for ASCAD.",
        model_name=None,
        torch_dtype=None,
    ):
        super().__init__()
        if model_type not in self.MODEL_NAMES:
            raise ValueError("model_type must be 'gpt2' or 'llama3'")
        if trace_len < patch_size:
            raise ValueError("trace_len must be at least patch_size")
        if patch_size <= 0 or stride <= 0:
            raise ValueError("patch_size and stride must be positive")

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "RouteATimeLLMClassifier requires transformers; "
                "install it with `pip install transformers`."
            ) from exc

        default_name, d_model = self.MODEL_NAMES[model_type]
        self.model_type = model_type
        self.trace_len = trace_len
        self.patch_size = patch_size
        self.stride = stride
        self.num_patches = (trace_len - patch_size) // stride + 1
        self.d_model = d_model
        self.prompt_text = prompt_text
        self.model_name = model_name or default_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_dtype = torch_dtype or (
            torch.bfloat16 if model_type == "llama3" else torch.float32
        )
        self.llm = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=load_dtype,
        )
        for parameter in self.llm.parameters():
            parameter.requires_grad = False
        self.llm.eval()

        self.in_projector = nn.Linear(patch_size, d_model)
        self.out_head = nn.Linear(self.num_patches * d_model, num_classes)

    def train(self, mode=True):
        """Keep the frozen backbone in evaluation mode while heads train."""
        super().train(mode)
        self.llm.eval()
        return self

    def forward(self, x, prompt_text=None):
        if x.ndim == 3:
            if x.shape[-1] != 1:
                raise ValueError("3D inputs must have shape (batch, length, 1)")
            x = x.squeeze(-1)
        if x.ndim != 2:
            raise ValueError("x must have shape (batch, length) or (batch, length, 1)")
        if x.shape[-1] != self.trace_len:
            raise ValueError(f"expected trace length {self.trace_len}, got {x.shape[-1]}")

        patches = x.unfold(-1, self.patch_size, self.stride)
        patch_embeds = self.in_projector(patches)

        prompt = prompt_text or self.prompt_text
        tokens = self.tokenizer(prompt, return_tensors="pt").input_ids
        embedding_layer = self.llm.get_input_embeddings()
        tokens = tokens.to(embedding_layer.weight.device)
        prompt_embeds = embedding_layer(tokens).expand(x.shape[0], -1, -1)
        patch_embeds = patch_embeds.to(prompt_embeds.dtype)
        combined = torch.cat((prompt_embeds, patch_embeds), dim=1)

        # Do not wrap this call in no_grad: gradients must reach the trainable
        # patch projector through the frozen backbone's input embeddings.
        outputs = self.llm(inputs_embeds=combined)
        ts_hidden = outputs.last_hidden_state[:, -self.num_patches :, :]
        return self.out_head(ts_hidden.reshape(x.shape[0], -1).to(self.out_head.weight.dtype))
