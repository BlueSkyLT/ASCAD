# Plan: Trace-to-Token S-Box Classification with a Frozen LLM

## 1. What the task is

Each ASCAD example is one power trace `x` and one target S-box value
`y ∈ {0, ..., 255}`. This is a supervised 256-class classification problem:

```text
trace x  ->  logits[256]  ->  softmax probabilities  ->  predicted S-box byte
```

The model is not asked to generate text. Language is used as an interface and
as a source of pretrained sequence-processing features; the final prediction
is a normal classification head.

## 2. What a trace segment means

A segment is not automatically an “alpha token”, “beta token”, or a single
known leakage variable. A side-channel trace is a time-ordered mixture of
operations, masking effects, transitions, noise, and acquisition artefacts.
Some regions may correlate with alpha/beta shares or C leakage, while other
regions may carry complementary or redundant evidence. Their meanings should
be treated as latent and verified empirically, not assigned by visual
position alone.

Divide the trace into overlapping windows (patches). Each patch is a short
numeric vector. A trainable projection maps that vector into the LLM hidden
dimension:

```text
trace [T] -> patches [N, patch_size] -> patch embeddings [N, d_model]
```

The patch index preserves temporal order. Overlap gives nearby operations
multiple views and reduces sensitivity to an arbitrary boundary. Start with
several patch sizes and strides; choose them using validation performance and
the backbone context limit.

## 3. “Different traces become different sentences”: the correct analogy

The sentence analogy is useful only at the sequence-structure level:

- patches are ordered units, like words;
- a trace becomes an ordered sequence of learned continuous embeddings;
- two traces can contain shifted, noisy, or differently expressed evidence.

However, patches are not literal words and should not be converted to text
through hand-written names. Converting every numeric window to a spelling or
token string would discard amplitude and phase information and introduce
arbitrary semantics. The reliable representation is a continuous patch
embedding. A short fixed domain prompt can provide task context, but it does
not describe every patch.

If interpretable “trace words” are desired, learn them as a secondary
analysis: cluster patch embeddings, inspect their time locations and leakage
correlations, and test whether replacing or masking those clusters changes
classification. Do not use cluster names as labels unless independently
validated.

## 4. Model architecture

Use the existing `RouteATimeLLMClassifier` as the Route A implementation:

1. Normalize traces using statistics from the profiling/training split only.
2. Extract overlapping patches with `unfold`.
3. Apply a trainable linear patch projector to the frozen LLM dimension.
4. Tokenize one domain prompt, for example
   `Classify the ASCAD side-channel trace and infer its S-box output.`
5. Convert prompt tokens to the frozen model's input embeddings.
6. Concatenate prompt embeddings followed by trace patch embeddings.
7. Run the combined sequence through the frozen GPT-2 or Llama-3 backbone.
8. Keep the final hidden states corresponding to the patches.
9. Flatten them and apply a trainable linear head producing 256 logits.

The prompt is context, not the answer. It must not contain the true S-box
value, plaintext-derived target, or any label-dependent text.

## 5. What is trained?

The baseline trains only:

- the numeric patch-to-LLM projector;
- the output classification head.

The language-model backbone is frozen (`requires_grad=False`) and kept in
evaluation mode. The forward pass must still retain the computation graph
from the patch embeddings through the frozen layers to the output head, so
the projector receives gradients. Freezing parameters is different from
wrapping the whole forward pass in `torch.no_grad()`.

For each batch, minimize cross-entropy:

```text
loss = CrossEntropyLoss(logits, sbox_target)
```

The target is the integer S-box class, not a text string. Save only the
projector and head when the backbone is reproducibly loaded from its Hub
checkpoint.

## 6. Training procedure

1. Split profiling traces into training and validation sets without leakage.
2. Fit normalization on training traces and apply the same transform to
   validation and attack traces.
3. Start with GPT-2 on a small subset and confirm that loss decreases and
   tensor shapes are correct.
4. Train the projector and head with Adam/RAdam, gradient clipping, and
   early stopping on validation cross-entropy.
5. Compare patch sizes, strides, prompt variants, and normalization choices.
6. Run the selected configuration on all profiling traces.
7. Load the same frozen checkpoint and trained heads for attack evaluation.
8. Scale to Llama-3 only after the GPT-2 pipeline is reproducible.

The model should be evaluated against established CNN/MLP/TimesNet baselines
using identical profiling/attack splits and preprocessing.

## 7. How the final result is produced

Inference uses the same fixed prompt and the same patching pipeline used in
training:

```text
new trace -> normalize -> patch/project -> prepend prompt embeddings
          -> frozen LLM -> patch hidden states -> linear head
          -> 256 logits -> argmax or probability distribution
```

The predicted S-box output is `argmax(logits)`. For side-channel key recovery,
retain the full 256-class probability vector and combine log-probabilities
across attack traces according to the ASCAD ranking procedure. Prompting does
not replace this aggregation and does not require autoregressive generation.

## 8. Experiments that establish whether the method works

Report:

- cross-entropy, accuracy, top-k accuracy, MSE/MAE only where meaningful;
- key rank or guessing entropy on attack traces;
- confidence calibration and confusion patterns across 256 classes;
- runtime and memory for GPT-2 versus Llama-3.

Use ablations to distinguish real benefit from prompt or capacity effects:

- frozen pretrained backbone versus randomly initialized backbone;
- prompt versus no prompt or a neutral prompt;
- learned continuous patches versus hand-defined segment labels;
- overlapping versus non-overlapping patches;
- frozen backbone versus controlled fine-tuning;
- shuffled patch order and masked patch regions;
- different patch sizes and trace alignment/desynchronization.

An important sanity check is label permutation: if labels are randomized,
validation accuracy should return near chance. Another is temporal masking:
removing a genuinely informative region should reduce performance more than
removing an irrelevant region.

## 9. Recommended implementation milestones

### Milestone A — pipeline verification

Use GPT-2, a small profiling subset, and short runs. Verify normalization,
patch counts, prompt/patch sequence lengths, gradient flow to both trainable
modules, frozen-backbone parameters, and output shape `[batch, 256]`.

### Milestone B — ASCADv2 benchmark

Train on the complete profiling split, evaluate on the attack split, and
compare with the repository's existing classifiers. Keep the prompt fixed
while tuning architecture to avoid prompt overfitting.

### Milestone C — representation analysis

Measure patch importance by masking, visualize learned patch embeddings, and
test whether clusters correspond consistently to leakage regions across
traces. Treat any discovered alpha/beta/C interpretation as a result to
validate, not as an assumption in the model.

### Milestone D — capacity scaling

Repeat the locked benchmark with Meta-Llama-3-8B using the required Hugging
Face access and suitable GPU memory. Report gains relative to GPT-2 and
separate gains from model capacity, precision, and context length.

## 10. Main methodological answer

The goal is **not** to teach the model a dictionary mapping “segment name” to
token and then ask it to complete a sentence. The goal is to learn a
continuous trace representation and a discriminative mapping from that
representation to 256 S-box logits. The prompt supplies task context; the
trained classification head supplies the answer. Interpretability of
individual segments is investigated afterward with masking, clustering, and
leakage correlation experiments.
