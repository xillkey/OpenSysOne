<div align="center">

# OpenSysOne

**Train System-1 decision models on any encoder**

Give the model some context and a few typed questions, and get calibrated probabilities for every question in a single forward pass. No text generation, no parsing, no made-up answers.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#limitations)

[Quickstart](#quickstart) · [Usage](#usage) · [Design switches](#design-switches) · [How it works](#how-it-works) · [Ablations](#ablations) · [Roadmap](#roadmap)

**English** · [中文](README_CN.md)

</div>

---

*Train Laya/Jev-style System-1 decision models on any swappable encoder backbone: one forward pass, typed questions (choice / score / bool), calibrated probabilities, a `/v1/systemone` server and ONNX export.*

## Overview

Many decisions inside agents and business workflows are really multiple-choice questions: which team should handle this ticket, how urgent is this message, is the customer asking for a refund. Having a large language model generate an answer and then parsing it is slow and can go off the rails. A **decision model** (System-1 model) takes a different approach: it feeds the question and its options together into a bidirectional encoder and outputs a probability for every option in a single forward pass.

OpenSysOne is a framework for training this kind of model. It follows the same idea as Laya and Jev, with one key difference: **the backbone is freely swappable**. Going from ModernBERT or XLM-R to your own encoder is a one-line config change.

```python
from opensysone import Predictor

pred = Predictor.from_pretrained("runs/toy-tiny/final")   # the toy model trained in Quickstart
pred.decide({
    "state": {"channel": "email", "lang": "en", "message": "Hi, I was charged twice for my subscription. This is urgent, please reply today. I want a refund."},
    "questions": {
        "team":    {"type": "choice", "question": "Which team should handle this ticket?", "options": ["billing", "shipping", "technical", "account"]},
        "urgency": {"type": "score",  "question": "How urgent is this ticket?", "min": 1, "max": 5},
        "refund":  {"type": "bool",   "question": "Is the customer asking for a refund?"},
    },
})
# {"results": {"team":    {"answer": "billing", "confidence": 1.0,  "probabilities": {...}},
#              "urgency": {"answer": "4", "confidence": 0.93, "expected": 3.93, ...},
#              "refund":  {"answer": "yes", "probability": 1.0, ...}},
#  "latency_ms": 6.5}
```

> OpenSysOne is an independent implementation. It is not affiliated with Laya (Convai Innovations) or Jev (TypeSafe) and contains none of their code or weights. "Jev-style interface" means the request structure is similar (state + questions); check the system you are integrating with for exact field details.

## Features

- 🔌 **Swappable backbone**: any bidirectional encoder on Hugging Face works with a one-line config change, and custom backbones can be plugged in through a registry.
- 🧩 **Three question types, one model**: `choice` (single choice), `score` (ordinal rating) and `bool` (yes/no). Options are supplied at request time, so a new option set needs no retraining.
- 🎯 **Calibrated probabilities**: after training, a per-type temperature is fitted on the validation set, optionally bucketed by number of options, so confidences can be used directly for thresholds.
- 📐 **Strictly proper scoring rules**: log loss, plus Ranked Probability Score for score questions and optional Brier; or switch to the RLCD loss.
- 🏷️ **Soft labels**: labels can be probability distributions, which makes distillation from an LLM teacher easy.
- 🎛️ **Switchable design**: question position, option marker, option representation, parallel layout (invariant to option order), head structure and more are all config switches, convenient for ablations.
- 💾 **Low-memory friendly**: LoRA, gradient checkpointing, mixed precision; small models train on CPU without a GPU.
- 🚀 **Ready to deploy**: a `POST /v1/systemone` server with request micro-batching, plus ONNX export and onnxruntime inference.
- 🧪 **Testable offline**: build a randomly initialized tiny BERT from your data and run the full pipeline without network access.

## Installation

```bash
git clone https://github.com/xillkey/OpenSysOne.git
cd opensysone
pip install -e ".[all]"
```

Optional extras:

| extra | contents |
|---|---|
| (none) | training, evaluation, inference |
| `lora` | LoRA fine-tuning (peft) |
| `serve` | HTTP server (FastAPI, uvicorn) |
| `onnx` | ONNX export and inference (onnx, onnxruntime) |
| `dev` | everything above, plus pytest |
| `all` | lora + serve + onnx |

Requires Python ≥ 3.9, PyTorch ≥ 2.1 and transformers ≥ 4.40; ModernBERT needs transformers ≥ 4.48. Tested on Python 3.13, PyTorch 2.14, transformers 5.20 and peft 0.21.

## Quickstart

The following runs on CPU and does not download any pretrained model:

```bash
# 1. Generate bilingual (Chinese/English) toy ticket data
python examples/toy_tickets/make_data.py

# 2. Build a randomly initialized tiny BERT from that data (only for exercising the pipeline)
opensysone make-tiny-backbone --corpus examples/toy_tickets/data/train.jsonl --out runs/tiny-backbone

# 3. Train; the model is calibrated on the validation set at the end, final model in runs/toy-tiny/final
opensysone train -c configs/toy_tiny.yaml

# 4. Evaluate, and measure single-request latency
opensysone eval --model runs/toy-tiny/final --data examples/toy_tickets/data/test.jsonl --latency 50

# 5. Run inference on one request
opensysone predict --model runs/toy-tiny/final --input examples/request.json

# 6. Start the server
opensysone serve --model runs/toy-tiny/final --port 8000
```

Step 3 takes about 5 minutes on a 2-core CPU. Reference results on the toy test set (after calibration):

| type | accuracy | NLL | ECE | score MAE |
|---|---|---|---|---|
| choice | 1.000 | 0.000 | 0.000 | – |
| score | 0.987 | 0.105 | 0.005 | 0.028 |
| bool | 1.000 | 0.000 | 0.000 | – |

CPU latency p50 for a single request (3 questions) is about 7 ms.

> This data is template-generated. The numbers above only show that the pipeline works; they say nothing about performance on real tasks.

Switching to a real pretrained backbone is just a config change, e.g. `opensysone train -c configs/modernbert_base.yaml`.

## Usage

### Data format

JSONL, one example per line. `state` can be text or a JSON object (serialized to a JSON string with sorted keys).

```json
{"id": "t00001",
 "state": {"channel": "email", "message": "I was charged twice for my subscription, please reply today. I want a refund."},
 "questions": {
   "team":    {"type": "choice", "text": "Which team should handle this?", "options": ["billing", "shipping", "technical", "account"], "label": "billing"},
   "urgency": {"type": "score",  "text": "How urgent is it?", "min": 1, "max": 5, "label": 4},
   "refund":  {"type": "bool",   "text": "Is a refund requested?", "label": true}}}
```

| field | description |
|---|---|
| `type` | `choice` / `score` / `bool`; aliases such as `rating`, `scale`, `yesno`, `noul` are also accepted |
| `text` | the question; `question` also works |
| `options` | required for `choice`; `score` can define custom levels with `options` or generate integer levels with `min` / `max`; not needed for `bool` |
| `label` | hard label: option name, option index, `true` / `false` / `"yes"`; soft label: a probability list or `{"option": prob}`; `bool` also accepts a probability in 0–1 |

`questions` can also be a list of objects with an `id` field. Questions without a `label` are skipped during training. Check your data with:

```bash
opensysone validate-data --data your_train.jsonl
```

### Training

```bash
opensysone train -c configs/modernbert_base.yaml
opensysone train -c configs/modernbert_base.yaml --set train.lr=2e-5 --set data.max_length=256   # override any config from the CLI
```

The output directory contains:
- `best/`: the step with the best validation NLL, uncalibrated.
- `final/`: the calibrated final model; use this for inference and deployment.
- `metrics.json`: training history and final validation metrics.

Each model directory contains `backbone/` (weights and tokenizer), `head.safetensors` and `opensysone.json` (serialization parameters, temperatures, training config).

<details>
<summary><b>Common training options (<code>train:</code> section)</b></summary>

| key | default | description |
|---|---|---|
| `lr` / `head_lr` | 3e-5 / 1e-3 | learning rates for the backbone and the output head |
| `llrd` | 0.9 | layer-wise learning rate decay; lower layers get smaller rates; 1.0 disables it |
| `epochs` / `max_steps` | 3 / 0 | `max_steps > 0` overrides epochs |
| `batch_size` / `grad_accum` | 16 / 1 | |
| `precision` | auto | bf16 or fp16 automatically on GPU, fp32 on CPU |
| `eval_every` | 0 | evaluate every N steps; 0 means once per epoch |
| `early_stop_patience` | 3 | stop after this many evaluations without validation NLL improvement |
| `calibrate` | true | fit temperatures on the validation set after training |
| `calibrate_by_options` | false | additionally fit temperatures bucketed by number of options; see [Design switches](#design-switches) |

</details>

### Swapping the backbone

**Hugging Face encoders** (most common):

```yaml
backbone:
  name: hf
  model: answerdotai/ModernBERT-base     # any bidirectional encoder
  gradient_checkpointing: false
  trust_remote_code: false
  attn_implementation: null               # e.g. flash_attention_2
```

The framework automatically adds two special tokens, `[Q]` and `[OPT]`, extends the vocabulary and initializes their embeddings.

| scenario | example backbones |
|---|---|
| English, low latency | `answerdotai/ModernBERT-base` / `-large` |
| Multilingual | `jhu-clsp/mmBERT-base`, `FacebookAI/xlm-roberta-base`, `intfloat/multilingual-e5-base` |
| Mainly Chinese | `hfl/chinese-roberta-wwm-ext`, `BAAI/bge-base-zh-v1.5` |
| English, accuracy first | `microsoft/deberta-v3-base` / `-large` |

The `hf` adapter does not support decoder-only (GPT-style) or encoder-decoder (T5-style) models; for the latter you can write a custom backbone that wraps `T5EncoderModel`.

> **Check the tokenizer before switching backbones.** If much of your target language turns into `[UNK]`, the model will be confidently wrong. A quick `tokenizer.tokenize("your sample text")` is enough to check.

**LoRA** (low memory):

```yaml
backbone:
  name: hf
  model: FacebookAI/xlm-roberta-base
  gradient_checkpointing: true
  train_embeddings: true          # let the new [Q]/[OPT] embeddings learn too
  lora: {r: 16, alpha: 32, dropout: 0.05, target_modules: all-linear}
```

LoRA is merged into the weights on save, so the exported model has exactly the same format as a fully fine-tuned one. See [`configs/multilingual_lora.yaml`](configs/multilingual_lora.yaml) for a full example.

**Custom backbones** (plugins): implement the `Backbone` contract (`tokenizer`, `hidden_size`, `forward(ids, mask) -> [B, T, H]`, `save(dir)`), register it with the decorator, and list the module in the config. To support `data.parallel_options`, also set `supports_parallel = True` and make `forward` accept `position_ids` and `option_ids` (the built-in `hf` adapter already does).

```python
from opensysone.backbone import HFEncoder, register_backbone

@register_backbone("mean-pooled-hf")
class MeanPooledHF(HFEncoder):
    ...
```

```yaml
plugins: [examples.custom_backbone]
backbone: {name: mean-pooled-hf, model: answerdotai/ModernBERT-base, mix: 0.1}
```

Plugin module names are recorded in the model directory and imported automatically on load. See [`examples/custom_backbone.py`](examples/custom_backbone.py) for a complete example.

### Evaluation and calibration

```bash
opensysone eval --model runs/x/final --data test.jsonl --latency 100 --json report.json
opensysone eval --model runs/x/final --data test.jsonl --no-temperature              # results before calibration
opensysone calibrate --model runs/x/final --data another_dev.jsonl                   # refit temperatures on new data
opensysone calibrate --model runs/x/final --data another_dev.jsonl --by-options      # also fit per option-count bucket
```

The report gives accuracy, NLL, Brier and ECE (expected calibration error over 15 bins) per question type, plus MAE of the expected value for score questions. When setting thresholds, use the returned `probabilities` or `confidence`, and re-derive thresholds on your own data.

With option-count bucketing, only (type, bucket) pairs with at least 20 examples get their own temperature; the rest fall back to the per-type temperature. Worth trying if the number of options varies a lot across your questions (some have 2, some have 10); unnecessary if it is fixed.

### Inference and deployment

**Python**:

```python
from opensysone import Predictor

pred = Predictor.from_pretrained("runs/toy-tiny/final", device="auto")
result = pred.decide(request)                        # single
results = pred.decide_many(requests, batch_size=64)  # batch
```

**HTTP server**:

```bash
opensysone serve --model runs/toy-tiny/final --host 0.0.0.0 --port 8000 --max-batch 32 --max-wait-ms 5
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d @examples/request.json
```

```json
{"results": {
   "team":    {"type": "choice", "answer": "billing", "confidence": 0.93, "probabilities": {"billing": 0.93, "shipping": 0.04, "technical": 0.02, "account": 0.01}},
   "urgency": {"type": "score",  "answer": "4", "confidence": 0.61, "probabilities": {"1": 0.01, "2": 0.03, "3": 0.2, "4": 0.61, "5": 0.15}, "expected": 3.86},
   "refund":  {"type": "bool",   "answer": "yes", "confidence": 0.88, "probabilities": {"no": 0.12, "yes": 0.88}, "probability": 0.88}},
 "model": "toy-tiny/final", "latency_ms": 9.7}
```

Requests that arrive within `--max-wait-ms` of each other are merged into one inference batch. `GET /healthz` and `GET /v1/model` are also available.

**ONNX**:

```bash
opensysone export-onnx --model runs/toy-tiny/final --out runs/toy-tiny/onnx
opensysone predict --onnx --model runs/toy-tiny/onnx --input examples/request.json
opensysone serve   --onnx --model runs/toy-tiny/onnx
```

The exported graph outputs calibrated probabilities directly (including bucketed temperatures), with the tokenizer and metadata in the same directory. Export traces a real encoded example, so all [design switches](#design-switches) can be exported; tests cover the Laya-style combination and `parallel_options` (on a tiny BERT). onnxruntime prunes unused graph inputs, and `OnnxPredictor` feeds only the inputs the graph actually needs. Backbones using custom ops or flash-attention may need a different opset or attention implementation to export.

## Design switches

These switches borrow ideas from Laya's public code. **All are off by default, so default behavior is unchanged.** Model directories saved by older versions (0.1.0) still load as-is.

> [!IMPORTANT]
> There are **no ablation conclusions yet** for these switches (see [Ablations](#ablations)). Do not assume "on is better"; compare on your own data and backbone.

| config key | default | description |
|---|---|---|
| `data.order` | `state_first` | `question_first`: put the question and options first and the state last, so an overlong state is only truncated at the tail |
| `data.marker` | `opt` | `mask`: use the tokenizer's own `[MASK]` as the option marker instead of the added `[OPT]`; the backbone must have a mask token. The idea is that an MLM-pretrained encoder already gives `[MASK]` positions meaning; **not yet validated** |
| `data.bool_as_options` | `false` | `true`: write yes/no questions as two options `no: …` / `yes: …` and use the choice scorer, instead of the sigmoid head on `[Q]` |
| `data.parallel_options` | `false` | `true`: all options share the same position ids and cannot attend to each other, so option order no longer affects results (strict invariance verified in tests on a tiny BERT). Requires backbone support |
| `head.pooling` | `span` | `marker`: use only the hidden state at the option marker instead of averaging over the whole option span |
| `head.head_layers` | `0` | add N transformer layers on top of the backbone (Laya uses 2) |
| `head.type_embedding` | `false` | add a question-type embedding to every position |
| `head.scorer` | `per_type` | `shared`: one scorer shared by all question types, using only the option representation without question interaction features |
| `loss.name` | `ce` | `rlcd`: add a REINFORCE term on top of soft cross-entropy. Gaussian noise is added to the logits, the reward is "log score + spherical score − RPS", normalized within the group as the advantage; noise σ decays linearly from `sigma_start` to `sigma_end`. Also `rl_samples`, `w_sph`, `w_rps` |
| `train.calibrate_by_options` | `false` | additionally fit temperatures per (type, option-count bucket: 2 / 3-5 / 6-10 / 11+); buckets with too few examples fall back to the per-type temperature |

A combination that approximates Laya's public code. This is only our reading of its design, **not a reproduction of Laya**:

```yaml
head: {pooling: marker, head_layers: 2, type_embedding: true, scorer: shared}
data: {order: question_first, marker: mask, bool_as_options: true, shuffle_options: false}
loss: {name: rlcd}
train: {calibrate_by_options: true}
```

<details>
<summary><b>Implementation details</b></summary>

- Occurrences of the strings `[Q]`, `[OPT]` and `[MASK]` in input text are replaced with spaces first, so they cannot impersonate markers.
- For BERT-style backbones, `parallel_options` is implemented with a standard 4D attention mask and is tested. ModernBERT uses per-layer-type masks (requires transformers ≥ 5) and **has not been run on a real ModernBERT yet**.
- `bool_as_options` uses fixed English descriptions: `no: the statement does not hold` / `yes: the statement holds`.
- Switches are written to `opensysone.json` in the model directory and restored automatically at load and inference time.

</details>

## How it works

```
state ────┐
          ├─► [CLS] state [SEP] [Q] question [OPT] option1 [OPT] option2 … [SEP]
question ─┘                         │                │
                  bidirectional encoder (swappable: ModernBERT / mmBERT / XLM-R / DeBERTa / custom)
                                    │                │
                            h[Q] (bool head)   mean hidden state of option i span (choice / score heads)
                                    └──── one distribution per question ÷ temperature (calibrated per type)
```

**Serialization**: one sequence per question. The question and each option get a fixed token budget (`question_token_budget`, `option_token_budget`), and the state is truncated last, so long text never squeezes out the options. Exceeding `max_options` raises an error; with many labels, split the task into hierarchical questions (pick a coarse category first, then a fine one).

**Output heads** (default config):
- **Option representation** `h_opt`: the mean hidden state over each option span (the `[OPT]` marker plus the option's tokens), computed in one pass with prefix sums and directly exportable to ONNX. Early versions used only the `[OPT]` position; small backbones failed to link the marker to the option text that follows it, and choice accuracy stayed at chance. Switching to span averaging fixed this.
- **choice**: interaction features `[h_opt, h_q, h_opt·h_q]` built from `h_opt` and the `[Q]` hidden state `h_q`, passed through an MLP to a logit, then softmax. The scoring MLP shares weights across options, so the number of options can vary.
- **score**: the same features with a separate MLP, treating ordinal levels as options, and also outputting the expected value.
- **bool**: a single logit `z` from the `[Q]` and `[CLS]` hidden states, represented as `[0, z]`; softmax over it is equivalent to a sigmoid.

All three types are unified as "a distribution over some options", so loss, calibration, export and serving share one code path. During training, choice options are randomly shuffled so the model reads content rather than position.

**Loss**: soft-label cross-entropy, plus `rps_weight` × RPS for score questions and optionally `brier_weight` × Brier; question types can be weighted with `type_weights`.

### How Laya does it, and how to get there with switches

Laya (per its public code and docs) uses the same overall idea, with different choices at several points:

- **Backbone**: ModernBERT-large (English, 421M total) or mmBERT-base (multilingual, 322M total), fully fine-tuned.
- **Layout**: question and options come first, the state last; each option is marked with the tokenizer's own `[MASK]` token, and yes/no questions (`noul`) are written as two options rather than using a separate sigmoid head.
- **Head**: a question-type embedding added to every token, two extra transformer layers on top of the encoder, and one small MLP shared across types that turns the vector at each option's `[MASK]` position into a logit. Laya also has an act head that reads the pooled vector plus summary features of the answer distribution; OpenSysOne does not implement it.
- **Training**: RLCD, i.e. soft cross-entropy plus a REINFORCE term on Gaussian-perturbed logits whose reward is a strictly proper score (log + spherical − RPS).
- **Calibration**: temperatures that also depend on the number of options.

Each of these maps to a [design switch](#design-switches), so you can move from the OpenSysOne default to a Laya-like setup one piece at a time:

| Laya choice | OpenSysOne default | switch |
|---|---|---|
| question first, state last | state first | `data.order: question_first` |
| `[MASK]` as option marker | added `[OPT]` token | `data.marker: mask` |
| yes/no as two options | sigmoid head on `[Q]` | `data.bool_as_options: true` |
| vector at the marker position | mean over the option span | `head.pooling: marker` |
| 2 extra transformer layers | none | `head.head_layers: 2` |
| question-type embedding | none | `head.type_embedding: true` |
| one scorer for all types | per-type scorers with question interaction | `head.scorer: shared` |
| RLCD loss | soft CE + RPS | `loss.name: rlcd` |
| temperature per option count | temperature per type | `train.calibrate_by_options: true` |
| ModernBERT-large / mmBERT-base | any encoder | `backbone.model: answerdotai/ModernBERT-large` |

All switches default to off and can be combined freely; the full Laya-like combination is shown under [Design switches](#design-switches). This is our reading of Laya's design, not a reproduction: no official Laya weights are included, the act head is missing, and whether each switch helps is still open (see [Ablations](#ablations)).

## Ablations

The repo includes a harder bilingual routing dataset and an ablation script for comparing [design switches](#design-switches) one by one on the same backbone and data.

- **[`examples/routing/make_data.py`](examples/routing/make_data.py)**: 8 actions × 8 objects = 64 intents, with messages paraphrased in Chinese and English and English intent names as options. 16 of the intents **never appear in the training or validation sets** (not even as distractors), and `test_unseen` asks only about these 16, to test whether the model actually reads option text instead of memorizing labels. Each message has three questions: intent (choice, 3–8 options), urgency (1–5 score), and whether a given object is mentioned (bool).
- **[`scripts/ablation.py`](scripts/ablation.py)**: 12 variants, including the default config, marker-only pooling, a 2-layer head, a Laya-style head, question first, `[MASK]` marker, bool as options, no option shuffling, parallel layout, RLCD, and `laya_like` combining everything. It reports accuracy and NLL on seen and unseen intents, ECE per type and per option-count bucket, whether answers stay the same after reordering options, and latency. Completed variants are skipped, so interrupted runs can resume.

```bash
python examples/routing/make_data.py
opensysone make-tiny-backbone --corpus examples/routing/data/train.jsonl --out runs/routing-backbone \
    --hidden-size 128 --layers 4 --heads 2 --max-position 256
python scripts/ablation.py --seeds 0 1 2 --epochs 20 --out runs/ablation   # about 3 hours per seed on a 2-core CPU
```

**Current status**: only one full 25-epoch run of the default config has been completed, reaching about 97% on intent and about 99% on urgency on the validation set; bool is the hardest at only about 66%. The full 12-variant comparison **has no results yet**, so no claims are made here about which switch is better. These experiments also use only a randomly initialized tiny BERT, so any potential advantage of `[MASK]` on pretrained models cannot show up. Trustworthy conclusions need multi-seed runs on real pretrained encoders. Contributions of results are welcome!

## Related projects

| project | backbone | notes |
|---|---|---|
| [Laya](https://github.com/NandhaKishorM/laya) | ModernBERT / mmBERT encoders | official implementation, with the RLCD training package and three sets of weights |
| [Mapika/decider](https://github.com/Mapika/decider) | decoder LLMs such as Qwen and Gemma | choice / score / bool, with calibration, serving and GGUF; training script accepts a model choice |
| [jevlike](https://github.com/vinnylarouge/jevlike) | byte embeddings learned from scratch, or a frozen HF encoder | minimal ~800-line implementation, choice only |
| [laya-opencv](https://github.com/uu889/laya-opencv/wiki) | any HF encoder | local workbench with a web UI |
| [NanoJev](https://github.com/TianyuCodings/NanoJev) | Qwen3-0.6B with a decision head | focused on game tasks |
| [Kev](https://github.com/jaredpalmer/kev), [Nimble](https://github.com/bespokelabsai/nimble) | Qwen3.5 with LoRA | decoder approach, a few fixed sizes |
| [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) | any open language model | no training; reads option logits directly |

OpenSysOne focuses on: **the encoder approach, one model handling all three question types, backbones swapped via config or plugins, with LoRA, calibration and ONNX export**. It overlaps significantly with Laya's own training package; if you only need Laya's official weights, just use Laya.

## Project structure

```
opensysone/
├── opensysone/
│   ├── schema.py        data format, label parsing
│   ├── serializer.py    state + question + options → token sequence, marker positions, parallel layout
│   ├── backbone.py      backbone registry, HF adapter, LoRA, parallel option masks
│   ├── model.py         output heads (configurable), losses (CE+RPS / RLCD), save and load
│   ├── data.py          dataset (option shuffling augmentation) and collate
│   ├── trainer.py       training loop (LLRD, AMP, early stopping, calibration)
│   ├── calibrate.py     temperature scaling per type, optionally per option-count bucket
│   ├── metrics.py       accuracy / NLL / Brier / ECE / latency
│   ├── predictor.py     inference and response format
│   ├── serve.py         FastAPI server with micro-batching
│   ├── onnx_export.py   ONNX export and onnxruntime inference
│   ├── tiny.py          build a random tiny BERT from a corpus (for testing)
│   └── cli.py           command-line entry point
├── configs/             example configs (tiny / ModernBERT / multilingual LoRA)
├── examples/            data generators (toy_tickets, routing), sample request, custom backbone plugin
├── scripts/             ablation.py: design switch ablations
└── tests/               unit and end-to-end tests
```

## Development and contributing

```bash
pip install -e ".[dev]"
pytest -q          # 36 tests, about 20 seconds on CPU, no network needed
```

The tests generate small datasets, build random tiny backbones, run the full training, calibration, inference, HTTP server and ONNX export flow, and compare ONNX against PyTorch outputs. `tests/test_variants.py` also covers the layout combinations, order invariance of the parallel layout (while also checking that the default layout does **not** have this property, so the test is not vacuous), the gradient direction of RLCD, save and load of bucketed temperatures, and training and ONNX consistency under switch combinations.

Issues and PRs are welcome. Contributions we would especially like:
- training results and recommended configs on real pretrained backbones;
- multi-seed results from [`scripts/ablation.py`](scripts/ablation.py);
- new backbone adapters (especially validation of the parallel layout on ModernBERT).

Please make sure `pytest -q` passes before opening a PR. If a change affects metrics, include before and after numbers.

## Limitations

- Decision models suit narrow, well-defined tasks and have weak zero-shot ability; fine-tune and evaluate on your own data before going to production.
- Context length is limited by the backbone; long documents need to be chunked or summarized first.
- With many options, either raise `max_options` and `max_length`, or split into hierarchical questions.
- **Not yet run on real pretrained encoders (ModernBERT, XLM-R, mmBERT, etc.).** Hyperparameters in the example configs are educated guesses and unvalidated.
- The randomly initialized tiny backbones are only for testing the pipeline and their metrics mean nothing; both bundled datasets are template-generated and do not reflect the difficulty of real tasks.
- The [design switches](#design-switches) have no ablation conclusions yet; the `parallel_options` implementation for ModernBERT is untested.

## Roadmap

- [ ] Validate configs on real pretrained backbones and publish multi-seed ablation results (script already exists)
- [ ] Teacher labeling tool (call an LLM to generate soft labels)
- [ ] Share one encoding across all questions of the same example to avoid redundant computation
- [ ] Hierarchical labels and very large option sets
- [ ] Multi-GPU training (DDP / accelerate)
- [ ] Dynamic quantization (int8) for inference

## Citation

If OpenSysOne helps your work, you can cite it as:

```bibtex
@software{opensysone,
  title  = {OpenSysOne: Training System-1 Decision Models on Swappable Encoder Backbones},
  author = {{OpenSysOne contributors}},
  year   = {2026},
  url    = {https://github.com/xillkey/OpenSysOne}
}
```

## License

[Apache License 2.0](LICENSE)

This project is not affiliated with Convai Innovations (Laya) or TypeSafe (Jev). Project names and trademarks mentioned here belong to their respective owners.
