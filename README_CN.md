<div align="center">

# OpenSysOne

**在任意编码器上训练 System-1 决策模型**

给模型一段素材和几道类型化的题，一次前向传播就得到每道题的校准概率。不生成文本，不需要解析，也不会编造答案。

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#局限)

[快速开始](#快速开始) · [使用指南](#使用指南) · [设计开关](#设计开关) · [工作原理](#工作原理) · [对照实验](#对照实验) · [路线图](#路线图)

</div>

---

*Train Laya/Jev-style System-1 decision models on any swappable encoder backbone: one forward pass, typed questions (choice / score / bool), calibrated probabilities, a `/v1/systemone` server and ONNX export.*

## 简介

很多 Agent 和业务流程里的决定其实是"选择题"：工单交给哪个团队、这条消息有多紧急、客户是不是要退款。用大语言模型生成答案再去解析，既慢又可能出格。**决策模型**（System-1 model）换了一种做法：把题目和选项一起交给双向编码器，一次前向传播直接输出每个选项的概率。

OpenSysOne 是一个训练这类模型的框架。它的思路与 Laya、Jev 相同，重点在于**骨干网络可以随意替换**：从 ModernBERT、XLM-R 到你自己的编码器，都只是改一行配置。

```python
from opensysone import Predictor

pred = Predictor.from_pretrained("runs/toy-tiny/final")   # 快速开始里训练的玩具模型
pred.decide({
    "state": {"channel": "email", "lang": "zh", "message": "客服你好，我的订阅被重复扣费了，比较急，请今天回复。我要求退款。"},
    "questions": {
        "team":    {"type": "choice", "question": "这张工单应该交给哪个团队？", "options": ["billing", "shipping", "technical", "account"]},
        "urgency": {"type": "score",  "question": "这张工单有多紧急？", "min": 1, "max": 5},
        "refund":  {"type": "bool",   "question": "客户是否要求退款？"},
    },
})
# {"results": {"team":    {"answer": "billing", "confidence": 1.0,  "probabilities": {...}},
#              "urgency": {"answer": "4", "confidence": 0.93, "expected": 3.93, ...},
#              "refund":  {"answer": "yes", "probability": 1.0, ...}},
#  "latency_ms": 6.5}
```

> OpenSysOne 是独立实现，与 Laya（Convai Innovations）和 Jev（TypeSafe）没有关联，不包含它们的代码或权重。"Jev 风格接口"指请求结构相似（state + questions），字段细节请以你要对接的系统为准。

## 特性

- 🔌 **骨干可替换**：Hugging Face 上任何双向编码器改一行配置即可使用，也可以通过注册表插入自定义骨干。
- 🧩 **三种题型，一个模型**：`choice`（单选）、`score`（有序打分）、`bool`（是非）。选项在请求时才给出，换一套选项不用重新训练。
- 🎯 **校准过的概率**：训练后在验证集上按题型拟合温度，可选再按选项数分桶，置信度可以直接用来设阈值。
- 📐 **严格适当评分规则**：对数损失，加上打分题的 Ranked Probability Score，可选 Brier；也可以换成 RLCD 损失。
- 🏷️ **软标签**：标签可以是概率分布，方便用大模型当教师做蒸馏。
- 🎛️ **可切换的设计**：题目位置、选项标记、选项表示方式、并行布局（与选项顺序无关）、头部结构等都是配置开关，方便做对照实验。
- 💾 **小显存可用**：LoRA、梯度检查点、混合精度；没有 GPU 也能在 CPU 上训小模型。
- 🚀 **直接部署**：带请求微批处理的 `POST /v1/systemone` 服务，以及 ONNX 导出与 onnxruntime 推理。
- 🧪 **离线可测**：能从你的数据生成一个随机初始化的小 BERT，不联网也能跑通完整流程。

## 安装

```bash
git clone https://github.com/xillkey/OpenSysOne.git
cd opensysone
pip install -e ".[all]"
```

按需安装可选依赖：

| extra | 内容 |
|---|---|
| （无） | 训练、评估、推理 |
| `lora` | LoRA 微调（peft） |
| `serve` | HTTP 服务（FastAPI、uvicorn） |
| `onnx` | ONNX 导出与推理（onnx、onnxruntime） |
| `dev` | 以上全部，加上 pytest |
| `all` | lora + serve + onnx |

需要 Python ≥ 3.9、PyTorch ≥ 2.1、transformers ≥ 4.40；ModernBERT 需要 transformers ≥ 4.48。已在 Python 3.13、PyTorch 2.14、transformers 5.20、peft 0.21 上测试。

## 快速开始

下面的流程在 CPU 上就能跑，不需要下载任何预训练模型：

```bash
# 1. 生成中英双语的工单示例数据
python examples/toy_tickets/make_data.py

# 2. 用这份数据造一个随机初始化的小 BERT（只用于跑通流程）
opensysone make-tiny-backbone --corpus examples/toy_tickets/data/train.jsonl --out runs/tiny-backbone

# 3. 训练；结束时自动在验证集上校准，最终模型在 runs/toy-tiny/final
opensysone train -c configs/toy_tiny.yaml

# 4. 评估，并测单条请求的延迟
opensysone eval --model runs/toy-tiny/final --data examples/toy_tickets/data/test.jsonl --latency 50

# 5. 推理一条请求
opensysone predict --model runs/toy-tiny/final --input examples/request.json

# 6. 启动服务
opensysone serve --model runs/toy-tiny/final --port 8000
```

第 3 步在 2 核 CPU 上约 5 分钟。玩具数据 test 集上的参考结果（校准后）：

| 题型 | accuracy | NLL | ECE | 打分 MAE |
|---|---|---|---|---|
| choice | 1.000 | 0.000 | 0.000 | – |
| score | 0.987 | 0.105 | 0.005 | 0.028 |
| bool | 1.000 | 0.000 | 0.000 | – |

单条请求（3 道题）的 CPU 延迟 p50 约 7 ms。

> 这份数据是模板生成的，上面的数字只说明流程跑通了，不代表真实任务的效果。

换成真正的预训练骨干只需要换配置，例如 `opensysone train -c configs/modernbert_base.yaml`。

## 使用指南

### 数据格式

JSONL，每行一个样本。`state` 可以是文本，也可以是 JSON 对象（会被序列化成键排序后的 JSON 字符串）。

```json
{"id": "t00001",
 "state": {"channel": "email", "message": "我的订阅被重复扣费了，请今天回复。我要求退款。"},
 "questions": {
   "team":    {"type": "choice", "text": "交给哪个团队？", "options": ["billing", "shipping", "technical", "account"], "label": "billing"},
   "urgency": {"type": "score",  "text": "有多紧急？", "min": 1, "max": 5, "label": 4},
   "refund":  {"type": "bool",   "text": "是否要求退款？", "label": true}}}
```

| 字段 | 说明 |
|---|---|
| `type` | `choice` / `score` / `bool`，也接受 `rating`、`scale`、`yesno`、`noul` 等别名 |
| `text` | 题目，也可以写作 `question` |
| `options` | `choice` 必填；`score` 可以用 `options` 自定义档位，或用 `min` / `max` 生成整数档位；`bool` 不需要 |
| `label` | 硬标签：选项名、选项下标、`true` / `false` / `"yes"`；软标签：概率列表或 `{"选项": 概率}`；`bool` 还可以直接给 0–1 的概率 |

`questions` 也可以写成带 `id` 字段的列表。没有 `label` 的题在训练时会被跳过。用下面的命令检查数据：

```bash
opensysone validate-data --data your_train.jsonl
```

### 训练

```bash
opensysone train -c configs/modernbert_base.yaml
opensysone train -c configs/modernbert_base.yaml --set train.lr=2e-5 --set data.max_length=256   # 命令行覆盖任意配置
```

输出目录里有：
- `best/`：验证集 NLL 最好的一步，未校准。
- `final/`：校准后的最终模型，推理和部署用这个。
- `metrics.json`：训练过程和最终验证集指标。

每个模型目录包含 `backbone/`（权重和分词器）、`head.safetensors` 和 `opensysone.json`（序列化参数、温度、训练配置）。

<details>
<summary><b>常用训练配置（<code>train:</code> 段）</b></summary>

| 键 | 默认 | 说明 |
|---|---|---|
| `lr` / `head_lr` | 3e-5 / 1e-3 | 骨干和输出头的学习率 |
| `llrd` | 0.9 | 分层学习率衰减，越靠近底层学习率越小；1.0 表示关闭 |
| `epochs` / `max_steps` | 3 / 0 | `max_steps > 0` 时覆盖轮数 |
| `batch_size` / `grad_accum` | 16 / 1 | |
| `precision` | auto | GPU 上自动选 bf16 或 fp16，CPU 用 fp32 |
| `eval_every` | 0 | 每多少步评估一次，0 表示每轮一次 |
| `early_stop_patience` | 3 | 验证集 NLL 连续多少次不下降就停止 |
| `calibrate` | true | 训练结束后在验证集上拟合温度 |
| `calibrate_by_options` | false | 另外按选项数分桶拟合温度，见[设计开关](#设计开关) |

</details>

### 更换骨干网络

**Hugging Face 编码器**（最常用）：

```yaml
backbone:
  name: hf
  model: answerdotai/ModernBERT-base     # 换成任何双向编码器
  gradient_checkpointing: false
  trust_remote_code: false
  attn_implementation: null               # 例如 flash_attention_2
```

框架会自动加入 `[Q]`、`[OPT]` 两个特殊标记，扩展词表并初始化它们的嵌入。

| 场景 | 可选骨干（示例） |
|---|---|
| 英文、低延迟 | `answerdotai/ModernBERT-base` / `-large` |
| 多语言 | `jhu-clsp/mmBERT-base`、`FacebookAI/xlm-roberta-base`、`intfloat/multilingual-e5-base` |
| 中文为主 | `hfl/chinese-roberta-wwm-ext`、`BAAI/bge-base-zh-v1.5` |
| 英文、精度优先 | `microsoft/deberta-v3-base` / `-large` |

`hf` 适配器不支持纯解码器（GPT 类）和编码-解码模型（T5 类）；后者可以写一个包装 `T5EncoderModel` 的自定义骨干。

> **换骨干前先检查分词器。** 如果目标语言大量变成 `[UNK]`，模型会"自信地答错"。用 `tokenizer.tokenize("你的样例文本")` 就能快速确认。

**LoRA**（小显存）：

```yaml
backbone:
  name: hf
  model: FacebookAI/xlm-roberta-base
  gradient_checkpointing: true
  train_embeddings: true          # 让新加的 [Q]/[OPT] 嵌入也能学
  lora: {r: 16, alpha: 32, dropout: 0.05, target_modules: all-linear}
```

保存时 LoRA 会被合并进权重，导出的模型和全量微调的格式完全一样。完整示例见 [`configs/multilingual_lora.yaml`](configs/multilingual_lora.yaml)。

**自定义骨干**（插件）：实现 `Backbone` 约定（`tokenizer`、`hidden_size`、`forward(ids, mask) -> [B, T, H]`、`save(dir)`），用装饰器注册，再在配置里列出模块。要支持 `data.parallel_options`，还需要设置 `supports_parallel = True`，并让 `forward` 接受 `position_ids` 和 `option_ids`（内置的 `hf` 适配器已经实现）。

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

插件模块名会写进模型目录，加载时自动导入。完整例子见 [`examples/custom_backbone.py`](examples/custom_backbone.py)。

### 评估与校准

```bash
opensysone eval --model runs/x/final --data test.jsonl --latency 100 --json report.json
opensysone eval --model runs/x/final --data test.jsonl --no-temperature              # 看校准前的效果
opensysone calibrate --model runs/x/final --data another_dev.jsonl                   # 在新数据上重新拟合温度
opensysone calibrate --model runs/x/final --data another_dev.jsonl --by-options      # 另外按选项数分桶拟合
```

报告按题型给出 accuracy、NLL、Brier、ECE（15 个分箱的期望校准误差），打分题额外给出期望值的 MAE。设阈值时请用返回的 `probabilities` 或 `confidence`，并在你自己的数据上重新确定阈值。

按选项数分桶时，只有样本数不少于 20 的（题型, 桶）会单独拟合，其余回退到该题型的温度。如果你的任务里选项数差别很大（有的题 2 个选项，有的 10 个），可以试试；选项数固定的任务没必要。

### 推理与部署

**Python**：

```python
from opensysone import Predictor

pred = Predictor.from_pretrained("runs/toy-tiny/final", device="auto")
result = pred.decide(request)                        # 单条
results = pred.decide_many(requests, batch_size=64)  # 批量
```

**HTTP 服务**：

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

在 `--max-wait-ms` 内同时到达的请求会合并成一个批次推理。另外提供 `GET /healthz` 和 `GET /v1/model`。

**ONNX**：

```bash
opensysone export-onnx --model runs/toy-tiny/final --out runs/toy-tiny/onnx
opensysone predict --onnx --model runs/toy-tiny/onnx --input examples/request.json
opensysone serve   --onnx --model runs/toy-tiny/onnx
```

导出的图直接输出校准后的概率（包括分桶温度），分词器和元数据放在同一个目录里。导出时用一条真实编码出的样例来追踪，所以各种[设计开关](#设计开关)都能导出；测试覆盖了 Laya 式组合和 `parallel_options`（用的是小 BERT）。onnxruntime 会剪掉图里用不到的输入，`OnnxPredictor` 只喂图实际需要的输入。用了自定义算子或 flash-attention 的骨干，可能需要调整 opset 或注意力实现才能导出。

## 设计开关

下面这些开关借鉴了 Laya 公开代码里的做法，**全部默认关闭，默认行为不变**。旧版本（0.1.0）保存的模型目录仍然可以直接加载。

> [!IMPORTANT]
> 这些开关的效果**还没有对照实验的结论**（见[对照实验](#对照实验)）。不要默认认为"打开更好"，请在自己的数据和骨干上比较。

| 配置键 | 默认 | 说明 |
|---|---|---|
| `data.order` | `state_first` | `question_first`：把题目和选项放在前面、state 放最后，过长的 state 只截断尾部 |
| `data.marker` | `opt` | `mask`：用分词器自带的 `[MASK]` 做选项标记，不依赖新增的 `[OPT]`；骨干需要有 mask token。设想是 MLM 预训练的编码器对 `[MASK]` 位置本来就有意义，**尚未验证** |
| `data.bool_as_options` | `false` | `true`：把是非题写成 `no: …` / `yes: …` 两个选项，用单选的打分器，而不是 `[Q]` 上的 sigmoid 头 |
| `data.parallel_options` | `false` | `true`：所有选项共用同一段位置编号，并禁止选项之间互相注意，选项的排列顺序不再影响结果（测试里在小 BERT 上验证了严格不变）。需要骨干支持 |
| `head.pooling` | `span` | `marker`：只取选项标记那一个位置的隐状态，不对整段选项求平均 |
| `head.head_layers` | `0` | 在骨干之上再加 N 层 transformer（Laya 用 2 层） |
| `head.type_embedding` | `false` | 给每个位置加一个题型嵌入 |
| `head.scorer` | `per_type` | `shared`：所有题型共用一个打分器，只看选项表示，不用题目交互特征 |
| `loss.name` | `ce` | `rlcd`：在软交叉熵之外加 REINFORCE 项。对 logits 加高斯噪声，以"对数得分 + 球面得分 − RPS"为奖励，组内归一化后作优势；噪声 σ 从 `sigma_start` 线性降到 `sigma_end`。另有 `rl_samples`、`w_sph`、`w_rps` |
| `train.calibrate_by_options` | `false` | 额外按（题型, 选项数分桶：2 / 3-5 / 6-10 / 11+）拟合温度，样本不足的桶回退到题型温度 |

一个按 Laya 公开代码"近似"的组合。这只是对其设计的理解，**不是 Laya 的复现**：

```yaml
head: {pooling: marker, head_layers: 2, type_embedding: true, scorer: shared}
data: {order: question_first, marker: mask, bool_as_options: true, shuffle_options: false}
loss: {name: rlcd}
train: {calibrate_by_options: true}
```

<details>
<summary><b>实现细节</b></summary>

- 文本里如果出现 `[Q]`、`[OPT]`、`[MASK]` 这些字符串，会先被替换成空格，避免冒充标记。
- `parallel_options` 对 BERT 类骨干用普通的 4D 注意力掩码实现，已经测过。ModernBERT 走的是按层类型分别给掩码的写法（需要 transformers ≥ 5），**还没有在真实的 ModernBERT 上跑过**。
- `bool_as_options` 使用固定的英文描述：`no: the statement does not hold` / `yes: the statement holds`。
- 开关会写进模型目录的 `opensysone.json`，加载和推理时自动还原。

</details>

## 工作原理

```
state ──┐
        ├─► [CLS] state [SEP] [Q] 题目 [OPT] 选项1 [OPT] 选项2 … [SEP]
题目 ───┘                         │                │
                          双向编码器（可替换：ModernBERT / mmBERT / XLM-R / DeBERTa / 自定义）
                                  │                │
                              h[Q] (bool 头)   选项 i 片段的平均隐状态 (choice / score 头)
                                  └──── 每题一个概率分布 ÷ 温度（按题型校准）
```

**序列化**：每道题一条序列。题目和每个选项各有固定的 token 预算（`question_token_budget`、`option_token_budget`），state 最后截断，所以长文本不会把选项挤掉。选项数超过 `max_options` 会报错；标签很多时，建议拆成层级题（先选大类，再选小类）。

**输出头**（默认配置）：
- **选项表示** `h_opt`：每个选项片段（`[OPT]` 标记加上该选项的 token）上隐状态的平均值，用前缀和一次算完，可以直接导出 ONNX。早期版本只取 `[OPT]` 那一个位置，小骨干学不会把标记和后面的选项文字对应起来，单选准确率停在随机水平；改成片段平均后问题消失。
- **choice**：对 `h_opt` 和 `[Q]` 的隐状态 `h_q` 构造交互特征 `[h_opt, h_q, h_opt·h_q]`，经 MLP 得到 logit，再做 softmax。打分 MLP 在选项之间共享权重，所以选项数量可变。
- **score**：同样的特征、独立的 MLP，把有序档位当作选项，并输出期望值。
- **bool**：用 `[Q]` 和 `[CLS]` 的隐状态得到一个 logit `z`，表示为 `[0, z]`，softmax 后等价于 sigmoid。

三种题型统一成"在若干选项上的分布"，所以损失、校准、导出和服务共用一套代码。训练时会随机打乱单选题的选项顺序，让模型看内容而不是看位置。

**损失**：软标签交叉熵，加上 `rps_weight` × 打分题的 RPS，可选 `brier_weight` × Brier；题型之间可以用 `type_weights` 加权。

## 对照实验

仓库带了一个更难的双语路由数据集和一个对照脚本，用来在同一个骨干、同一份数据上逐项比较[设计开关](#设计开关)。

- **[`examples/routing/make_data.py`](examples/routing/make_data.py)**：8 个动作 × 8 个对象 = 64 种意图，消息用中英文同义改写，选项是英文意图名。其中 16 种意图**完全不出现在训练集和验证集里**（包括不当干扰项），`test_unseen` 只问这 16 种，用来检验模型是否真的读懂了选项文字，而不是记住了标签。每条消息有三道题：意图（单选，3–8 个选项）、紧急度（1–5 打分）、是否提到某个对象（是非）。
- **[`scripts/ablation.py`](scripts/ablation.py)**：12 个变体，包括默认配置、只取标记位置、加 2 层头、Laya 式头、题目在前、`[MASK]` 标记、是非题当选项、不打乱选项、并行布局、RLCD，以及全部组合的 `laya_like`。输出已见和未见意图的准确率、NLL、按题型和按选项数分桶的 ECE、换选项顺序后答案是否一致，以及延迟。已完成的变体会被跳过，中断后可以接着跑。

```bash
python examples/routing/make_data.py
opensysone make-tiny-backbone --corpus examples/routing/data/train.jsonl --out runs/routing-backbone \
    --hidden-size 128 --layers 4 --heads 2 --max-position 256
python scripts/ablation.py --seeds 0 1 2 --epochs 20 --out runs/ablation   # 2 核 CPU 上每个种子约 3 小时
```

**当前状态**：只完整跑过默认配置的一次 25 轮训练，验证集上意图约 97%、紧急度约 99%，是非题最难，只到约 66%。12 个变体的完整对照**还没有结果**，因此这里不给出"哪个开关更好"的结论。此外，上述实验只用了随机初始化的小 BERT，`[MASK]` 在预训练模型上的潜在优势无法体现。可信的结论需要在真实预训练编码器上多种子重复。欢迎贡献结果！

## 与相关项目的关系



| 项目 | 骨干 | 说明 |
|---|---|---|
| [Laya](https://github.com/NandhaKishorM/laya) | ModernBERT / mmBERT 编码器 | 官方实现，带 RLCD 训练包和三套权重 |
| [Mapika/decider](https://github.com/Mapika/decider) | Qwen、Gemma 等解码器大模型 | 单选 / 打分 / 是非，带校准、服务、GGUF，训练脚本可指定模型 |
| [jevlike](https://github.com/vinnylarouge/jevlike) | 从零学的字节嵌入，或冻结的 HF 编码器 | 约 800 行的最小实现，只有单选 |
| [laya-opencv](https://github.com/uu889/laya-opencv/wiki) | 任意 HF 编码器 | 带网页界面的本地工作台 |
| [NanoJev](https://github.com/TianyuCodings/NanoJev) | Qwen3-0.6B 加决策头 | 以游戏任务为主 |
| [Kev](https://github.com/jaredpalmer/kev)、[Nimble](https://github.com/bespokelabsai/nimble) | Qwen3.5 加 LoRA | 解码器路线，固定的几个尺寸 |
| [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) | 任意开源语言模型 | 不训练，直接读选项 logit |

OpenSysOne 关注的是：**编码器路线、一个模型同时处理三种题型、骨干靠配置或插件替换，并且带 LoRA、校准和 ONNX 导出**。它与 Laya 自带的训练包有明显重叠；如果你只需要 Laya 的官方权重，直接用 Laya 就好。

## 项目结构

```
opensysone/
├── opensysone/
│   ├── schema.py        数据格式、标签解析
│   ├── serializer.py    state + 题目 + 选项 → token 序列、标记位置、并行布局
│   ├── backbone.py      骨干注册表、HF 适配器、LoRA、并行选项掩码
│   ├── model.py         输出头（可配置）、损失（CE+RPS / RLCD）、保存与加载
│   ├── data.py          数据集（选项乱序增广）与 collate
│   ├── trainer.py       训练循环（LLRD、AMP、早停、校准）
│   ├── calibrate.py     按题型、可选按选项数分桶的温度缩放
│   ├── metrics.py       accuracy / NLL / Brier / ECE / 延迟
│   ├── predictor.py     推理与响应格式
│   ├── serve.py         FastAPI 服务与微批处理
│   ├── onnx_export.py   ONNX 导出与 onnxruntime 推理
│   ├── tiny.py          从语料构建随机小 BERT（测试用）
│   └── cli.py           命令行入口
├── configs/             示例配置（tiny / ModernBERT / 多语言 LoRA）
├── examples/            示例数据生成器（toy_tickets、routing）、请求样例、自定义骨干插件
├── scripts/             ablation.py：设计开关的对照实验
└── tests/               单元测试与端到端测试
```

## 开发与贡献

```bash
pip install -e ".[dev]"
pytest -q          # 36 个测试，CPU 上约 20 秒，不需要联网
```

测试会生成小数据集、构建随机小骨干，完整跑一遍训练、校准、推理、HTTP 服务和 ONNX 导出，并对比 ONNX 与 PyTorch 的输出。`tests/test_variants.py` 还覆盖了各种布局组合、并行布局的顺序不变性（同时检查默认布局确实**不**具备这一性质，以免测试形同虚设）、RLCD 的梯度方向、分桶温度的保存与加载，以及开关组合下的训练和 ONNX 一致性。

欢迎提 Issue 和 PR。特别欢迎这些贡献：
- 在真实预训练骨干上的训练结果和推荐配置；
- [`scripts/ablation.py`](scripts/ablation.py) 的多种子对照结果；
- 新的骨干适配器（尤其是 ModernBERT 并行布局的验证）。

提交 PR 前请确保 `pytest -q` 全部通过。改动影响指标时，请附上改动前后的数字。

## 局限

- 判定模型适合窄而明确的任务，零样本能力弱，上线前应在自己的数据上微调和评估。
- 上下文长度受骨干限制，长文档需要先切分或摘要。
- 选项很多时，要么提高 `max_options` 和 `max_length`，要么拆成层级题。
- **还没有在真实的预训练编码器（ModernBERT、XLM-R、mmBERT 等）上跑过。** 示例配置里的超参数是按经验写的，未经验证。
- 随机初始化的小骨干只用于测试流程，指标没有参考意义；自带的两个数据集都是模板生成的，不代表真实任务的难度。
- [设计开关](#设计开关)还没有对照结论；`parallel_options` 在 ModernBERT 上的实现未经测试。

## 路线图

- [ ] 在真实预训练骨干上验证配置，并公开多种子对照结果（脚本已有）
- [ ] 教师模型标注工具（调用大模型生成软标签）
- [ ] 同一样本的多道题共享一次编码，减少重复计算
- [ ] 层级标签与超大选项集
- [ ] 多 GPU 训练（DDP / accelerate）
- [ ] 推理侧动态量化（int8）

## 引用

如果 OpenSysOne 对你的工作有帮助，可以这样引用：

```bibtex
@software{opensysone,
  title  = {OpenSysOne: Training System-1 Decision Models on Swappable Encoder Backbones},
  author = {{OpenSysOne contributors}},
  year   = {2026},
  url    = {https://github.com/xillkey/OpenSysOne}
}
```

## 许可证

[Apache License 2.0](LICENSE)

本项目与 Convai Innovations（Laya）和 TypeSafe（Jev）没有关联。文中提到的项目名称和商标归各自所有者所有。