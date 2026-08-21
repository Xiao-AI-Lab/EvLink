<div align="center">

# EvLink: Source-grounded Evidence Linking for Graph RAG

<p>
  <img alt="EMNLP 2026 Main Conference" src="https://img.shields.io/badge/EMNLP_2026-Main_Conference-B31B1B?style=flat-square">
  <a href="https://github.com/Xiao-AI-Lab/EvLink"><img alt="GitHub 仓库" src="https://img.shields.io/badge/GitHub-EvLink-181717?style=flat-square&logo=github&logoColor=white"></a>
  <a href="https://github.com/Xiao-AI-Lab/EvLink/actions/workflows/tests.yml"><img alt="测试" src="https://github.com/Xiao-AI-Lab/EvLink/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://github.com/Xiao-AI-Lab/EvLink/actions/workflows/package.yml"><img alt="打包" src="https://github.com/Xiao-AI-Lab/EvLink/actions/workflows/package.yml/badge.svg"></a>
</p>
<p>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-2F855A?style=flat-square"></a>
  <img alt="Graph RAG" src="https://img.shields.io/badge/Graph_RAG-source--grounded-0F766E?style=flat-square">
  <a href="reproduce/README.md"><img alt="复现协议" src="https://img.shields.io/badge/reproduction-protocol-7C3AED?style=flat-square"></a>
</p>

<p><strong>已被 EMNLP 2026 Main Conference 接收</strong></p>
<p>Linyao Zheng · Xuhang Shi · Zhifang Mao · Sai Zhou · Shuaixian An · Xiuquan Hou</p>

<p><a href="README.md">English</a> | <strong>简体中文</strong></p>

[快速开始](#-快速开始) · [检索器接入](#-检索器接入) · [论文复现](#-论文复现) · [产物契约](#-产物契约) · [引用与联系](#-引用与联系)

</div>

EvLink 是一种面向 Graph RAG、基于来源支撑证据链接的图检索器。它既可以
作为端到端研究流水线运行，也可以从现有检索器产生的候选结果中选出紧凑、
满足固定预算的证据集合。

<p align="center">
  <img src="assets/evlink-overview.png" width="100%" alt="EvLink 方法概览">
</p>
<p align="center"><em>EvLink 构建由来源支撑的证据链接，归纳查询局部的证据区域，并通过证据需求覆盖选出紧凑的支持证据。</em></p>

> [!NOTE]
> Python 发行包与导入包均命名为 `evidencelink`。

## ✨ 核心特点

- 🔗 **来源支撑的链接**为图遍历保留段落级见证信息；
- 🎯 **覆盖感知选择**围绕问题的证据需求，在固定 reader 预算内组织证据；
- 🔌 **检索器接入**接收稠密、稀疏或图检索器输出的有序候选，无需用户替换
  现有技术栈。

本包提供明确的产物结构（schema）、确定性的端到端冒烟示例，以及可检查的
选择轨迹。

随附的冒烟示例有意保持小规模和确定性；将示例输入替换为对应的已准备产物，
即可运行基准评测。

## 🧭 流水线

公开流水线从头到尾沿用论文术语：

```text
语料 + 问题
  -> OpenIE 事实
  -> 来源支撑的证据链接索引
  -> 查询局部证据归纳
  -> 候选池 C_q
  -> 证据需求 B(q)
  -> 支持度缓存
  -> 覆盖感知的证据选择
  -> 最终证据集合 R_q
  -> 可选的 reader 问答
```

## ⚡ 快速开始

```bash
git clone https://github.com/Xiao-AI-Lab/EvLink.git
cd EvLink
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
python examples/end_to_end.py
```

复现协议见 [reproduce/README.md](reproduce/README.md)，产物格式见
[ARTIFACTS.md](ARTIFACTS.md)，v0.1 发布边界见
[docs/RELEASE_SCOPE.md](docs/RELEASE_SCOPE.md)。v0.1 之后的应用计划记录在
[docs/ROADMAP.md](docs/ROADMAP.md) 中。

## 🔌 检索器接入

将有序候选列表传给 `EvidenceSelector`。默认路径离线且具有确定性；通过
`EvidenceSelectorConfig` 可以启用模型驱动的证据需求与绑定。

```python
from evidencelink import EvidenceSelector, EvidenceSelectorConfig

selector = EvidenceSelector(
    EvidenceSelectorConfig(reader_budget_k=2, evidence_need_mode="anchor_list")
)
result = selector.select(
    question="Who founded Acme Corporation and where was the founder born?",
    candidates=[
        {
            "doc_id": "d0",
            "title": "Acme Corporation",
            "text": "Acme Corporation was founded by Alice Chen.",
            "score": 0.98,
        },
        {
            "doc_id": "d1",
            "title": "Alice Chen",
            "text": "Alice Chen was born in Singapore.",
            "score": 0.94,
        },
    ],
)

print([item.title for item in result.evidence])
print(result.evidence_needs)
print(result.trace)
```

运行完整示例：

```bash
python examples/external_retriever.py
```

传入 `workdir="runs/selection"` 可保留候选池、证据需求、绑定缓存和选择产物，
便于检查。外部候选会被视为兼容性输入；只有 EvLink 索引生成的候选才带有
本方法的来源支撑链接来源信息（provenance）。

本项目包含适配当前 HippoRAG 和 LightRAG 结果结构的无依赖适配器（adapter）。
EvLink 不会安装或导入这两个第三方包：

```python
from evidencelink import candidates_from_hipporag, candidates_from_lightrag

hipporag_candidates = candidates_from_hipporag(retrieval_result)
lightrag_candidates = candidates_from_lightrag(query_result)
```

离线示例位于 `examples/integrations/hipporag.py` 和
`examples/integrations/lightrag.py`。适配器会保留上游元数据（metadata），
但不会合成 EvLink 的边见证信息。

## 🧪 论文复现

将 EvLink 接入其他基准或论文复现工作流时，请使用 `evidencelink.api`。
API 名称与论文产物保持一致：候选池 `C_q`、证据需求 `B(q)`、支持度缓存和
最终证据集合 `R_q`。

```python
from evidencelink import PaperPipelineConfig, run_paper_pipeline

result = run_paper_pipeline(
    corpus_path="corpus.jsonl",
    questions_path="questions.jsonl",
    workdir="runs/evidencelink",
    config=PaperPipelineConfig(dataset="custom", force=True),
)

print(result["selection"])
print(result["selection_summary"])
```

安装后的 runner 也提供同样的接口边界：

```bash
evidencelink-pipeline \
  --corpus examples/corpus.jsonl \
  --questions examples/questions.jsonl \
  --workdir runs/demo \
  --dataset demo \
  --force
```

## 🗂️ 基准数据集

EvLink 为论文使用的五个基准提供了轻量级注册表和转换器：

| 数据集 | 规范名称 | 源格式 |
| --- | --- | --- |
| HotpotQA | `hotpotqa` | `context` + `supporting_facts` |
| 2WikiMultiHopQA | `2wikimultihopqa` | `context` + `supporting_facts` |
| MuSiQue | `musique` | `paragraphs` + `is_supporting` |
| Natural Questions | `nq_rear` | `contexts` + `is_supporting` |
| PopQA | `popqa` | `paragraphs` + `is_supporting` |

仓库与 Python 包均不分发基准数据集的源 JSON。项目支持托管下载
2WikiMultiHopQA、HotpotQA 和 MuSiQue，下载器会自动校验已下载文件。
NQ-ReAR 和 PopQA 需要手动放置源文件。

```bash
evidencelink-download-datasets --list
evidencelink-download-datasets \
  --dataset 2wikimultihopqa,hotpotqa,musique
```

随后，转换器会读取 `<dataset>.json` 和 `<dataset>_corpus.json`，并写出
标准的 `corpus.jsonl` 与 `questions.jsonl` 输入：

```bash
evidencelink-prepare-dataset \
  --dataset musique \
  --output-root runs/datasets/musique \
  --force
```

上游使用条款与手动源文件要求见 [datasets/README.md](datasets/README.md)。

## ▶️ 端到端演示

默认演示路径完全离线：使用简单 OpenIE 抽取器、简单的整题证据需求、简单
绑定缓存和确定性 embedding。

```bash
python scripts/run_pipeline.py \
  --corpus examples/corpus.jsonl \
  --questions examples/questions.jsonl \
  --workdir runs/demo \
  --dataset demo
```

对应的版本化运行方式为：

```bash
python scripts/run_reproduce_config.py reproduce/configs/offline-smoke.json
```

对于模型驱动的阶段，将相应阶段切换为 `llm`，并提供 OpenAI-compatible
端点：

```bash
python scripts/run_pipeline.py \
  --corpus corpus.jsonl \
  --questions questions.jsonl \
  --workdir runs/evidencelink \
  --openie-mode llm \
  --evidence-need-mode llm \
  --binding-mode llm \
  --llm-base-url "$EVLINK_LLM_BASE_URL" \
  --api-key "$EVLINK_API_KEY"
```

## 🛠️ 分阶段 CLI

```bash
python scripts/build_openie.py --corpus corpus.jsonl --output openie_facts.jsonl

python scripts/build_index.py \
  --corpus corpus.jsonl \
  --openie openie_facts.jsonl \
  --output evidence_link_index.json

python scripts/build_candidate_pool.py \
  --questions questions.jsonl \
  --corpus corpus.jsonl \
  --index evidence_link_index.json \
  --output candidate_pool.jsonl

python scripts/build_evidence_needs.py \
  --questions questions.jsonl \
  --output evidence_needs.jsonl \
  --mode whole_question

python scripts/build_binding_cache.py \
  --candidate-pool candidate_pool.jsonl \
  --evidence-needs evidence_needs.jsonl \
  --output binding_cache.json \
  --binding-model simple-binding

python scripts/run_evidence_selection.py \
  --dataset custom \
  --pool-json candidate_pool.jsonl \
  --requirement-report evidence_needs.jsonl \
  --binding-cache-path binding_cache.json \
  --output-json evidence_selection.json \
  --embedding-name deterministic-hash \
  --llm-binding-model simple-binding
```

## 📐 产物契约

主要产物格式汇总在 [ARTIFACTS.md](ARTIFACTS.md) 中。

事实是证据链接的依据材料，段落仍然是检索状态。候选池 `C_q` 不是最终证据
集合；最终证据集合 `R_q` 由覆盖感知的证据选择生成。

## ✅ 官方维护示例

以下示例属于经过测试的 v0.1 契约：

| 示例 | 用途 |
| --- | --- |
| `examples/end_to_end.py` | 完整的确定性流水线。 |
| `examples/external_retriever.py` | 通用有序候选接入。 |
| `examples/integrations/hipporag.py` | HippoRAG 结果结构适配器。 |
| `examples/integrations/lightrag.py` | LightRAG 结构化结果适配器。 |

## 🗃️ 仓库结构

```text
evidencelink/   可安装的 SDK
examples/       官方维护的离线示例
scripts/        分阶段命令与发布校验命令
reproduce/      版本化配置与指标定义
datasets/       玩具样例与源数据准备文档
tests/          公开契约与集成测试
```

## 🩺 故障排查

- 默认 embedding 后端是 `deterministic-hash`，端点为 `offline`。
  切换到模型驱动的 embedding 服务时，需要同时设置 `embedding_name` 和
  `embedding_base_url`。
- 模型驱动的 OpenIE、证据需求和绑定阶段需要 OpenAI-compatible 端点
  与 API key；离线示例不需要。
- 不要在已有的模型驱动索引中更换 embedding 模型。请重新构建索引，确保
  文档向量和查询向量位于同一 embedding 空间。

## 📚 引用与联系

EvLink 已被 EMNLP 2026 Main Conference 接收。引用元数据与作者列表见
[CITATION.cff](CITATION.cff)。终稿论文链接和正式论文集 BibTeX 将在公开书目
记录可用后补充。
如需报告 bug、咨询集成问题或反馈复现问题，请使用
[GitHub Issues](https://github.com/Xiao-AI-Lab/EvLink/issues)。
