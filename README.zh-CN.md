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

<p><a href="README.md">English</a> | <strong>简体中文</strong></p>

[快速开始](#-快速开始) · [检索器接入](#-检索器接入) · [论文复现](#-论文复现) · [产物契约](#-产物契约) · [引用与联系](#-引用与联系)

</div>

EvLink 是一个面向 Graph RAG 的图检索器。它用有原文依据的证据链接组织段落，
既能运行完整的研究流水线，也能接在现有检索器后面，从候选结果中挑出一组
紧凑、篇数可控的证据。

<p align="center">
  <img src="assets/evlink-overview.png" width="100%" alt="EvLink 方法概览">
</p>
<p align="center"><em>EvLink 用有原文依据的链接连接段落，先找回当前问题需要的局部证据，再根据证据需求覆盖选出紧凑的支持集合。</em></p>

> [!NOTE]
> PyPI 包名和 Python 导入名都是 `evidencelink`。

## ✨ 核心特点

- 🔗 **链接有据可查**：每条链接都保留段落级依据，图遍历结果可以回到原文核查；
- 🎯 **按问题补齐证据**：在固定篇数内覆盖问题的不同证据需求，减少
  重复和无效候选；
- 🔌 **可以接入现有检索器**：直接接收稠密、稀疏或图检索器给出的有序候选，
  不必推翻现有检索链路。

仓库为各阶段定义了清晰的产物格式，每一步选择都有轨迹可查，并附带一个完全
离线、结果固定的端到端示例。

这个示例刻意做得很小，方便先确认环境和流程。要跑正式评测，只需把示例输入
换成准备好的数据集产物。

## 🧭 流水线

整体流程与论文中的定义一致：

```text
语料 + 问题
  -> OpenIE 事实
  -> 构建有原文依据的证据链接索引
  -> 找回当前查询的局部证据区域
  -> 候选池 C_q
  -> 证据需求 B(q)
  -> 支持度缓存
  -> 按覆盖情况选择证据
  -> 最终证据集合 R_q
  -> 可选的答案生成
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

更多细节可以继续看：复现步骤在 [reproduce/README.md](reproduce/README.md)，
各类中间产物的字段定义在 [ARTIFACTS.md](ARTIFACTS.md)，v0.1 的发布范围在
[docs/RELEASE_SCOPE.md](docs/RELEASE_SCOPE.md)。后续应用层计划记录在
[docs/ROADMAP.md](docs/ROADMAP.md)。

## 🔌 检索器接入

如果已经有自己的检索器，不必重建整套系统。把按相关性排序的候选列表交给
`EvidenceSelector` 即可。默认实现完全离线、可复现；需要时也可以通过
`EvidenceSelectorConfig` 启用模型，让它抽取证据需求并判断候选是否提供支持。

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

完整示例可以直接运行：

```bash
python examples/external_retriever.py
```

设置 `workdir="runs/selection"` 后，EvLink 会把候选池、证据需求、支持度缓存
和最终选择结果全部落盘，方便排查和复现实验。需要注意，外部候选只是二阶段
选择器的输入；只有 EvLink 索引生成的候选才带有本方法定义的链接来源信息
（provenance）。

## 🧪 论文复现

做基准评测或复现论文时，建议直接使用 `evidencelink.api`。API 名称与论文
一一对应：候选池 `C_q`、证据需求 `B(q)`、支持度缓存，以及最终证据集合
`R_q`。

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

安装包后，也可以通过命令行运行同一套流程：

```bash
evidencelink-pipeline \
  --corpus examples/corpus.jsonl \
  --questions examples/questions.jsonl \
  --workdir runs/demo \
  --dataset demo \
  --force
```

## 🗂️ 基准数据集

项目内置了论文中五个数据集的注册信息和格式转换工具：

| 数据集 | 规范名称 | 源格式 |
| --- | --- | --- |
| HotpotQA | `hotpotqa` | `context` + `supporting_facts` |
| 2WikiMultiHopQA | `2wikimultihopqa` | `context` + `supporting_facts` |
| MuSiQue | `musique` | `paragraphs` + `is_supporting` |
| Natural Questions | `nq_rear` | `contexts` + `is_supporting` |
| PopQA | `popqa` | `paragraphs` + `is_supporting` |

考虑到数据许可和仓库体积，GitHub 仓库与 Python 包都不直接附带原始 JSON。
2WikiMultiHopQA、HotpotQA 和 MuSiQue 可以用下面的命令下载，下载完成后会
自动校验文件；NQ-ReAR 和 PopQA 需要手动准备。

```bash
evidencelink-download-datasets --list
evidencelink-download-datasets \
  --dataset 2wikimultihopqa,hotpotqa,musique
```

下载或放置好源文件后，转换工具会读取 `<dataset>.json` 和
`<dataset>_corpus.json`，生成统一的 `corpus.jsonl` 与 `questions.jsonl`：

```bash
evidencelink-prepare-dataset \
  --dataset musique \
  --output-root runs/datasets/musique \
  --force
```

原始数据的使用条款和手动准备要求见
[datasets/README.md](datasets/README.md)。

## ▶️ 端到端演示

默认演示不依赖在线模型：OpenIE、证据需求、支持度判断和 embedding 都使用
仓库自带的简单确定性实现。

```bash
python scripts/run_pipeline.py \
  --corpus examples/corpus.jsonl \
  --questions examples/questions.jsonl \
  --workdir runs/demo \
  --dataset demo
```

也可以直接运行仓库中固定版本的配置：

```bash
python scripts/run_reproduce_config.py reproduce/configs/offline-smoke.json
```

需要模型能力时，可以把对应阶段单独切换为 `llm`，并提供兼容 OpenAI API
的服务地址和密钥：

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

各阶段的产物格式都写在 [ARTIFACTS.md](ARTIFACTS.md) 中。

OpenIE 事实只用来说明段落为什么可以相连，真正被检索和选择的单位始终是
段落。`C_q` 只是候选池，经过覆盖选择后得到的 `R_q` 才是最终送入答案生成
模型的证据集合。

## ✅ 已测试示例

CI 会持续运行下面两个示例，确保 v0.1 的公开入口保持可用：

| 示例 | 用途 |
| --- | --- |
| `examples/end_to_end.py` | 完整、离线且结果固定的流水线。 |
| `examples/external_retriever.py` | 接入外部检索候选的通用示例。 |

## 🗃️ 仓库结构

```text
evidencelink/   可安装的 Python SDK
examples/       可以直接运行的离线示例
scripts/        各阶段命令和发布检查脚本
reproduce/      固定版本的配置与指标定义
datasets/       小型测试数据和数据准备说明
tests/          公开接口与集成测试
```

## 🩺 故障排查

- 开箱默认使用 `deterministic-hash` embedding，服务地址是 `offline`，因此
  不需要启动 embedding 服务。切换到真实模型时，请同时设置
  `embedding_name` 和 `embedding_base_url`。
- OpenIE、证据需求或支持度判断切换到模型模式后，需要提供兼容 OpenAI API
  的服务地址和 API key；离线示例不需要。
- 索引建好后不要直接更换 embedding 模型。换模型时请重建索引，确保文档
  和查询使用同一个向量空间。

## 📚 引用与联系

EvLink 已被 EMNLP 2026 Main Conference 接收。

**作者：** Linyao Zheng、Xuhang Shi、Zhifang Mao、Sai Zhou、Shuaixian An、
Xiuquan Hou。

机器可读的引用信息在 [CITATION.cff](CITATION.cff) 中。终稿论文链接和正式
论文集 BibTeX 会在公开书目记录发布后补上。

遇到 bug、接入问题或复现问题，请在
[GitHub Issues](https://github.com/Xiao-AI-Lab/EvLink/issues) 中反馈。

---

## 🚀 EvLink++ Coming Soon
