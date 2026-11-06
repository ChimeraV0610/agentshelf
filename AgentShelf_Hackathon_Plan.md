# AgentShelf：Hackathon 执行计划（"When the customer is a bot"）

## Context

UCLA Anderson MSBA hackathon，选题 "When the customer is a bot"。团队初步方向是 AgentShelf：让小卖家和品牌看到"AI 购物助手眼里的自己"，含 Agent 准确度评估、货架诊断、Listing 改写三个模块，数据用 Amazon ESCI。本文件是对该方向的评估、改进后的最终方案、技术栈、分步 roadmap、评估设计、prompt 草稿、目录结构、风险、pitch 结构和加分项。

**本计划的默认假设**（原文留了占位符，以下按最常见配置写；如果实际不同，只需按"砍法"一节调整范围）：
- 团队 4 人：ML-A（检索/评估）、ML-B（LLM agent/改写）、FE（Streamlit）、BA（抽样设计、统计分析、slides、pitch）。
- 比赛约 24–36 小时有效开发时间。
- 算力：笔记本 CPU + 一个 Colab GPU 用于一次性 embedding。
- LLM API：有 Claude API（也可能有 OpenAI），预算约 $50–100。

**核对过的数据事实**（来自 esci-data README，2026-11 读取）：
- 精简版（small_version==1）：48,300 query、1,118,011 条标注、平均每 query 23.15 个候选。
- 其中 US：**29,844 个 query、601,354 条标注、平均每 query 20.15 个候选**。这就是我们的全部工作集。
- README 列出的 baseline：Task 1 nDCG 0.83（候选内重排）；Task 2 四分类 macro F1 0.23、micro F1 0.62。这两个数字是 pitch 里天然的对照组。
- 标签分布约 E 65% / S 22% / C 3% / I 10%（论文数字，第一步脚本要打印确认）。
- 许可 Apache-2.0。

---

## 1. 方向评估

### 1.1 你们方向的优点
- **数据和主题严丝合缝**。ESCI 本质就是"给一个需求，一堆候选商品，哪个才是对的"，正是 AI 购物 agent 每天在做的决策。人工 E/S/C/I 标签是现成的 ground truth，这是本题下别的数据集都给不了的。
- **目标用户清晰且付费意愿明确**：小卖家（用户 4）和品牌（用户 1）是明确的 B2B 买单方。评委（MSBA 背景）会认 "GEO for commerce / 面向 AI 的 listing 优化" 这个商业故事。
- **三模块是递进关系**：先证明 agent 会买错（问题存在）→ 再告诉卖家你在 agent 眼里排第几、为什么没被选（诊断）→ 再给出可验证的改进（处方）。这是完整的产品闭环，而不是三个孤立 notebook。
- **可量化**：每一步都有数字（买错率、排名、选中率提升），比大多数 hackathon 项目的"我们做了个 chatbot"扎实得多。

### 1.2 弱点和评委可能的质疑（必须在 pitch 里主动回应）
1. **"你们的 agent 不是真的 agent"**。Rufus、Perplexity Shopping、ChatGPT shopping 都是黑盒，你们用自己写的 LLM agent 做代理。应对：不要只用一个模型。用两个不同厂商/不同档位的模型组成 "agent panel"，证明结论对模型不敏感；把它类比为"跨浏览器测试"。
2. **"没有价格和评论，agent 决策不真实"**。主题原文说 AI 根据商品信息、价格、评论决策。应对：（a）明说这是"文本层"的分析，而文本层恰恰是卖家唯一完全可控的杠杆；价格是定价问题，评论是运营问题，listing 文本是今天就能改的；（b）Amazon Reviews 2023 join 作为加分项，覆盖到多少就展示多少。
3. **"Listing 改写就是 SEO 作弊 / 过拟合到标注 query"**。这是最致命的质疑，必须用实验设计堵死（见第 5 节）：改写器**看不到任何 query**，评估全在该商品自己的 query 上；用 I/S 标签的 query 做负向对照，证明改写没有让商品在不该被选的场景被选中；再做一个故意作弊的 keyword-stuffing 基线，展示忠实度检查把它拦下来、并且它在负向对照上会"误伤"。
4. **"人工标签本身有噪声，agent 和人不一致不等于 agent 错"**。应对：措辞上用 "disagreement with human judgment"，但核心指标选最硬的那个：**agent 最终选中的商品被人工标为非 Exact 的比例**（买错率）。这个指标对标签噪声最不敏感，也最容易被听懂。

### 1.3 备选切入点比较
| 方向 | 目标用户 | ESCI 能否支撑 | 评委说服力 | 结论 |
|---|---|---|---|---|
| **AgentShelf（卖家/品牌侧）** | 用户 1、4 | 强：标签 + 文本字段全有 | 高：B2B、可量化、有闭环 | **推荐，主线** |
| "Did my bot buy right?"（购物者侧审计） | 用户 2 | 中：能算买错率，但购物者拿不到候选列表和标签 | 中：消费者产品缺变现路径，demo 像研究报告 | 作为 AgentShelf 模块 1 的叙事角度，不独立成项目 |
| Brand Share of Agent Picks（营销侧） | 用户 3、1 | 强：product_brand 字段现成 | 高：一张"各品牌在 agent 推荐里的份额"图很抓人 | **推荐并入 AgentShelf 作为品牌视图**，成本几乎为零（复用模块 1 输出） |
| Bot 购买真实性核验（支付侧） | 用户 5 | 无：没有交易、设备、行为数据 | 低：只能编造数据 | 放弃 |

**最终建议**：主线保持 AgentShelf，吸收两点：模块 1 同时产出"品牌份额"视图（服务用户 1、3）；pitch 开头用购物者视角（用户 2）讲痛点，再切到卖家视角讲产品。

---

## 2. 最终方案（项目定位）

**AgentShelf 是面向小卖家和品牌的 "AI 购物助手可见度诊断与优化" 工具。** 当越来越多订单由 AI 助手代人下单时，卖家失去了与买家的直接接触，广告和页面设计失效，决定成败的只剩下 listing 文本本身。AgentShelf 用真实的 Amazon 搜索需求（ESCI 的 29,844 个 US query）和一组模拟 AI 购物 agent，先量化 agent 的"买错率"证明问题真实存在，再告诉每个卖家：对于真实的用户需求，你的商品能不能被 agent 检索到、排第几、会不会被选中、没被选中的原因是什么；最后给出经过忠实度校验的 listing 改写，并用留出的真实 query 和负向对照证明改写带来的提升是真实的、不是关键词堆砌。

---

## 3. 技术栈

| 层 | 选择 | 理由 / 备注 |
|---|---|---|
| 数据处理 | `pandas` + `pyarrow`（读 parquet 时用 `filters` 只读 US 行），`duckdb` 可选 | products 表全 locale 带 description 可能超 1 GB 内存；pyarrow filter 下推能避免读全表。join 必须用 `(product_id, product_locale)` 复合键，product_id 跨 locale 不唯一 |
| 文本清洗 | `re` + `html.unescape`，或 `lxml` 去 HTML 标签 | description 里有 HTML；bullet 用换行拼接 |
| 稀疏检索 | **`bm25s`**（配 `PyStemmer`） | 纯 numpy，索引 ~50 万短文本几十秒，查询极快；比 rank_bm25 快两个量级 |
| 稠密检索 | **`sentence-transformers` + `BAAI/bge-small-en-v1.5`**（384 维） | 质量/速度平衡最好的小模型；Colab T4 上 50 万条约 5–10 分钟，M 系列笔记本 CPU 约 30–60 分钟（只做一次，存 `.npy` float16）。备选 `all-MiniLM-L6-v2`（更快、略弱） |
| 向量索引 | **`faiss-cpu` `IndexFlatIP`**（归一化后内积=余弦） | 50 万×384 精确检索完全够快，不引入 HNSW 的调参和召回损失；备选直接 numpy 分块矩阵乘 |
| 混合检索 | RRF（k=60）融合 BM25 与 dense | 10 行代码，稳定提升 |
| 重排（可选） | `cross-encoder/ms-marco-MiniLM-L-6-v2` 对 top-50 重排 | 可选开关。ms-marco 是网页域，在商品域不一定赢 bge；跑一次 nDCG 决定是否保留 |
| LLM（bulk 判别 agent） | **`claude-sonnet-5-5`**（$2/$10 per MTok）做主 agent；**`claude-haiku-4-5`**（$1/$5）做第二个 panel agent；若有 OpenAI，用 `gpt-4o-mini` 替代 Haiku 作为"异厂商"对照更有说服力 | Sonnet 5.5 用 `output_config.effort: "low"` 控成本；Haiku 4.5 不开 thinking |
| LLM（改写 + 忠实度判官） | **`claude-opus-5-5`**（$4/$20），adaptive thinking，`effort: "medium"` | 量小（几百次调用）、质量敏感，用最强的 |
| LLM 结构化输出 | Anthropic SDK 的 `output_config.format`（JSON schema）/ `client.messages.parse()`；OpenAI 用 `response_format: json_schema` | 保证 JSON 可解析，省掉重试逻辑 |
| 缓存 | `sqlite3` 或 `diskcache`，key = sha256(model, system, user, schema, seed) | 所有调用先查缓存；demo 期间 100% 命中 |
| 并发/重试 | `asyncio` + `Semaphore(8–16)` + `tenacity` | 429 退避 |
| 评估 | 自写 `ndcg()`（20 行）并用 `ranx` 交叉验证一次；`scikit-learn`（confusion matrix、F1、kappa）；`scipy`/`statsmodels`（bootstrap、McNemar） | |
| Demo | **Streamlit** + `plotly` | 只读 `results/` 下的 parquet/json |
| 环境 | Python 3.11，`uv`（或 pip）+ `requirements.txt`，`Makefile` 串起各步骤 | |

**API 成本估算**（1,000 个 query，平均 20 个候选，listwise 一次调用判完全部候选并选一个）：

| 任务 | 调用量 | 输入/输出 token | 模型 | 费用 |
|---|---|---|---|---|
| Agent 判别 | 1,000 | ~3k / ~0.9k 每次 → 3M / 0.9M | Sonnet 5.5 | ≈ $15 |
| 第二 agent | 1,000 | 同上 | Haiku 4.5 | ≈ $8 |
| Query 部门标注 | ~180（每次 50 个 query） | 2M / 0.3M 合计 | Haiku 4.5 | ≈ $3 |
| 诊断补跑（商品相关 query 不在样本里） | ~1,200 | 同判别 | Sonnet 5.5 | ≈ $18 |
| Listing 改写 | 200 | 1.5k / 0.6k | Opus 5.5 | ≈ $4 |
| 忠实度判官 | 200 + 200（作弊基线） | 2k / 0.3k | Opus 5.5 | ≈ $5 |
| 改写后重判 | ~1,200 | 同判别 | Sonnet 5.5 | ≈ $18 |
| **合计** | | | | **≈ $70**，留 30% 余量到 $90 |

Batch API 半价，但异步返回时间不可控，hackathon 里不建议依赖；实时调用 + 并发 12 即可。

---

## 4. Roadmap（按步骤，不标时间）

### 第零步：环境与数据加载
- **目标**：所有人能在本地跑通 `make data`，拿到干净的 US 子集。
- **做什么**：
  - `scripts/00_download.sh`：`git lfs` clone esci-data（或从 HF 镜像下载两个 parquet）。
  - `src/data/load.py`：`load_examples_us()` 用 pyarrow filters 读 `small_version==1 & product_locale=='us'`；`load_products_us()` 只读 US 行；`clean_text()` 去 HTML、合并 bullets；`build_doc_text(product, fields=...)` 生成检索用文本（默认 `title + brand + color + 前 2 条 bullet`，截断 256 token）。
  - `scripts/01_build_corpus.py`：输出 `data/processed/examples_us.parquet`、`products_us.parquet`、`corpus.parquet`（product_id, doc_text）。
  - 打印：query 数、标注数、商品数、E/S/C/I 分布、title/bullet/description 空值率、每 query 候选数分布、split 分布。
- **输入/输出**：原始 parquet → `data/processed/*.parquet` + `reports/data_profile.md`。
- **完成标准**：query 数 = 29,844，标注数 = 601,354；空值率表写进 profile；一个随机 query 的候选和标签能肉眼对上。
- **负责**：ML-A。

### 第一步：候选内 nDCG 评测框架
- **目标**：一个函数评估任意排序器，并拿到两个 baseline 数字。
- **做什么**：
  - `src/eval/ndcg.py`：`gain = {'E':1.0,'S':0.1,'C':0.01,'I':0.0}`；`ndcg_at_k(labels_in_ranked_order, k=None)`；`evaluate_ranker(ranker_fn, examples_df) -> {mean_ndcg, per_query}`。
  - 跑随机排序和"按原始顺序"两个 baseline；用 `ranx` 对 20 个 query 交叉验证自写实现。
- **输入/输出**：examples_us → `results/ndcg_baselines.json`。
- **完成标准**：随机排序 nDCG 落在合理区间（约 0.7–0.8，因为 E 占多数）；与 ranx 一致到 1e-6。
- **负责**：ML-A。

### 第二步：全量索引（BM25 + dense + hybrid）
- **目标**：对任意 query 在全部 US 商品里检索 top-k；并在候选内重排评 nDCG。
- **做什么**：
  - `src/retrieval/bm25_index.py`：`build()` 用 bm25s 建索引存 `indices/bm25/`；`search(queries, k)`；`score_single_doc(query, doc_text)` 用索引的 IDF、avgdl 对单个新文本打分（第七步改写后重算排名要用，bm25s 不支持增量）。
  - `src/retrieval/dense_index.py`：`encode_corpus()`（Colab 跑，存 `indices/dense/emb.npy` float16 + `ids.npy`）；`build_faiss()`；`search()`；`encode_single_doc()`。
  - `src/retrieval/hybrid.py`：`rrf(rank_lists, k=60)`。
  - `src/retrieval/rerank.py`（可选）：cross-encoder 对 top-50 重排。
  - `scripts/02_eval_retrieval.py`：（a）候选内 nDCG：BM25 / dense / hybrid / +cross-encoder；（b）全量召回：对每个 (query, E 商品) 对，记录该商品在全量 top-100 的排名，报 recall@10/50、MRR。
- **输入/输出**：corpus.parquet → `indices/`；→ `results/retrieval_eval.json`、`results/full_corpus_ranks.parquet`（query_id, product_id, rank_bm25, rank_dense, rank_hybrid）。
- **完成标准**：hybrid 候选内 nDCG 明显高于随机，与 README baseline 0.83 可比；全量 recall@50 有数字；单 doc 重打分函数与整体检索结果在抽样 50 个商品上排名一致。
- **负责**：ML-A（embedding 在 Colab 上由 ML-B 或 FE 帮跑）。

### 第三步：Query 分层与抽样
- **目标**：1,000 个 test query 的分层样本，分层字段能支撑"按部门报告"。
- **做什么**：
  - **指出一个错误假设**：ESCI 没有类目字段。"按类目分层"要自己造标签。推荐 `scripts/03_tag_departments.py`：用 Haiku 4.5 每 50 个 query 一批，映射到固定的 12 个部门（Electronics, Home & Kitchen, Clothing/Shoes, Beauty & Personal Care, Health, Grocery, Toys & Games, Sports & Outdoors, Tools & Automotive, Office/Books/Media, Pet, Baby/Other）。备选：bge 对 query 做 embedding + KMeans(k=12) 再人工命名。
  - 另两个分层维度（纯规则，`src/sampling/strata.py`）：
    - `is_brand_query`：query 的 token 是否命中其候选商品的 `product_brand`。
    - `difficulty`：候选里 E 占比 ≤25% / 25–75% / >75%。
  - `scripts/04_sample_queries.py`：只从 `split=='test'` 抽；过滤候选数 ≥ 8；按部门比例分配、每部门至少 40；总 1,000；固定 `seed=42`；同时输出每条 query 的候选列表（打乱顺序的固定随机排列，存下来保证后续可复现）。
- **输入/输出**：examples_us → `data/samples/queries_1000.parquet`、`data/samples/candidates_1000.parquet`（query_id, product_id, shuffled_pos, esci_label）。
- **完成标准**：每部门 ≥ 40；三维度边际分布写进 `reports/sample_design.md`；部门标签抽 30 条人工看准确率 ≥ 85%。
- **负责**：BA（设计）+ ML-B（跑标注脚本）。

### 第四步：LLM client 与 agent 判别器
- **目标**：对 1,000 个 query 跑两个 agent，产出每候选的标签和最终选择，全部缓存。
- **做什么**：
  - `src/llm/client.py`：`call(model, system, user, schema, temperature=0, seed) -> dict`，先查 sqlite 缓存；异步并发 12；tenacity 退避；记录 token 用量到 `results/usage.jsonl`。
  - `src/agent/prompts.py`：第 6 节的 judge prompt；`src/agent/schema.py`：JSON schema。
  - `src/agent/judge.py`：`judge_query(query, candidates, model)`；候选以 `c01…c20` 编号呈现（不给 ASIN，避免模型凭记忆而不是凭文本判断），字段用 title / brand / color / bullets / description(截 600 字符)。
  - `scripts/05_run_agents.py --model ... --limit 50` 先跑 50 条做 sanity（看 JSON 合法率、E/S/C/I 分布是否离谱），再全量，再换第二个模型。
- **输入/输出**：samples → `results/agent_runs/{model}.parquet`（query_id, product_id, agent_label, confidence, missing_attrs, reason_code, is_pick）。
- **完成标准**：JSON 合法率 ≥ 99%；两模型各 1,000 条全部命中缓存可重放；总费用在估算 ±30% 内。
- **负责**：ML-B。

### 第五步：Agent 评估分析（模块 1 + 品牌视图）
- **目标**：产出 pitch 第 2 页的所有数字。
- **做什么**（`src/analysis/agent_eval.py`，输出到 `results/agent_eval.json` 与若干 csv）：
  - 候选级：人工 vs agent 混淆矩阵、macro/micro F1（对照 README baseline 0.23 / 0.62）、按部门的 F1。
  - **买错率**（headline）：agent 最终 pick 的商品，人工标签为 S / C / I 的比例；按部门、按 is_brand_query、按 difficulty 拆；S 当 E 买的占比单独报。
  - 弃选率：agent 返回 `null`（"要问用户"）的比例，以及弃选时候选里其实有 E 的比例（过度保守）。
  - 两个 agent 之间的 Cohen's kappa，和各自与人工的 kappa。
  - **品牌视图**：在被 agent pick 的商品里，各 brand 的份额 vs 该 brand 在候选里的份额（"agent 偏好指数"）；E 标商品里有无 brand 字段与被 pick 概率的关系。
  - 全部比例给 95% bootstrap CI。
- **完成标准**：一张表 + 四张图（混淆矩阵、按部门买错率、brand 偏好 top/bottom、两 agent 一致性）。
- **负责**：BA（分析、作图）+ ML-B（接口）。

### 第六步：货架诊断（模块 2）
- **目标**：对 150–200 个商品输出"AI 眼里的我"报告。
- **做什么**：
  - `src/diagnose/select_products.py`：候选商品池 = 出现在 US 全集中、有 ≥ 3 个 E 标 query 且 ≥ 1 个 S/I 标 query 的商品；再按"基线弱"筛（hybrid 全量排名在 ≥ 半数 E query 上 > 10，或 agent pick 率低）；按部门分层取 150–200 个。
  - `src/diagnose/shelf.py`：对每个商品 P：
    - 检索指标：P 的每个 E query 在全量 top-10 / top-50 的命中率、平均排名（截断 100）、BM25 vs dense 各自排名（能说"关键词找得到但语义找不到"或反之）。
    - Agent 指标：对 P 的每个 E query（候选用 ESCI 原候选，固定打乱），agent 是否 pick P；报 pick 率并给"公平份额基线" `1/#E候选`，用 `pick率 / 公平份额` 作为相对指标。
    - 未选中原因：汇总 agent 对 P 的 `reason_code` 与 `missing_attributes`，取 top-3。
    - 这些 query 大多不在第四步的 1,000 样本里，需要补跑 agent（预算已含）。
  - 导出 `results/diagnostics/{product_id}.json` 和汇总 `results/diagnostics_summary.parquet`。
- **完成标准**：每个商品 JSON 都有检索、agent、原因三块；汇总表能画"排名分布"和"原因分布"。
- **负责**：ML-A（检索部分）+ ML-B（agent 部分）。

### 第七步：Listing 改写 + 忠实度检查（模块 3）
- **目标**：对第六步的商品生成改写版，并重跑诊断得到 before/after。
- **做什么**：
  - `src/rewrite/dept_profile.py`：从 **train split** 的同部门 query 里提炼"该部门买家常指定的属性类型"（如尺寸、兼容机型、材质、数量、适用人群），每部门一段 ≤ 150 词的文字。**绝不使用该商品自己的任何 query**。
  - `src/rewrite/rewriter.py`：Opus 5.5，输入 = 原 listing 字段 + 部门属性概要，输出 = 新 title（≤ 200 字符）+ 5 条 bullet，每条附 `evidence`（引用原文片段）。
  - `src/rewrite/faithfulness.py`：两层。规则层：新文本里出现的数字/单位/型号必须在原文出现；禁用词表（best, #1, premium quality 等空话）；title 长度上限；重复 n-gram 率上限。LLM 层：Opus 5.5 判官 prompt，逐条 claim 判 `supported / unsupported / unverifiable`，任何 `unsupported` 则打回重写（最多 2 轮）。
  - `src/rewrite/stuffed_baseline.py`：故意作弊的对照——把该商品 E query 的高频词直接塞进标题。用于展示（a）忠实度检查会拦下它（b）它在负向对照上会误伤。
  - `src/rewrite/rescore.py`：用第二步的 `score_single_doc` / `encode_single_doc` 重算该商品在每个相关 query 上的新排名（其它商品不变）；把候选列表里 P 的文本换成新版、**同一打乱顺序**重跑 agent。
  - `scripts/07_rewrite_and_rescore.py` 输出 `results/rewrite/before_after.parquet`（product_id, query_id, label, rank_before, rank_after, pick_before, pick_after, variant ∈ {original, rewritten, stuffed}）。
- **完成标准**：≥ 150 个商品有通过忠实度检查的改写版；stuffed 版全部被检查标红；before/after 表齐全。
- **负责**：ML-B。

### 第八步：统计检验与防作弊验证
- **目标**：把"提升是真的"做成可展示的证据。
- **做什么**（`src/analysis/rewrite_eval.py`）：
  - 正向：E query 上 pick 率、top-10 命中率、平均排名的 before/after，配对 bootstrap CI；pick 率用 McNemar。
  - **负向对照**：S/I query 上 pick 率 before/after，期望不升（上限：不超过 +2pp）。
  - stuffed 基线同上两项，期望正向提升更大但负向误伤明显。
  - 按部门拆；报"提升最大/最小的 5 个商品"用于 demo 讲故事。
- **完成标准**：一张 2×3 表（original / rewritten / stuffed × E 指标 / S-I 指标 / 忠实度通过率）。
- **负责**：BA。

### 第九步：Streamlit demo
- **目标**：三页、全部读缓存、断网能跑。
- **做什么**（`app/`）：
  - `Home.py`：问题页，买错率大数字 + 按部门条形图 + 品牌偏好图（读 `results/agent_eval.json`）。
  - `pages/1_Shelf_Diagnosis.py`：下拉选商品（150–200 个），显示原 listing、"真实用户是这样搜你的"（E query 列表）、每个 query 的全量排名和 agent 是否选中、agent 原话的拒选理由词云/条形图。
  - `pages/2_Rewrite.py`：左右对比原文/改写，忠实度检查结果（每条 bullet 绿/红 + evidence），before/after 排名和 pick 率，底部"作弊版"开关展示被拦下的效果。
  - `app/data.py`：统一 `@st.cache_data` 读 results。
  - 预录一段 90 秒 demo 视频做兜底。
- **完成标准**：拔掉网线能完整演示；任一页加载 < 2 秒。
- **负责**：FE（第三步后即可开始用 mock 数据搭骨架）。

### 第十步：Pitch 与排练
- 见第 9 节。BA 主写，全员排练两遍，严格 5 分钟。
- **负责**：BA + 全员。

### 砍法（时间不够时的最小可行版本）
保留：第零到第五步（全量跑）+ 第六、七步只做 50 个商品 + 第九步两页。砍掉：cross-encoder、第二 agent（改为抽 200 条跑 Haiku 作一致性抽检）、品牌视图。

---

## 5. 抽样与评估设计

### 5.1 样本量依据
- **1,000 个 query**：比例类指标（买错率）95% CI 半宽 ≈ 1.96 × √(0.25/1000) ≈ 3.1pp，足够说"约 X%"。按 12 个部门拆后每部门 ≥ 40，部门级 CI 约 ±15pp，只用于排序和找极端部门，不做部门间显著性声明。
- **约 20,000 条候选级判断**：C 类约 3% → 600 条，macro F1 稳定。
- **150–200 个商品的配对实验**：McNemar 在不一致对占 30%、效应 10pp 时 n≈150 即达 80% power；每个商品带 3–6 个 E query 和 1–3 个 S/I query，总配对 ≈ 1,000 条。

### 5.2 分层方式
- 仅用 `split=='test'`（标注集的 test 切分），避免任何"训练集泄漏"的质疑，即使我们不训练模型。
- 三维度：LLM 标注的 12 个部门（主）× is_brand_query × difficulty。按部门比例分配，每部门下限 40，另两维度只做事后检查边际分布不偏。
- 候选 ≥ 8 的 query 才入样（候选太少 agent 任务退化）。

### 5.3 报告指标
| 模块 | 指标 | 对照 |
|---|---|---|
| 检索 | 候选内 nDCG（E=1, S=0.1, C=0.01, I=0），BM25 / dense / hybrid / +CE | 随机排序；README baseline 0.83 |
| 检索 | 全量 recall@10/50、MRR（对 E 商品） | BM25 vs dense |
| Agent | 混淆矩阵、macro/micro F1、按部门 F1 | README Task 2 baseline 0.23 / 0.62 |
| Agent | **买错率**（pick 被标为非 E 的比例）、S-as-E 率、弃选率、过度保守率 | 随机 pick 的买错率 ≈ 1 − E 占比 |
| Agent | 两 agent kappa、各自与人工 kappa | |
| 品牌 | brand 偏好指数 = pick 份额 / 候选份额 | 1.0 |
| 诊断 | 商品级：E query 的 top-10 命中率、平均排名、pick 率 / 公平份额 | |
| 改写 | E query 上 Δpick 率、Δtop-10 命中率、Δ平均排名（配对 bootstrap CI、McNemar） | stuffed 基线 |
| 改写 | S/I query 上 Δpick 率（**期望 ≈ 0**） | stuffed 基线（期望明显 > 0） |
| 改写 | 忠实度通过率、被拦下的 claim 数 | |

### 5.4 如何证明改写不是作弊或过拟合
1. **信息隔离**：改写器只看商品原文 + 部门级属性概要（来自 train split 的其它商品的 query）。该商品自己的 query 从未进入 prompt。代码里 `rewriter.py` 的函数签名不接收 query 参数，评委可以直接看。
2. **留出评估**：所有评估 query 对改写器都是未见的。
3. **负向对照**：用 ESCI 的 S/I 标签，天然得到"这个商品不该被选中"的 query。改写后这些 query 上 pick 率不升，说明提升来自更清楚地表达自己，而不是更像所有 query。
4. **作弊基线**：stuffed 版展示两件事：忠实度检查能识别（不在原文的 claim、重复 n-gram）；它在 S/I query 上误伤更大，即"短期得分、长期退货"。
5. **忠实度双层检查**：规则层（数字/型号/单位必须源于原文）+ LLM 判官（逐 claim 蕴含判断，附 evidence）。
6. **位置偏差控制**：改写前后用同一候选打乱顺序；可选再跑一个反序取均值。
7. **多 agent 一致**：提升在两个不同模型上同向。

---

## 6. Agent prompt 设计

### 6.1 判别 agent（listwise：标注全部候选 + 选一个）

System prompt（英文，LLM 对英文指令更稳）：

```
You are a shopping assistant acting on behalf of a customer who typed a search query
and asked you to pick the product to buy. You can only see the product listings below;
you cannot browse, ask follow-up questions, or see prices, ratings or reviews.

For every candidate, assign exactly one relevance label using these definitions
(they follow Amazon's human annotation guidelines):
- E (Exact): the product satisfies all the specifications stated or clearly implied by the query.
- S (Substitute): the product is somewhat relevant; it fails to satisfy some aspect of the
  query but could serve as a functional substitute.
- C (Complement): the product does not satisfy the query but could be used together with
  the exact product (e.g. a case for the requested phone).
- I (Irrelevant): the product is irrelevant or fails a central aspect of the query.

Then decide what you would add to the cart: the single best candidate id, or null if no
candidate is an E and you would rather ask the customer. Base every judgment strictly on
the listing text; do not rely on outside knowledge about specific products or brands.
For each candidate, list the query attributes the listing fails to confirm, and pick one
reason_code for why it is not the best choice.
Respond with JSON only, matching the provided schema.
```

User message 模板：

```
Query: "{query}"

Candidates (order is random):
[c01] Title: {title}
      Brand: {brand} | Color: {color}
      Bullets: {bullets_joined, 截 400 字符}
      Description: {description, 截 600 字符}
[c02] ...
```

JSON schema：

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["judgments", "pick", "pick_reason"],
  "properties": {
    "judgments": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["id", "label", "confidence", "missing_attributes", "reason_code"],
        "properties": {
          "id": {"type": "string", "pattern": "^c[0-9]{2}$"},
          "label": {"type": "string", "enum": ["E", "S", "C", "I"]},
          "confidence": {"type": "number", "minimum": 0, "maximum": 1},
          "missing_attributes": {
            "type": "array", "items": {"type": "string"}, "maxItems": 4,
            "description": "Query requirements the listing does not confirm, e.g. 'size 10', 'compatible with iPhone 15'"
          },
          "reason_code": {
            "type": "string",
            "enum": ["best_match", "missing_spec", "spec_mismatch", "wrong_product_type",
                     "unclear_title", "accessory_or_part", "quantity_or_pack_mismatch",
                     "brand_mismatch", "other"]
          }
        }
      }
    },
    "pick": {"type": ["string", "null"], "pattern": "^c[0-9]{2}$"},
    "pick_reason": {"type": "string", "maxLength": 300}
  }
}
```

实现注意：`temperature=0`；Sonnet 5.5 用 `output_config.effort: "low"`；每次调用把 `c01…` 到 product_id 的映射存在缓存记录里；候选超过 30 个时截到 30（保留全部 E 并随机补齐），并记录截断标记。

### 6.2 Listing 改写器（Opus 5.5）

```
You are a listing copywriter for an e-commerce seller. Your reader is an AI shopping
agent that decides purely from listing text, so clarity and completeness of factual
attributes matter more than persuasion.

Rewrite the title (max 200 characters) and write exactly 5 bullet points for the product
below. Rules:
1. Every fact must be supported by the ORIGINAL listing. Do not invent specifications,
   compatibility, materials, quantities, certifications or use cases.
2. Put the product type and its most decision-relevant attributes first in the title.
3. No marketing filler ("premium", "best", "#1", "amazing"). No repeated phrases.
4. Shoppers in this department typically specify: {department_attribute_profile}.
   If the original listing states any of these, surface them; if it does not, do NOT guess.
5. For each bullet, include the exact span of the original text that supports it.
Return JSON only.
```

Schema：`{"title": str, "bullets": [{"text": str, "evidence": str}] (exactly 5), "attributes_not_available": [str]}`。最后一个字段让模型显式列出"部门常问但原文没写"的属性，正好成为给卖家的"你该补充的信息"建议，是 demo 里的额外亮点。

### 6.3 忠实度判官（Opus 5.5）

```
You are auditing a rewritten product listing against the original. For each claim in the
rewritten title and bullets, decide whether it is SUPPORTED by the original text,
UNSUPPORTED (contradicts or adds a fact not present), or UNVERIFIABLE (vague, no factual
content). Quote the supporting span when SUPPORTED. Be strict: a more specific claim than
the original (e.g. "stainless steel" when the original says "metal") is UNSUPPORTED.
Return JSON only.
```

Schema：`{"claims": [{"text": str, "verdict": "SUPPORTED|UNSUPPORTED|UNVERIFIABLE", "evidence": str|null}], "overall_pass": bool, "stuffing_flag": bool}`。规则层在 LLM 层之前跑，命中即直接 fail，省调用。

### 6.4 部门标注（Haiku 4.5）
批量 50 个 query，返回 `[{"query_id": ..., "department": enum}]`，enum 为第三步的 12 个部门。

---

## 7. 项目文件结构

```
agentshelf/
├── README.md
├── Makefile                      # make data / index / sample / agents / diagnose / rewrite / app
├── requirements.txt
├── .env.example                  # ANTHROPIC_API_KEY, OPENAI_API_KEY
├── configs/
│   └── default.yaml              # 路径、模型名、k 值、seed、样本量、部门 enum
├── data/
│   ├── raw/                      # esci-data parquet（gitignore）
│   ├── processed/                # examples_us / products_us / corpus
│   └── samples/                  # queries_1000 / candidates_1000 / products_for_diagnosis
├── indices/                      # bm25/  dense/（gitignore）
├── cache/
│   └── llm_cache.sqlite          # 所有 LLM 调用（提交到 repo 以便 demo 复现）
├── src/
│   ├── data/load.py
│   ├── eval/ndcg.py
│   ├── retrieval/{bm25_index,dense_index,hybrid,rerank}.py
│   ├── sampling/strata.py
│   ├── llm/client.py
│   ├── agent/{prompts,schema,judge}.py
│   ├── diagnose/{select_products,shelf}.py
│   ├── rewrite/{dept_profile,rewriter,faithfulness,stuffed_baseline,rescore}.py
│   └── analysis/{agent_eval,rewrite_eval,plots}.py
├── scripts/                      # 00_download.sh … 08_make_report.py，按 roadmap 编号
├── results/                      # *.json / *.parquet / figures/，demo 只读这里
│   ├── diagnostics/
│   └── rewrite/
├── reports/                      # data_profile.md, sample_design.md, findings.md
├── app/
│   ├── Home.py
│   ├── pages/1_Shelf_Diagnosis.py
│   ├── pages/2_Rewrite.py
│   └── data.py
├── notebooks/                    # 探索用，不进 pipeline
└── pitch/                        # slides、demo 视频、讲稿
```

---

## 8. 风险与应对

| 风险 | 概率 | 应对 |
|---|---|---|
| products parquet 全 locale 读进内存爆掉 | 中 | pyarrow `filters=[('product_locale','==','us')]`；先不读 description 列；实在不行 duckdb 直接 SQL 过滤 |
| git lfs 下载慢/失败 | 中 | 备用 HF 镜像；一人先下好后用网盘分发 |
| embedding 在 CPU 上太慢 | 中 | Colab T4 跑一次存 npy；或退到 MiniLM；或只对"出现在任意候选列表里的商品"编码（与全集差别不大但先跑通） |
| API 429 / 速度慢 | 高 | 并发 12 + 指数退避；先 50 条 sanity 再放量；两个模型并行跑；缓存保证中断可续 |
| API 超预算 | 低 | `results/usage.jsonl` 实时累计，`client.py` 内置硬上限（默认 $90）超了就抛异常 |
| JSON 不合法 | 低 | 结构化输出 + 一次重试；仍失败记 `parse_error` 不阻塞 |
| 候选 >30 导致 prompt 过长或 agent 标注质量下降 | 中 | 截到 30，保留全部 E；记录截断率 |
| 标签噪声导致 macro F1 看着低 | 高 | 预期中的；pitch 主推买错率；kappa 同时报，说明"人与人之间也不完全一致" |
| 部门 LLM 标注不准 | 中 | 抽 30 条人工核；错的部门合并 |
| Amazon Reviews 2023 join 覆盖率低 / 文件太大 | 高 | 仅作加分项；只拉 2–3 个 category 的 meta 文件、流式读取、按 parent_asin 和 asin 双键匹配；覆盖率 < 30% 就只做案例展示不做统计 |
| 改写提升不显著 | 中 | 商品池按"基线弱"筛选就是为了留出提升空间；如仍不显著，诚实报告并强调负向对照和忠实度框架本身的价值 |
| Demo 现场网络/API 问题 | 中 | 全部读缓存；sqlite 缓存进 repo；预录 90 秒视频 |
| Streamlit 页面加载慢 | 低 | `@st.cache_data`，results 用 parquet 不用 csv |
| 评委质疑"你们的 agent 不是真 agent" | 高 | 两模型 panel + 一致性数据；措辞"模拟 agent 面板" |

---

## 9. Pitch 结构（5 页 + demo）

**第 1 页：Hook（购物者视角）**
- 一句话："Your next customer is a bot. It never sees your ad, your photos, or your storefront. It reads your listing text and decides."
- 一个来自数据的真实例子：某 query，agent 选了 S 商品而非 E，并给出它的理由（agent 原话）。

**第 2 页：问题有多大（模块 1 的数字）**
- 大数字：买错率 X%（95% CI），其中 S-as-E 占 Y%。
- 按部门条形图；品牌偏好指数 top/bottom 3。
- 一句话对照：两个不同厂商的 agent 结论一致（kappa）。

**第 3 页：AgentShelf 是什么（产品）**
- 三步闭环图：See（AI 眼里的你）→ Explain（为什么没被选）→ Fix（可验证的改写）。
- 直接切 demo（90 秒）：选一个商品 → 看到它在真实 query 上排第 37、agent 从未选中、理由"未确认尺寸/兼容性" → 点改写 → 每条 bullet 带 evidence 绿勾 → 排名升到 top 5、pick 率从 0% 到 60% → 打开"作弊版"开关：忠实度检查全红，S/I query 上误选上升。
- 回到 slides。

**第 4 页：证据（模块 3 的数字 + 防作弊）**
- 2×3 表：original / rewritten / stuffed × E 指标 / S-I 指标 / 忠实度通过率。
- 三句话：改写器看不到 query；留出 query 评估；负向对照不升。

**第 5 页：商业与下一步**
- 用户：小卖家按 listing 数付费的 SaaS；品牌按 agent share-of-voice 监测订阅；营销团队把它当"面向 AI 的 SEO 审计"。
- 路线：接入真实 agent（Rufus/ChatGPT shopping 的可观察输出）、加入价格评论维度、多语言。
- 收尾一句："When the customer is a bot, the listing is the store."

**Demo 最有冲击力的讲法**：先让观众看到 agent 的拒选理由原话（人话、具体），再看改写后同一个 agent 把它选中，最后用"作弊版"展示系统会拒绝走捷径。三个节拍，每个 30 秒。

---

## 10. 加分项（按性价比排序）

1. **"你该补充的信息"建议**：改写 schema 里的 `attributes_not_available` 直接变成卖家 to-do 列表，几乎零成本，demo 很加分。
2. **价格与评分维度**：join Amazon Reviews 2023 的 `meta_*` 文件（price, average_rating, rating_number），对覆盖到的 query 跑一版"带价格评论"的 agent，看买错率和品牌偏好怎么变。能回答评委"没有价格"的质疑。
3. **位置偏差实验**：同一候选两种顺序各跑一次，报 agent 的 pick 随位置变化的幅度；这是"agent 不像人"的另一个直观证据。
4. **多语言**：ES/JP 子集用 `bge-m3` 做检索、用同一 agent prompt 跑 200 条，报买错率是否更高。
5. **商品图片（多模态）**：ESCI 没有图片字段，需要从 Reviews 2023 meta 的 `images` 拿 URL；只做 10 个案例展示"看图后是否改判"，不做统计。
6. **真实 agent 对照**：手动在 ChatGPT shopping / Perplexity 上跑 20 个 query，看其选择与我们模拟 agent 的一致性，作为"模拟 agent 有效性"的小样本验证。

---

## 验证（计划执行后如何确认）

- `make data` 打印的 query / 标注数与 README 一致（29,844 / 601,354）。
- `make index && make eval-retrieval` 产出 `results/retrieval_eval.json`，hybrid nDCG 高于随机且与 ranx 校验一致。
- `make agents --limit 50` 的 JSON 合法率 ≥ 99%，然后全量两模型，`results/usage.jsonl` 费用在预算内。
- `make diagnose && make rewrite` 产出 ≥ 150 个商品的 before/after；stuffed 版 100% 被忠实度检查标红。
- 断网状态下 `streamlit run app/Home.py` 三页全部可用。
