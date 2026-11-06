# AgentShelf · 5 slides（讲稿要点，数字均来自仓库 results/）

## 1 · Hook：你的下一个顾客是个 bot
- "Your next customer is a bot. It never sees your ad, your photos or your storefront. It reads your listing text and decides."
- Google 的 agent 购物标准、Copilot 结账、Amazon Rufus：AI 替人挑商品，只读标题、要点、描述、属性。
- 受害者：1–20 个 SKU、请不起代理公司的小卖家。他们看不到 agent 为什么没选自己。
- 真实例子（白噪音机货架）：agent 给 Hatch Rest+ 的理由是 "does not state battery life, no fan sound mentioned"。它不是不好，是没说。

## 2 · 证据：上货架靠相关性，被买下靠别的
- Amazon ESCI：60 万条人工相关性标注（US），我们取 test 集 4 万对做逻辑回归。
- 买家搜索词在 listing 里的覆盖率 0–25% → 31% 被判 Exact；75–100% → 61%。标准化系数 0.49，是唯一显著的特征。
- 规格数量、属性数量、要点数、空话比例对"相关性"几乎没影响（系数都在 ±0.05 内）。
- 关键词检索 BM25 在候选内 nDCG 0.824（随机 0.747，官方 baseline 0.83）：说明"能不能被找到"已经是解决得差不多的问题。
- 结论：相关性决定你能不能上货架；agent 在 8 个相关商品里买谁，没有任何公开数据集有标注。所以必须模拟。这就是产品。

## 3 · 产品：AgentShelf，四步，现场端到端（切 demo，90 秒）
- 贴 Amazon 链接 → 自动取 listing、7 个真实竞品（同一搜索词下人工标 Exact 的商品）、模拟买家。
- Step 1 审计：0–100 分，五项：买家原话、可比数字规格、关键属性、结构、空话。
- Step 2 模拟买家：每条真实需求句交给本地 LLM agent，只看文本，顺序打乱，必须给全部 8 个商品排名并写理由。指标：选中率 vs 公平份额 1/8、前 3 率、平均排名。
- Step 3 忠实改写 + 复测：只用原文事实，标题前置品类和决定性参数，要点写成"名称: 事实"，每条带原文依据；两道审核（规则 + LLM 判官）；同一批买家复测。
- Step 4 补缺口：agent 问到但 listing 没写的事实（保修、覆盖面积），商家填上再测。
- 全程本地 Ollama，断网能跑。

## 4 · 结果与诚实性
- 6 个真实货架（空气净化器、婴儿白噪音机、婴儿背带、止吠项圈、1TB SSD、背包），每个货架 6 个买家，改写前后对比：
  - [跑完后填：平均选中率 before → after，top-3 率，平均排名；哪几个货架提升最大]
  - 白噪音机：选中率 17% → [after]，审计 41 → 50，改写一轮通过审核。
- 防作弊三件套：改写器函数不接收任何买家需求；每条改写带原文依据并经两道审核；审核真的会拦——7B 模型曾编出 "32 oz filter capacity, 15m x 15m coverage"，规则层当场拦下。
- 局限直说：agent 是本地 7B 模型模拟，不是 Rufus；买家需求句是我们写的；选中率没有人工标注可对照。

## 5 · 商业与下一步
- 用户：小卖家按 listing 数付费的自助工具；品牌按 "agent share of shelf" 订阅监测。
- 路线：接入真实 agent 的可观察输出；加价格和评论维度；用云端模型把 6 个货架扩到 ESCI 的 3 万个搜索词。
- 收尾："When the customer is a bot, the listing is the store."
