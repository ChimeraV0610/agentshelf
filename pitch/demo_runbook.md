# AgentShelf · 现场演示操作顺序

## 演示前（前一天晚上）
1. `ollama serve` 保持运行；`ollama list` 确认 `qwen2.5:7b`（备选 `qwen2.5:3b`）。
2. 预热缓存：`.venv/bin/python scripts/engine_cli.py --all --buyers 6 --rewrite`。跑完后 6 个货架的默认路径现场全部秒出。
3. `make app` 启动，走一遍默认路径（下面第 2–5 步）确认每步都是缓存命中（Seconds 显示 0）。
4. 断网重走一次，确认无报错。
5. 准备一个 ASIN 做"现场新输入"环节，并提前跑一次让它也进缓存（例如粘贴链接后完整走到补缺口）。

## 现场（90 秒 demo）
1. 侧栏：模型 `ollama:qwen2.5:7b`，货架 "air purifier"，买家 4。
2. Step 0 选 "Pick one of the shelf products"，默认已选中一个弱 listing；读出审计分和第一条 issue："买家搜索用的词它一个都没用"。
3. Step 2 点 "Run simulated buyers"：指着表格读一个买家的理由和"它选了哪个竞品"。核心数字：选中率 vs 公平份额 1/8。
4. Step 3 点 "Rewrite and retest"：左右对比；强调每条要点带原文依据、审核通过；再读复测后的选中率。
5. Step 4 填 1–2 个"商家待确认"事实（如 warranty: 2 years），点 Retest；看 Result 柱状图三阶段。
6. 收尾一句："同一货架、同一批买家、同一个 agent，只改了文本。"

## 评委追问预案
- "agent 是真的吗？" 本地 7B 模型模拟，prompt 只给商品文本；Evidence 页有 ESCI 60 万条人工标注的证据说明文本决定相关性；agent 选谁没有公开标签，所以必须模拟。
- "改写会不会作弊？" 改写器函数不接收任何买家需求；两道忠实度审核；演示里可以展示一个被拦下的编造例子（"32 oz filter capacity"）。
- "Amazon 链接抓不到怎么办？" 先查本地 120 万商品库，再试在线，最后贴文本。
- "为什么慢？" 本地模型每个买家 25–50 秒；换 3B 模型或预热缓存即可，云端 API 则是几秒。

## 故障切换
- Ollama 挂了：侧栏切 "heuristic (offline, not an LLM)"，流程照跑，明说这是启发式替身。
- 页面卡住：侧栏 Reset；或重启 `make app`。
- 彻底崩：放预录视频。
