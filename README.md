# eco-geo

一个可以通过 **GitHub Actions 一键运行** 的品牌 GEO 评估与咨询交付项目。

这个 repo 现在支持两条主线：

- **自动 benchmark 模式**：不给 benchmark 也能跑。脚本会调用 DeepSeek，先推断品牌品类、竞品集合、query panel，再生成一版初始 GEO evaluation。
- **full-detail monitoring 模式**：先生成 query universe，再逐条调用 DeepSeek 做 GEO probe，保留完整调用账本，并输出 aggregate metrics、dashboard 和咨询级 PDF。

运行后会生成：

- `report.md`
- `summary.json`
- `dashboard.html`
- `deepseek_calls.json`（full-detail monitoring 调用账本）
- `aggregate_metrics.json`（full-detail monitoring 聚合指标）
- `synthesis.json`（管理层摘要与行动建议）
- `executive_report.pdf`（可交付的咨询级 PDF）
- `executive_report.tex`（PDF 的可审计 LaTeX 源稿）
- **把报告直接 commit 回 repo 的 `reports/` 目录**

## 目录

- `config/brand.yaml`：品牌配置
- `config/manual_benchmark.example.yaml`：可选的人工 benchmark 覆盖模板
- `scripts/run_geo_eval.py`：DeepSeek 辅助评估与 dashboard 生成脚本
- `scripts/run_geo_eval_v4_safe.py`：full-detail GEO monitoring runner
- `scripts/build_consulting_pdf.py`：从 JSON 结果生成咨询级 PDF 和 LaTeX 源稿
- `.github/workflows/run-geo-eval.yml`：基础 GEO evaluation workflow
- `.github/workflows/run-geo-eval-v4-safe.yml`：full-detail monitoring + consulting PDF workflow

## 先做什么

### 1) 配 DeepSeek Secret

到：

`Settings -> Secrets and variables -> Actions -> New repository secret`

至少配置：

- `DEEPSEEK_API_KEY`

可选：

- `DEEPSEEK_BASE_URL`

### 2) 选择运行模式

#### 基础 benchmark 模式

编辑 `config/brand.yaml`。

最少填这些：

- `brand.name`
- `brand.website`
- `brand.market`
- `brand.category`（建议填，能提高推断质量）

然后运行：

`Actions -> Run GEO Evaluation -> Run workflow`

#### full-detail monitoring + 咨询级 PDF 模式

运行：

`Actions -> Run GEO Evaluation v4 Safe -> Run workflow`

输入：

- `brand_name`：品牌名
- `brand_brief`：一句话品牌简介
- `website`：官网，可空
- `monitor_runs`：监测 query 数量，例如 20 / 60 / 120
- `report_subdir`：报告提交目录，默认 `reports/latest`
- `deepseek_model`：DeepSeek 模型名，默认 `deepseek-chat`

workflow 会先跑完整 monitoring，再生成 `executive_report.pdf` 和 `executive_report.tex`，并把它们复制到 `reports/latest` 后 commit。

## 咨询级 PDF 说明

`reports/latest/executive_report.pdf` 是面向客户或管理层的交付物，不是原始日志 dump。它默认包含：

1. 封面与运行 KPI：DeepSeek calls、monitoring probes、成功率、mention rate、recommendation rate、token usage。
2. Executive Summary：来自 `synthesis.json` 的管理层摘要。
3. GEO Scorecard：visibility、inclusion、cognition、outcome 四维分数与风险等级。
4. Query Universe Coverage：按 query type 和 funnel stage 展示监测覆盖。
5. Competitive Pressure：竞品提及频次、推荐品牌频次。
6. Evidence Gaps and Risk：first-party source needed、citation likelihood、answer confidence、risk flag samples。
7. Strategic Findings：top findings 与 90-day action roadmap。
8. Operations Ledger：调用阶段、成功率、耗时、token 汇总。
9. Appendix：抽样展示 representative monitoring results。

### 脱敏策略

PDF 不直接打印完整 prompt 和 response_text。完整原始 trace 仍保留在：

- `deepseek_calls.json`
- `raw_runs.jsonl`

PDF 中只展示 executive-safe summary，并会脱敏：

- API key / bearer token
- email
- 中国大陆手机号

这样可以避免报告过长，也避免把敏感内容带进客户交付 PDF。

### 本地生成咨询级 PDF

已有 `dist/report` 或 `reports/latest` 结果时，可以直接运行：

```bash
python scripts/build_consulting_pdf.py --report-dir reports/latest
```

或显式指定文件：

```bash
python scripts/build_consulting_pdf.py \
  --calls reports/latest/deepseek_calls.json \
  --aggregate reports/latest/aggregate_metrics.json \
  --synthesis reports/latest/synthesis.json \
  --setup reports/latest/research_setup.json \
  --output reports/latest/executive_report.pdf \
  --tex-output reports/latest/executive_report.tex
```

## 产物说明

### `reports/latest/dashboard.html`

可直接给客户或内部业务团队看的 dashboard 版输出，包含：

- competitive leaderboard
- competitive pressure
- evidence map
- query universe
- journey gap matrix
- dimension decomposition
- competitor cards

### `reports/latest/executive_report.pdf`

可交付的咨询级 PDF。适合给客户、老板或销售团队转发。

### `reports/latest/executive_report.tex`

PDF 的 LaTeX 源稿。用于审计、二次编辑或迁移到更高级的 LaTeX 模板。

### `reports/latest/deepseek_calls.json`

完整 DeepSeek 调用账本，包含每次 call 的 stage、query、prompt、response、usage、duration、success 状态等。

### `reports/latest/aggregate_metrics.json`

适合做 dashboard 和 PDF 的结构化指标，包括：

- brand mention rate
- brand recommendation rate
- first-party source needed rate
- query type distribution
- funnel stage distribution
- competitor mention counts
- recommended brand counts
- sentiment distribution
- risk flag samples

### `reports/latest/synthesis.json`

管理层摘要、四维分数、evidence map、top findings 和 priority actions。

### `reports/latest/report.md`

管理层可直接看的文本报告。

### `reports/latest/summary.json`

适合接 dashboard 摘要或历史归档。

### `reports/latest/benchmark.generated.json`

DeepSeek 自动生成的基线：

- brand profile
- competitor set
- query panel
- evidence map
- market pressure
- journey gap matrix
- geo evaluation

## 关于 benchmark 的现实做法

你说得对，很多客户一开始根本没有 benchmark。这个项目现在就是按这个现实来设计的：

### 第 1 阶段：先让 DeepSeek 给出“初版 benchmark”

用于快速起盘：

- 谁是竞品
- 应该监控哪些 query
- 竞品的 GEO 成熟度谁高谁低
- 这个品牌在四层框架里大概会落在哪里
- 哪些动作最值得先做

### 第 2 阶段：再逐步把人工数据覆写进去

后面你可以把这些真实数据逐步补进来：

- query 抽样结果
- 官网 / 帮助中心 / FAQ 盘点
- Search Console
- GA / 站内转化
- 服务器日志
- 人工审阅的 hallucination case

这样项目就从“模型辅助推断”升级到“真实数据驱动评估”。

## Kioxia 这种客户怎么做

你甚至不用先准备 benchmark。

只要运行 v4 safe workflow：

- 品牌名：Kioxia
- 官网：`https://www.kioxia.com`
- 简介：NAND flash / SSD / enterprise storage brand
- monitor runs：60

DeepSeek 会先产出 query universe，再逐条 probing，并生成：

- 完整调用账本
- 竞品提及 / 推荐分布
- visibility / inclusion / cognition / outcome 分数
- 风险样本
- 优先行动建议
- 可交付 PDF

这不是“真实线上观测结果”，但它是一个**可以立即落地的第一版评估框架、竞争压力图谱和基线 consulting deliverable**。

## 后面适合继续扩展的方向

- 接入 Search Console API
- 接入 GA4 导出
- 接入日志样本
- 做 query panel 的批量人工审计
- 多品牌 / 多市场 / 多月份对比
- 自动生成历史归档到 `reports/history/`
- 把 PDF 模板进一步升级成品牌化模板（封面、客户 logo、顾问署名、目录页、图表主题）
