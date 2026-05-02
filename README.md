# eco-geo

一个可以通过 **GitHub Actions 一键运行** 的品牌 GEO 评估项目。

这次已经改成两种模式都支持：
- **自动模式**：不给 benchmark 也能跑。脚本会调用 DeepSeek，先推断品牌品类、竞品集合、query panel，再生成一版初始 GEO evaluation。
- **人工增强模式**：如果你后面已经有真实 query 观测、Search Console、GA、日志或人工审计数据，可以用 `config/manual_benchmark.example.yaml` 这种覆盖文件，替换 DeepSeek 初版里的某些字段。

运行后会：
- 生成 `report.md`
- 生成 `report.json`
- 生成 `summary.json`
- 生成 `benchmark.generated.json`
- 生成 `dashboard.html`
- **把报告直接 commit 回 repo 的 `reports/` 目录**

## 目录

- `config/brand.yaml`：品牌配置
- `config/manual_benchmark.example.yaml`：可选的人工 benchmark 覆盖模板
- `scripts/run_geo_eval.py`：DeepSeek 辅助评估与 dashboard 生成脚本
- `.github/workflows/run-geo-eval.yml`：一键运行的 GitHub Actions

## 先做什么

### 1) 配 DeepSeek Secret
到：
`Settings -> Secrets and variables -> Actions -> New repository secret`

至少配置：
- `DEEPSEEK_API_KEY`

可选：
- `DEEPSEEK_BASE_URL`

### 2) 改品牌配置
编辑 `config/brand.yaml`。

最少填这些：
- `brand.name`
- `brand.website`
- `brand.market`
- `brand.category`（建议填，能提高推断质量）

## 怎么跑

到：
`Actions -> Run GEO Evaluation -> Run workflow`

默认会：
- 读取 `config/brand.yaml`
- 用 DeepSeek 生成初版竞品集合、query panel、evidence map、journey gap matrix 和四层 GEO 评分
- 把结果 commit 到 `reports/latest`

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

### `reports/latest/report.md`
管理层可直接看的文本报告。

### `reports/latest/report.json`
结构化评分结果。

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

只要把 `config/brand.yaml` 写成：
- 品牌名：Kioxia
- 官网：`https://www.kioxia.com`
- 市场：Global 或你要看的区域
- 品类：NAND flash / SSD / enterprise storage

然后直接跑 action。

DeepSeek 会先产出：
- 它推断的竞品集合
- 一组 query panel
- evidence map 和 market pressure
- 初始 visibility / inclusion / cognition / outcome 分数
- 优先动作建议

这不是“真实线上观测结果”，但它是一个**可以立即落地的第一版评估框架、竞争压力图谱和基线 dashboard**。

## 后面适合继续扩展的方向

- 接入 Search Console API
- 接入 GA4 导出
- 接入日志样本
- 做 query panel 的批量人工审计
- 多品牌 / 多市场 / 多月份对比
- 自动生成历史归档到 `reports/history/`
