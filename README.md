# eco-geo

一个可通过 **GitHub Actions 一键运行** 的品牌 GEO 评估骨架项目。

它把 GEO 评估拆成四层：
- Visibility：品牌在生成式回答里的存在感
- Inclusion：品牌内容被抓取、理解、纳入候选的程度
- Cognition：AI 对品牌的定义、属性、叙事是否准确
- Outcome：AI 流量、转化、品牌搜索提升等业务结果

运行后会生成：
- `report.md`：管理层可读报告
- `report.json`：结构化原始结果
- `summary.json`：方便接后续 dashboard

## 目录

- `config/brand.yaml`：品牌定义、维度权重、阈值
- `config/benchmark.yaml`：评估样本与技术/业务快照
- `scripts/run_geo_eval.py`：评分与报告生成脚本
- `.github/workflows/run-geo-eval.yml`：手动触发的一键工作流

## 怎么用

### 1) 填配置
先编辑：
- `config/brand.yaml`
- `config/benchmark.yaml`

### 2) 在 GitHub Settings 里配置 DeepSeek Secret
仓库里不要直接写 key。

请到：
`Settings -> Secrets and variables -> Actions -> New repository secret`

至少创建：
- `DEEPSEEK_API_KEY`：你的 DeepSeek API Secret

可选：
- `DEEPSEEK_BASE_URL`：默认不填也能跑；只有在你要切换兼容端点时才需要

如果没有配置 `DEEPSEEK_API_KEY`，工作流仍然会运行，只是报告里的管理摘要会退回到规则生成版本。

### 3) 一键运行
到 GitHub：
`Actions -> Run GEO Evaluation -> Run workflow`

默认就会读取：
- `config/brand.yaml`
- `config/benchmark.yaml`

### 4) 看结果
运行完成后，在该次 workflow run 的 artifact 里下载：
- `geo-report/report.md`
- `geo-report/report.json`
- `geo-report/summary.json`

## benchmark 数据怎么填

### visibility_observations
一条 observation 代表一次：
`引擎 x query x 市场 x 设备 x 轮次`

建议字段：
- `mentioned`
- `first_party_cited`
- `citation_count`
- `total_citations`
- `preferred_brand`
- `position_bucket`
- `primary_source`
- `third_party_corroboration`

### cognition_observations
建议字段：
- `definition_accuracy`：0 / 1 / 2
- `attribute_recall_ratio`：0-1
- `narrative_alignment_ratio`：0-1
- `error_count`
- `high_risk_error`

### inclusion_snapshot
直接填阶段性快照：
- `crawl_reach_rate`
- `index_coverage`
- `structured_data_validity`
- `entity_consistency_score`
- `knowledge_asset_completeness`
- `llm_asset_availability`

### outcome_snapshot
建议同时填当前值和目标值：
- `ai_sessions`
- `engaged_session_rate`
- `conversion_rate`
- `assisted_conversion_count`
- `branded_search_lift`
- `benchmark_targets`

## 适合怎么扩展

你后面可以继续往这个骨架里加：
- 真正的 query 抓取器
- Search Console / GA /日志导入
- 多品牌、多市场对比
- 自动把报告发布到 Pages / Notion / 飞书
