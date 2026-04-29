#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

POSITION_WEIGHTS = {
    "top_half": 1.0,
    "middle": 0.75,
    "bottom_half": 0.5,
    "not_applicable": 0.5,
}


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def pct(value: float) -> float:
    return round(clamp(value) * 100, 2)


def avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def score_definition_accuracy(raw_scores: List[float]) -> float:
    if not raw_scores:
        return 0.0
    return avg([clamp(score / 2.0) for score in raw_scores])


def score_error_rate(error_counts: List[int], high_risk_flags: List[bool]) -> float:
    if not error_counts:
        return 0.0
    penalties = []
    for count, high_risk in zip(error_counts, high_risk_flags):
        penalty = min(count, 3) / 3.0
        if high_risk:
            penalty = min(1.0, penalty + 0.25)
        penalties.append(1.0 - penalty)
    return avg(penalties)


def score_against_target(actual: float, target: float) -> float:
    if target <= 0:
        return 0.0 if actual <= 0 else 1.0
    return clamp(actual / target)


def summarize_level(score_100: float, healthy: float, warning: float) -> str:
    if score_100 >= healthy:
        return "healthy"
    if score_100 >= warning:
        return "watch"
    return "risk"


@dataclass
class DimensionResult:
    name: str
    score_100: float
    metrics: Dict[str, float]
    details: Dict[str, Any]


class GeoEvaluator:
    def __init__(self, brand_config: Dict[str, Any], benchmark: Dict[str, Any]):
        self.brand_config = brand_config
        self.benchmark = benchmark
        self.brand_name = brand_config.get("brand", {}).get("name", "Brand")
        self.weights = brand_config.get(
            "weights",
            {"visibility": 35, "inclusion": 25, "cognition": 25, "outcome": 15},
        )
        thresholds = brand_config.get("thresholds", {})
        self.healthy = float(thresholds.get("healthy", 75))
        self.warning = float(thresholds.get("warning", 55))

    def evaluate_visibility(self) -> DimensionResult:
        observations = self.benchmark.get("visibility_observations", [])
        total = len(observations)
        mentions = sum(1 for item in observations if item.get("mentioned"))
        first_party = sum(1 for item in observations if item.get("first_party_cited"))
        winning = sum(1 for item in observations if item.get("preferred_brand") == self.brand_name)

        citation_shares = []
        weighted_visibility = []
        for item in observations:
            citation_count = float(item.get("citation_count", 0) or 0)
            total_citations = float(item.get("total_citations", 0) or 0)
            citation_share = citation_count / total_citations if total_citations > 0 else 0.0
            citation_shares.append(citation_share)

            observation_score = 0.0
            if item.get("mentioned"):
                observation_score += 1.0
            if item.get("first_party_cited"):
                observation_score += 2.0
            if item.get("primary_source"):
                observation_score += 1.0
            if item.get("third_party_corroboration"):
                observation_score += 1.0
            observation_score *= POSITION_WEIGHTS.get(item.get("position_bucket", "middle"), 0.75)
            weighted_visibility.append(min(observation_score / 5.0, 1.0))

        mention_rate = mentions / total if total else 0.0
        first_party_rate = first_party / total if total else 0.0
        citation_sov = avg(citation_shares)
        win_rate = winning / total if total else 0.0
        weighted_score = avg(weighted_visibility)

        score = (
            mention_rate * 0.25
            + first_party_rate * 0.30
            + citation_sov * 0.20
            + win_rate * 0.10
            + weighted_score * 0.15
        )
        return DimensionResult(
            name="Visibility",
            score_100=round(score * 100, 2),
            metrics={
                "brand_mention_rate": pct(mention_rate),
                "first_party_citation_rate": pct(first_party_rate),
                "citation_share_of_voice": pct(citation_sov),
                "competitive_win_rate": pct(win_rate),
                "weighted_visibility_score": pct(weighted_score),
            },
            details={"observation_count": total},
        )

    def evaluate_inclusion(self) -> DimensionResult:
        snapshot = self.benchmark.get("inclusion_snapshot", {})
        metrics = {
            "crawl_reach_rate": pct(float(snapshot.get("crawl_reach_rate", 0))),
            "index_coverage": pct(float(snapshot.get("index_coverage", 0))),
            "structured_data_validity": pct(float(snapshot.get("structured_data_validity", 0))),
            "entity_consistency_score": pct(float(snapshot.get("entity_consistency_score", 0))),
            "knowledge_asset_completeness": pct(float(snapshot.get("knowledge_asset_completeness", 0))),
            "llm_asset_availability": pct(float(snapshot.get("llm_asset_availability", 0))),
        }
        score = (
            metrics["crawl_reach_rate"] * 0.18
            + metrics["index_coverage"] * 0.22
            + metrics["structured_data_validity"] * 0.18
            + metrics["entity_consistency_score"] * 0.16
            + metrics["knowledge_asset_completeness"] * 0.18
            + metrics["llm_asset_availability"] * 0.08
        ) / 100.0
        return DimensionResult(
            name="Inclusion",
            score_100=round(score * 100, 2),
            metrics=metrics,
            details=snapshot,
        )

    def evaluate_cognition(self) -> DimensionResult:
        observations = self.benchmark.get("cognition_observations", [])
        definition_score = score_definition_accuracy(
            [float(item.get("definition_accuracy", 0)) for item in observations]
        )
        attribute_recall = avg(
            [float(item.get("attribute_recall_ratio", 0)) for item in observations]
        )
        narrative_alignment = avg(
            [float(item.get("narrative_alignment_ratio", 0)) for item in observations]
        )
        error_resilience = score_error_rate(
            [int(item.get("error_count", 0)) for item in observations],
            [bool(item.get("high_risk_error", False)) for item in observations],
        )

        score = (
            definition_score * 0.30
            + attribute_recall * 0.25
            + narrative_alignment * 0.25
            + error_resilience * 0.20
        )
        return DimensionResult(
            name="Cognition",
            score_100=round(score * 100, 2),
            metrics={
                "brand_definition_accuracy": pct(definition_score),
                "attribute_recall": pct(attribute_recall),
                "narrative_alignment": pct(narrative_alignment),
                "error_resilience": pct(error_resilience),
            },
            details={"observation_count": len(observations)},
        )

    def evaluate_outcome(self) -> DimensionResult:
        snapshot = self.benchmark.get("outcome_snapshot", {})
        targets = snapshot.get("benchmark_targets", {})
        ai_sessions_score = score_against_target(
            float(snapshot.get("ai_sessions", 0)), float(targets.get("ai_sessions", 0))
        )
        engaged_score = score_against_target(
            float(snapshot.get("engaged_session_rate", 0)),
            float(targets.get("engaged_session_rate", 0)),
        )
        conversion_score = score_against_target(
            float(snapshot.get("conversion_rate", 0)), float(targets.get("conversion_rate", 0))
        )
        assisted_score = score_against_target(
            float(snapshot.get("assisted_conversion_count", 0)),
            float(targets.get("assisted_conversion_count", 0)),
        )
        branded_lift_score = score_against_target(
            float(snapshot.get("branded_search_lift", 0)),
            float(targets.get("branded_search_lift", 0)),
        )
        score = (
            ai_sessions_score * 0.25
            + engaged_score * 0.20
            + conversion_score * 0.25
            + assisted_score * 0.15
            + branded_lift_score * 0.15
        )
        return DimensionResult(
            name="Outcome",
            score_100=round(score * 100, 2),
            metrics={
                "ai_sessions_vs_target": pct(ai_sessions_score),
                "engaged_session_rate_vs_target": pct(engaged_score),
                "conversion_rate_vs_target": pct(conversion_score),
                "assisted_conversions_vs_target": pct(assisted_score),
                "branded_search_lift_vs_target": pct(branded_lift_score),
            },
            details=snapshot,
        )

    def evaluate(self) -> Dict[str, Any]:
        visibility = self.evaluate_visibility()
        inclusion = self.evaluate_inclusion()
        cognition = self.evaluate_cognition()
        outcome = self.evaluate_outcome()
        dimension_map = {
            "visibility": visibility,
            "inclusion": inclusion,
            "cognition": cognition,
            "outcome": outcome,
        }

        total_weight = sum(float(value) for value in self.weights.values()) or 1.0
        overall = 0.0
        for key, result in dimension_map.items():
            overall += result.score_100 * (float(self.weights.get(key, 0)) / total_weight)

        overall = round(overall, 2)
        return {
            "brand": self.brand_config.get("brand", {}),
            "thresholds": {"healthy": self.healthy, "warning": self.warning},
            "weights": self.weights,
            "overall_score": overall,
            "overall_level": summarize_level(overall, self.healthy, self.warning),
            "dimensions": {
                key: {
                    "name": result.name,
                    "score": result.score_100,
                    "level": summarize_level(result.score_100, self.healthy, self.warning),
                    "metrics": result.metrics,
                    "details": result.details,
                }
                for key, result in dimension_map.items()
            },
            "notes": self.benchmark.get("notes", []),
        }


def generate_rule_based_insights(report: Dict[str, Any]) -> Dict[str, List[str]]:
    dims = report["dimensions"]
    strengths: List[str] = []
    risks: List[str] = []
    actions: List[str] = []

    if dims["visibility"]["metrics"]["first_party_citation_rate"] < 60:
        risks.append("第一方引用率偏低，AI 在回答时更依赖第三方来源，品牌官方叙事控制力不足。")
        actions.append("优先补齐 FAQ、pricing、comparison、trust 页面，并把关键事实改成清晰文本块与表格。")
    else:
        strengths.append("第一方引用率较稳，品牌自有内容已经进入一部分回答依据。")

    if dims["inclusion"]["metrics"]["index_coverage"] < 80:
        risks.append("索引覆盖率仍有缺口，说明关键页面还没有充分进入候选集。")
        actions.append("排查 noindex、canonical、站点地图、内部链接和可渲染文本问题。")
    else:
        strengths.append("抓取与索引底座相对稳健，可继续把重心放到内容与叙事层。")

    if dims["cognition"]["metrics"]["brand_definition_accuracy"] < 70:
        risks.append("品牌定义准确率偏低，AI 容易把品牌定位说窄、说泛或说错。")
        actions.append("建立标准定义句、核心属性字典和 sameAs/Organization/Product 标注，并同步所有官方资料。")
    else:
        strengths.append("品牌定义准确度不错，说明实体识别与核心定位已被部分引擎吸收。")

    if dims["cognition"]["metrics"]["narrative_alignment"] < 60:
        risks.append("希望被记住的 narrative 还没有稳定出现在回答里。")
        actions.append("围绕 3 到 5 个 narrative pillars 重写 about、产品页、案例页与 comparison 页。")

    if dims["outcome"]["score"] < 65:
        risks.append("业务结果层仍弱，当前 GEO 更像可见度建设，尚未稳定转成高质量访问与转化。")
        actions.append("给高引用页面加清晰 CTA、案例证明和落地页承接，并把 Search Console 与 GA 做联动看板。")
    else:
        strengths.append("结果层已有一定转化基础，可以开始做页面分组实验和 query 分层优化。")

    if not strengths:
        strengths.append("当前样本仍偏少，建议继续扩大 query 面板并提高重复抽样次数。")
    if not risks:
        risks.append("暂无显著短板，但仍建议每月固定复盘 query 面板和竞品份额变化。")
    if not actions:
        actions.append("维持现有结构，继续扩充 benchmark 数据与竞品对照。")

    return {
        "strengths": strengths[:3],
        "risks": risks[:4],
        "actions": actions[:4],
    }


def maybe_generate_llm_summary(
    report: Dict[str, Any], brand_config: Dict[str, Any], model: str
) -> Optional[str]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    brand_name = brand_config.get("brand", {}).get("name", "Brand")
    endpoint = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions")
    prompt = {
        "role": "user",
        "content": (
            "你是资深品牌 GEO 顾问。请基于下面 JSON，输出一段 250-400 字的中文管理摘要，"
            "并附 3 条优先动作。要求：只基于输入数据，不要虚构外部事实。\n\n"
            f"品牌：{brand_name}\n"
            f"数据：{json.dumps(report, ensure_ascii=False)}"
        ),
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是严谨的品牌 GEO 分析师。"},
            prompt,
        ],
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        return None
    return choices[0].get("message", {}).get("content")


def render_markdown(report: Dict[str, Any], insights: Dict[str, List[str]], llm_summary: Optional[str]) -> str:
    brand = report.get("brand", {})
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = []
    lines.append(f"# {brand.get('name', 'Brand')} GEO Evaluation Report")
    lines.append("")
    lines.append(f"- Generated at: {generated_at}")
    lines.append(f"- Website: {brand.get('website', 'N/A')}")
    lines.append(f"- Market: {brand.get('market', 'N/A')}")
    lines.append(f"- Overall GEO Score: **{report['overall_score']} / 100** ({report['overall_level']})")
    lines.append("")

    if llm_summary:
        lines.append("## Management Summary")
        lines.append("")
        lines.append(llm_summary.strip())
        lines.append("")

    lines.append("## Scorecard")
    lines.append("")
    lines.append("| Dimension | Score | Level |")
    lines.append("|---|---:|---|")
    for key in ["visibility", "inclusion", "cognition", "outcome"]:
        item = report["dimensions"][key]
        lines.append(f"| {item['name']} | {item['score']} | {item['level']} |")
    lines.append("")

    for key in ["visibility", "inclusion", "cognition", "outcome"]:
        item = report["dimensions"][key]
        lines.append(f"## {item['name']}")
        lines.append("")
        lines.append(f"Score: **{item['score']} / 100** ({item['level']})")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for metric_name, value in item["metrics"].items():
            lines.append(f"| {metric_name} | {value} |")
        lines.append("")

    lines.append("## Strengths")
    lines.append("")
    for bullet in insights["strengths"]:
        lines.append(f"- {bullet}")
    lines.append("")

    lines.append("## Risks")
    lines.append("")
    for bullet in insights["risks"]:
        lines.append(f"- {bullet}")
    lines.append("")

    lines.append("## Priority Actions")
    lines.append("")
    for idx, bullet in enumerate(insights["actions"], start=1):
        lines.append(f"{idx}. {bullet}")
    lines.append("")

    notes = report.get("notes", [])
    if notes:
        lines.append("## Notes")
        lines.append("")
        for bullet in notes:
            lines.append(f"- {bullet}")
        lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append(
        "本报告按照 Visibility / Inclusion / Cognition / Outcome 四层框架计算，"
        "通过观测样本、技术底座快照和业务结果快照，生成 0-100 的 Brand GEO Score。"
    )
    lines.append("")
    return "\n".join(lines)


def write_outputs(output_dir: Path, report: Dict[str, Any], markdown: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")

    summary = {
        "overall_score": report["overall_score"],
        "overall_level": report["overall_level"],
        "dimensions": {
            key: {"score": value["score"], "level": value["level"]}
            for key, value in report["dimensions"].items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GEO evaluation playbook.")
    parser.add_argument("--brand", required=True, help="Path to brand YAML config")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark YAML config")
    parser.add_argument("--output", default="dist/report", help="Output directory")
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek model name")
    args = parser.parse_args()

    brand_config = load_yaml(Path(args.brand))
    benchmark = load_yaml(Path(args.benchmark))

    evaluator = GeoEvaluator(brand_config, benchmark)
    report = evaluator.evaluate()
    insights = generate_rule_based_insights(report)

    llm_summary = None
    try:
        llm_summary = maybe_generate_llm_summary(report, brand_config, args.model)
    except Exception as exc:
        report["llm_summary_error"] = str(exc)

    markdown = render_markdown(report, insights, llm_summary)
    write_outputs(Path(args.output), report, markdown)

    print(json.dumps({
        "overall_score": report["overall_score"],
        "overall_level": report["overall_level"],
        "output": str(Path(args.output).resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
