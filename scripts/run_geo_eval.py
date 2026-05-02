#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

DEFAULT_WEIGHTS = {
    "visibility": 35,
    "inclusion": 25,
    "cognition": 25,
    "outcome": 15,
}
DIMS = ["visibility", "inclusion", "cognition", "outcome"]
EVIDENCE_KEYS = [
    "owned_surface_strength",
    "entity_clarity",
    "content_modularity",
    "trust_signal_density",
    "comparison_page_readiness",
    "faq_readiness",
    "documentation_readiness",
    "pricing_transparency",
    "schema_readiness",
    "narrative_control",
]
MARKET_PRESSURE_KEYS = [
    "peer_activation_index",
    "benchmark_percentile",
    "urgency_score",
    "gap_to_leading_peer",
    "narrative_disadvantage",
]
JOURNEY_STAGES = ["Discover", "Consider", "Validate", "Select", "Expand"]


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def clamp_score(value: Any) -> float:
    try:
        num = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(100.0, num))


def clamp_ratio(value: Any) -> float:
    try:
        num = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, num))


def avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def extract_json(text: str) -> Dict[str, Any]:
    text = strip_code_fence(text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in model response")
    return json.loads(match.group(0))


def infer_stage(score: float) -> str:
    if score >= 80:
        return "Leading"
    if score >= 65:
        return "Active"
    if score >= 45:
        return "Emerging"
    return "Under-activated"


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: Optional[str], model: str):
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for auto benchmark mode")
        self.api_key = api_key
        self.base_url = (base_url or "https://api.deepseek.com/v1/chat/completions").strip()
        self.model = model

    def chat_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return extract_json(content)


def build_auto_benchmark_prompt(brand_cfg: Dict[str, Any]) -> str:
    brand = brand_cfg.get("brand", {})
    output_schema = {
        "brand_profile": {
            "brand_name": "",
            "official_website": "",
            "market": "",
            "region": "",
            "language": "",
            "inferred_category": "",
            "brand_summary": "",
            "confidence": 0.0,
            "uncertainties": [""],
        },
        "competitors": [
            {
                "name": "",
                "why_in_set": "",
                "confidence": 0.0,
                "geo_maturity_stage": "",
                "overall_score_estimate": 0,
                "dimension_scores": {
                    "visibility": 0,
                    "inclusion": 0,
                    "cognition": 0,
                    "outcome": 0,
                },
                "strengths": [""],
                "evidence_signals": [""],
            }
        ],
        "query_panel": [
            {
                "type": "brand",
                "query": "",
                "intent": "",
                "funnel_stage": "Consider",
                "importance": 0,
            }
        ],
        "evidence_map": {
            "owned_surface_strength": 0,
            "entity_clarity": 0,
            "content_modularity": 0,
            "trust_signal_density": 0,
            "comparison_page_readiness": 0,
            "faq_readiness": 0,
            "documentation_readiness": 0,
            "pricing_transparency": 0,
            "schema_readiness": 0,
            "narrative_control": 0,
        },
        "market_pressure": {
            "peer_activation_index": 0,
            "benchmark_percentile": 0,
            "urgency_score": 0,
            "gap_to_leading_peer": 0,
            "narrative_disadvantage": 0,
        },
        "journey_gap_matrix": [
            {
                "stage": "Discover",
                "current_strength": 0,
                "competitor_pressure": 0,
                "opportunity": 0,
                "notes": "",
            }
        ],
        "geo_evaluation": {
            "methodology_note": "",
            "visibility": {
                "score": 0,
                "metrics": {
                    "brand_mention_likelihood": 0,
                    "first_party_citation_likelihood": 0,
                    "comparative_presence": 0,
                    "weighted_visibility": 0,
                },
                "rationale": "",
                "confidence": 0.0,
                "priority_actions": [""],
            },
            "inclusion": {
                "score": 0,
                "metrics": {
                    "crawl_index_readiness": 0,
                    "entity_clarity": 0,
                    "structured_content_readiness": 0,
                    "knowledge_asset_completeness": 0,
                },
                "rationale": "",
                "confidence": 0.0,
                "priority_actions": [""],
            },
            "cognition": {
                "score": 0,
                "metrics": {
                    "definition_accuracy_likelihood": 0,
                    "attribute_recall_likelihood": 0,
                    "narrative_alignment_likelihood": 0,
                    "hallucination_resilience": 0,
                },
                "rationale": "",
                "confidence": 0.0,
                "priority_actions": [""],
            },
            "outcome": {
                "score": 0,
                "metrics": {
                    "visit_intent_capture": 0,
                    "conversion_readiness": 0,
                    "brand_search_lift_potential": 0,
                    "measurement_maturity": 0,
                },
                "rationale": "",
                "confidence": 0.0,
                "priority_actions": [""],
            },
            "strengths": [""],
            "risks": [""],
            "executive_summary": "",
        },
    }
    prompt_lines = [
        "请为品牌 GEO 评估生成一个严谨的 JSON。目标品牌如下：",
        f"- 品牌名: {brand.get('name', '')}",
        f"- 官网: {brand.get('website', '')}",
        f"- 市场: {brand.get('market', 'global')}",
        f"- 区域: {brand.get('region', brand.get('market', 'global'))}",
        f"- 语言: {brand.get('language', 'zh-CN')}",
        f"- 已知品类: {brand.get('category', '')}",
        f"- 已知竞品: {json.dumps(brand.get('competitors', []) or [], ensure_ascii=False)}",
        f"- 希望叙事: {json.dumps(brand.get('narratives', []) or [], ensure_ascii=False)}",
        "",
        "要求：",
        "1. 只输出 JSON，不要 markdown。",
        "2. 不要假装联网，不要写‘最新新闻显示’之类的话；只能基于通用知识、品牌常识、品类逻辑和竞争格局推断一个可复核的初版 benchmark。",
        "3. 要让报告显得成熟、市场化、数据化，但所有分数要克制，不能全部偏高。",
        "4. competitors 要体现 GEO 成熟度差异，并给出他们为什么看起来更强的证据线索。",
        "5. query_panel 至少 24 条，覆盖 brand、category、problem、comparison、use_case、trust 六类，并补充 funnel_stage 与 importance。",
        "6. evidence_map、market_pressure、journey_gap_matrix 必须填写完整。",
        "7. geo_evaluation 四层需要给分数、metrics、rationale、confidence、priority_actions。",
        "8. 需要明确竞争压力： peers 正在加强 GEO，而目标品牌仍有明显空缺；但不要写得像广告文案，要像严肃顾问报告。",
        "9. 输出字段结构必须严格符合下面这个 JSON schema 示例：",
        json.dumps(output_schema, ensure_ascii=False, indent=2),
    ]
    return "\n".join(prompt_lines)


def create_auto_benchmark(brand_cfg: Dict[str, Any], client: DeepSeekClient) -> Dict[str, Any]:
    system_prompt = (
        "你是严格的品牌 GEO 研究员。你不能假装实时联网。"
        "你需要根据通用知识、竞争格局和品牌表述方式，给出一个像管理咨询团队会交付的初版 benchmark。"
        "输出必须是合法 JSON，字段完整，竞争压力和 peer benchmark 要足够鲜明。"
    )
    return client.chat_json(system_prompt, build_auto_benchmark_prompt(brand_cfg))


def normalize_dimension(raw: Dict[str, Any], key_metrics: List[str]) -> Dict[str, Any]:
    metrics = raw.get("metrics", {}) if isinstance(raw, dict) else {}
    return {
        "score": clamp_score(raw.get("score", 0) if isinstance(raw, dict) else 0),
        "metrics": {name: clamp_score(metrics.get(name, 0)) for name in key_metrics},
        "rationale": str(raw.get("rationale", "") if isinstance(raw, dict) else "").strip(),
        "confidence": clamp_ratio(raw.get("confidence", 0) if isinstance(raw, dict) else 0),
        "priority_actions": [str(x) for x in (raw.get("priority_actions", []) if isinstance(raw, dict) else [])][:5],
    }


def normalize_competitors(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    result: List[Dict[str, Any]] = []
    for item in items[:6]:
        if not isinstance(item, dict):
            continue
        dim_scores_raw = item.get("dimension_scores", {}) if isinstance(item.get("dimension_scores", {}), dict) else {}
        dim_scores = {dim: clamp_score(dim_scores_raw.get(dim, item.get(dim, 0))) for dim in DIMS}
        overall = clamp_score(item.get("overall_score_estimate", avg(list(dim_scores.values()))))
        result.append(
            {
                "name": str(item.get("name", "Unknown competitor")),
                "why_in_set": str(item.get("why_in_set", "")).strip(),
                "confidence": clamp_ratio(item.get("confidence", 0)),
                "geo_maturity_stage": str(item.get("geo_maturity_stage", infer_stage(overall))).strip() or infer_stage(overall),
                "overall_score_estimate": overall,
                "dimension_scores": dim_scores,
                "strengths": [str(x) for x in item.get("strengths", [])][:4],
                "evidence_signals": [str(x) for x in item.get("evidence_signals", [])][:5],
            }
        )
    return sorted(result, key=lambda x: x["overall_score_estimate"], reverse=True)


def normalize_query_panel(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    panel: List[Dict[str, Any]] = []
    for item in items[:60]:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        panel.append(
            {
                "type": str(item.get("type", "other")).strip() or "other",
                "query": query,
                "intent": str(item.get("intent", "")).strip(),
                "funnel_stage": str(item.get("funnel_stage", "Consider")).strip() or "Consider",
                "importance": clamp_score(item.get("importance", 50)),
            }
        )
    return panel


def normalize_numeric_map(raw: Any, keys: List[str]) -> Dict[str, float]:
    if not isinstance(raw, dict):
        raw = {}
    return {key: clamp_score(raw.get(key, 0)) for key in keys}


def normalize_journey_matrix(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        items = []
    rows: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage", "")).strip() or "Other"
        rows[stage] = {
            "stage": stage,
            "current_strength": clamp_score(item.get("current_strength", 0)),
            "competitor_pressure": clamp_score(item.get("competitor_pressure", 0)),
            "opportunity": clamp_score(item.get("opportunity", 0)),
            "notes": str(item.get("notes", "")).strip(),
        }
    for stage in JOURNEY_STAGES:
        rows.setdefault(
            stage,
            {
                "stage": stage,
                "current_strength": 0.0,
                "competitor_pressure": 0.0,
                "opportunity": 0.0,
                "notes": "",
            },
        )
    return [rows[stage] for stage in JOURNEY_STAGES] + [v for k, v in rows.items() if k not in JOURNEY_STAGES]


def merge_manual_benchmark(auto: Dict[str, Any], manual: Dict[str, Any]) -> Dict[str, Any]:
    if not manual:
        return auto
    merged = json.loads(json.dumps(auto, ensure_ascii=False))
    for key, value in manual.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def build_report(brand_cfg: Dict[str, Any], benchmark: Dict[str, Any]) -> Dict[str, Any]:
    weights = brand_cfg.get("weights", DEFAULT_WEIGHTS)
    healthy = float(brand_cfg.get("thresholds", {}).get("healthy", 75))
    warning = float(brand_cfg.get("thresholds", {}).get("warning", 55))
    geo_eval = benchmark.get("geo_evaluation", {}) if isinstance(benchmark.get("geo_evaluation", {}), dict) else {}
    brand_profile = benchmark.get("brand_profile", {}) if isinstance(benchmark.get("brand_profile", {}), dict) else {}

    dims = {
        "visibility": normalize_dimension(geo_eval.get("visibility", {}), ["brand_mention_likelihood", "first_party_citation_likelihood", "comparative_presence", "weighted_visibility"]),
        "inclusion": normalize_dimension(geo_eval.get("inclusion", {}), ["crawl_index_readiness", "entity_clarity", "structured_content_readiness", "knowledge_asset_completeness"]),
        "cognition": normalize_dimension(geo_eval.get("cognition", {}), ["definition_accuracy_likelihood", "attribute_recall_likelihood", "narrative_alignment_likelihood", "hallucination_resilience"]),
        "outcome": normalize_dimension(geo_eval.get("outcome", {}), ["visit_intent_capture", "conversion_readiness", "brand_search_lift_potential", "measurement_maturity"]),
    }

    total_weight = sum(float(v) for v in weights.values()) or 1.0
    overall = round(sum(dims[k]["score"] * (float(weights.get(k, 0)) / total_weight) for k in DIMS), 2)

    def level(score: float) -> str:
        if score >= healthy:
            return "healthy"
        if score >= warning:
            return "watch"
        return "risk"

    competitors = normalize_competitors(benchmark.get("competitors", []))
    query_panel = normalize_query_panel(benchmark.get("query_panel", []))
    evidence_map = normalize_numeric_map(benchmark.get("evidence_map", {}), EVIDENCE_KEYS)
    market_pressure = normalize_numeric_map(benchmark.get("market_pressure", {}), MARKET_PRESSURE_KEYS)
    journey_gap_matrix = normalize_journey_matrix(benchmark.get("journey_gap_matrix", []))

    if competitors:
        top_peer = competitors[0]
        peer_avg = round(avg([c["overall_score_estimate"] for c in competitors]), 2)
        competitive_gap = round(top_peer["overall_score_estimate"] - overall, 2)
        if market_pressure["gap_to_leading_peer"] == 0:
            market_pressure["gap_to_leading_peer"] = max(0.0, competitive_gap)
        if market_pressure["benchmark_percentile"] == 0:
            lower = sum(1 for c in competitors if c["overall_score_estimate"] < overall)
            market_pressure["benchmark_percentile"] = round((lower / max(1, len(competitors))) * 100, 2)
    else:
        top_peer = None
        peer_avg = 0.0
        competitive_gap = 0.0

    if market_pressure["peer_activation_index"] == 0 and competitors:
        market_pressure["peer_activation_index"] = round(avg([c["overall_score_estimate"] for c in competitors]), 2)
    if market_pressure["urgency_score"] == 0:
        market_pressure["urgency_score"] = round(min(100.0, max(100.0 - overall, competitive_gap + 40.0)), 2)
    if market_pressure["narrative_disadvantage"] == 0:
        market_pressure["narrative_disadvantage"] = round(max(0.0, avg([dims["cognition"]["score"], dims["visibility"]["score"]]) * -1 + 100.0), 2)

    confidence_avg = round(avg([dims[k]["confidence"] * 100 for k in DIMS]), 1)
    query_type_counts = Counter([q["type"] for q in query_panel])
    funnel_counts = Counter([q["funnel_stage"] for q in query_panel])
    query_summary = {
        "total_queries": len(query_panel),
        "types": dict(query_type_counts),
        "funnel_stages": dict(funnel_counts),
        "avg_importance": round(avg([q["importance"] for q in query_panel]), 2),
    }

    return {
        "generated_at": utc_now(),
        "brand": {
            "name": brand_cfg.get("brand", {}).get("name", brand_profile.get("brand_name", "Unknown Brand")),
            "website": brand_cfg.get("brand", {}).get("website", brand_profile.get("official_website", "")),
            "market": brand_cfg.get("brand", {}).get("market", brand_profile.get("market", "")),
            "category": brand_cfg.get("brand", {}).get("category", brand_profile.get("inferred_category", "")),
        },
        "weights": weights,
        "thresholds": {"healthy": healthy, "warning": warning},
        "overall_score": overall,
        "overall_level": level(overall),
        "confidence_avg": confidence_avg,
        "brand_profile": brand_profile,
        "methodology_note": str(geo_eval.get("methodology_note", "Comparative market benchmark, owned-surface review, answer-fit assessment, and competitive pressure modeling.")).strip(),
        "dimensions": {
            key: {
                "score": dims[key]["score"],
                "level": level(dims[key]["score"]),
                "metrics": dims[key]["metrics"],
                "rationale": dims[key]["rationale"],
                "confidence": dims[key]["confidence"],
                "priority_actions": dims[key]["priority_actions"],
            }
            for key in DIMS
        },
        "competitors": competitors,
        "top_peer": top_peer,
        "peer_average_score": peer_avg,
        "competitive_gap": competitive_gap,
        "query_panel": query_panel,
        "query_summary": query_summary,
        "evidence_map": evidence_map,
        "market_pressure": market_pressure,
        "journey_gap_matrix": journey_gap_matrix,
        "strengths": [str(x) for x in geo_eval.get("strengths", [])][:6],
        "risks": [str(x) for x in geo_eval.get("risks", [])][:6],
        "executive_summary": str(geo_eval.get("executive_summary", "")).strip(),
        "limitations": [str(x) for x in brand_profile.get("uncertainties", [])][:8],
    }


def esc(value: Any) -> str:
    return html.escape(str(value))


def pct(value: float) -> str:
    return f"{round(value, 1)}%"


def stage_chip(stage: str) -> str:
    color = {
        "Leading": "#00c389",
        "Active": "#2dd4bf",
        "Emerging": "#f59e0b",
        "Under-activated": "#ef4444",
    }.get(stage, "#94a3b8")
    return f"<span class='chip' style='border-color:{color};color:{color}'>{esc(stage)}</span>"


def metric_bar(label: str, value: float, tone: str = "default", sublabel: str = "") -> str:
    tone_color = {
        "default": "linear-gradient(90deg,#38bdf8,#818cf8)",
        "danger": "linear-gradient(90deg,#fb7185,#f97316)",
        "success": "linear-gradient(90deg,#34d399,#22c55e)",
        "warning": "linear-gradient(90deg,#f59e0b,#eab308)",
    }.get(tone, "linear-gradient(90deg,#38bdf8,#818cf8)")
    sub = f"<div class='subtext'>{esc(sublabel)}</div>" if sublabel else ""
    return (
        f"<div class='bar-row'><div class='bar-meta'><span>{esc(label)}</span><strong>{round(value,1)}</strong></div>"
        f"<div class='bar-shell'><div class='bar-fill' style='width:{max(0,min(100,value))}%;background:{tone_color}'></div></div>{sub}</div>"
    )


def render_markdown(report: Dict[str, Any]) -> str:
    brand = report["brand"]
    lines: List[str] = []
    lines.append(f"# {brand['name']} GEO Command Dashboard")
    lines.append("")
    lines.append(f"- Generated at: {report['generated_at']}")
    lines.append(f"- Website: {brand.get('website', '') or 'N/A'}")
    lines.append(f"- Market: {brand.get('market', '') or 'N/A'}")
    lines.append(f"- Category: {brand.get('category', '') or 'N/A'}")
    lines.append(f"- Overall GEO Score: **{report['overall_score']} / 100** ({report['overall_level']})")
    lines.append(f"- Competitive gap to leading peer: **{report.get('competitive_gap', 0)}**")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(report.get("executive_summary") or "No summary generated.")
    lines.append("")
    lines.append("## Competitive Pressure")
    lines.append("")
    lines.append(f"- Peer activation index: {report['market_pressure']['peer_activation_index']}")
    lines.append(f"- Benchmark percentile: {report['market_pressure']['benchmark_percentile']}")
    lines.append(f"- Urgency score: {report['market_pressure']['urgency_score']}")
    lines.append("")
    lines.append("## Dimension Scorecard")
    lines.append("")
    lines.append("| Dimension | Score | Level | Confidence |")
    lines.append("|---|---:|---|---:|")
    for key in DIMS:
        item = report["dimensions"][key]
        lines.append(f"| {key.title()} | {item['score']} | {item['level']} | {round(item['confidence']*100,1)}% |")
    lines.append("")
    lines.append("## Competitor Leaderboard")
    lines.append("")
    lines.append("| Brand | Stage | Overall | Visibility | Inclusion | Cognition | Outcome |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {brand['name']} | {infer_stage(report['overall_score'])} | {report['overall_score']} | {report['dimensions']['visibility']['score']} | {report['dimensions']['inclusion']['score']} | {report['dimensions']['cognition']['score']} | {report['dimensions']['outcome']['score']} |"
    )
    for comp in report.get("competitors", []):
        ds = comp["dimension_scores"]
        lines.append(
            f"| {comp['name']} | {comp['geo_maturity_stage']} | {comp['overall_score_estimate']} | {ds['visibility']} | {ds['inclusion']} | {ds['cognition']} | {ds['outcome']} |"
        )
    lines.append("")
    lines.append("## Query Universe")
    lines.append("")
    lines.append(f"- Total monitored queries: {report['query_summary']['total_queries']}")
    for key, value in report['query_summary']['types'].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Evidence Map")
    lines.append("")
    for key, value in report["evidence_map"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Priority Actions")
    lines.append("")
    for dim in DIMS:
        lines.append(f"### {dim.title()}")
        for action in report['dimensions'][dim]['priority_actions']:
            lines.append(f"- {action}")
        lines.append("")
    return "\n".join(lines)


def render_dashboard_html(report: Dict[str, Any]) -> str:
    brand = report["brand"]
    dim_rows = "".join(
        f"<div class='card'><div class='section-title'>{esc(dim.title())}</div>"
        + metric_bar("Score", report["dimensions"][dim]["score"], "default", report["dimensions"][dim]["level"])
        + "".join(metric_bar(k.replace('_', ' ').title(), v, "default") for k, v in report["dimensions"][dim]["metrics"].items())
        + ("<div class='narrative'>" + esc(report["dimensions"][dim]["rationale"]) + "</div>" if report["dimensions"][dim]["rationale"] else "")
        + ("<div class='list-title'>Priority actions</div><ul>" + "".join(f"<li>{esc(a)}</li>" for a in report["dimensions"][dim]["priority_actions"]) + "</ul>" if report["dimensions"][dim]["priority_actions"] else "")
        + "</div>"
        for dim in DIMS
    )

    competitor_rows = []
    competitor_rows.append(
        "<tr>"
        f"<td><strong>{esc(brand['name'])}</strong></td>"
        f"<td>{stage_chip(infer_stage(report['overall_score']))}</td>"
        f"<td>{report['overall_score']}</td>"
        f"<td>{report['dimensions']['visibility']['score']}</td>"
        f"<td>{report['dimensions']['inclusion']['score']}</td>"
        f"<td>{report['dimensions']['cognition']['score']}</td>"
        f"<td>{report['dimensions']['outcome']['score']}</td>"
        "<td><span class='muted'>Current client baseline</span></td>"
        "</tr>"
    )
    for comp in report.get("competitors", []):
        signals = ", ".join(comp.get("evidence_signals", [])[:2])
        ds = comp["dimension_scores"]
        competitor_rows.append(
            "<tr>"
            f"<td><strong>{esc(comp['name'])}</strong></td>"
            f"<td>{stage_chip(comp['geo_maturity_stage'])}</td>"
            f"<td>{comp['overall_score_estimate']}</td>"
            f"<td>{ds['visibility']}</td>"
            f"<td>{ds['inclusion']}</td>"
            f"<td>{ds['cognition']}</td>"
            f"<td>{ds['outcome']}</td>"
            f"<td>{esc(signals or comp.get('why_in_set', ''))}</td>"
            "</tr>"
        )

    evidence_cards = "".join(
        f"<div class='mini-card'><div class='mini-title'>{esc(key.replace('_', ' ').title())}</div>{metric_bar('', value, 'success' if value >= 70 else 'warning' if value >= 45 else 'danger')}</div>"
        for key, value in report["evidence_map"].items()
    )

    query_type_cards = "".join(
        f"<div class='mini-card'><div class='kpi-label'>{esc(k)}</div><div class='kpi-value'>{v}</div></div>"
        for k, v in report["query_summary"]["types"].items()
    )
    funnel_cards = "".join(
        f"<div class='mini-card'><div class='kpi-label'>{esc(k)}</div><div class='kpi-value'>{v}</div></div>"
        for k, v in report["query_summary"]["funnel_stages"].items()
    )

    query_table_rows = "".join(
        f"<tr><td>{esc(item['type'])}</td><td>{esc(item['funnel_stage'])}</td><td>{esc(item['query'])}</td><td>{round(item['importance'],1)}</td></tr>"
        for item in report.get("query_panel", [])[:30]
    )

    journey_rows = "".join(
        f"<div class='journey-card'><div class='section-title'>{esc(item['stage'])}</div>"
        + metric_bar("Current strength", item["current_strength"], "default")
        + metric_bar("Competitor pressure", item["competitor_pressure"], "danger")
        + metric_bar("Opportunity", item["opportunity"], "success")
        + (f"<div class='narrative'>{esc(item['notes'])}</div>" if item['notes'] else "")
        + "</div>"
        for item in report.get("journey_gap_matrix", [])
    )

    competitor_cards = "".join(
        "<div class='card'>"
        f"<div class='section-title'>{esc(comp['name'])} {stage_chip(comp['geo_maturity_stage'])}</div>"
        f"<div class='narrative'>{esc(comp.get('why_in_set', ''))}</div>"
        + "".join(metric_bar(dim.title(), comp['dimension_scores'][dim], 'default') for dim in DIMS)
        + ("<div class='list-title'>What makes them look advanced</div><ul>" + "".join(f"<li>{esc(x)}</li>" for x in comp.get('strengths', [])[:3]) + "</ul>" if comp.get('strengths') else "")
        + ("<div class='list-title'>Visible signals</div><ul>" + "".join(f"<li>{esc(x)}</li>" for x in comp.get('evidence_signals', [])[:4]) + "</ul>" if comp.get('evidence_signals') else "")
        + "</div>"
        for comp in report.get("competitors", [])[:4]
    )

    risk_items = "".join(f"<li>{esc(x)}</li>" for x in report.get("risks", [])) or "<li>No major risks listed.</li>"
    strength_items = "".join(f"<li>{esc(x)}</li>" for x in report.get("strengths", [])) or "<li>No major strengths listed.</li>"

    pressure = report["market_pressure"]
    top_peer_name = report["top_peer"]["name"] if report.get("top_peer") else "N/A"

    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>{esc(brand['name'])} GEO Dashboard</title>
<style>
:root {{ --bg:#07111f; --panel:#0f1b2d; --panel2:#111f34; --line:#22324a; --text:#e7eefb; --muted:#90a0b7; --accent:#60a5fa; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:linear-gradient(180deg,#08111d,#0b1325 40%,#08111d); color:var(--text); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif; }}
.container {{ max-width:1440px; margin:0 auto; padding:24px; }}
.hero {{ display:grid; grid-template-columns:2fr 1fr; gap:20px; margin-bottom:20px; }}
.card, .hero-card, .mini-card, .journey-card {{ background:rgba(15,27,45,.9); border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:0 10px 30px rgba(0,0,0,.18); }}
.hero-card h1 {{ margin:0 0 8px; font-size:34px; }}
.hero-card p {{ color:var(--muted); line-height:1.55; }}
.kpi-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:14px; }}
.kpi, .mini-card {{ background:rgba(17,31,52,.8); border:1px solid var(--line); border-radius:16px; padding:14px; }}
.kpi-label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
.kpi-value {{ font-size:28px; font-weight:700; margin-top:6px; }}
.kpi-sub {{ color:var(--muted); font-size:12px; margin-top:4px; }}
.grid-2 {{ display:grid; grid-template-columns:1.1fr .9fr; gap:20px; margin-bottom:20px; }}
.grid-4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:20px; }}
.grid-5 {{ display:grid; grid-template-columns:repeat(5,1fr); gap:16px; margin-bottom:20px; }}
.section-title {{ font-size:18px; font-weight:700; margin-bottom:12px; }}
.subtext, .muted, .narrative {{ color:var(--muted); line-height:1.55; }}
.bar-row {{ margin:10px 0 14px; }}
.bar-meta {{ display:flex; justify-content:space-between; gap:12px; margin-bottom:6px; font-size:13px; }}
.bar-shell {{ height:10px; border-radius:999px; background:#0b1424; overflow:hidden; border:1px solid #1f2d45; }}
.bar-fill {{ height:100%; border-radius:999px; }}
.chip {{ display:inline-flex; align-items:center; padding:4px 9px; border:1px solid; border-radius:999px; font-size:12px; margin-left:8px; }}
.table-wrap {{ overflow:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th, td {{ padding:12px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ color:#b9c7dd; font-weight:600; }}
.list-title {{ margin-top:14px; margin-bottom:8px; color:#c9d6ea; font-weight:600; }}
ul {{ margin:8px 0 0 18px; padding:0; }}
li {{ margin:6px 0; color:#dbe6f7; }}
.footer-note {{ color:var(--muted); font-size:12px; margin-top:16px; }}
@media (max-width: 1100px) {{ .hero,.grid-2,.grid-4,.grid-5 {{ grid-template-columns:1fr; }} .kpi-grid {{ grid-template-columns:repeat(2,1fr); }} }}
@media (max-width: 720px) {{ .kpi-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class='container'>
  <div class='hero'>
    <div class='hero-card'>
      <div class='kpi-label'>GEO command dashboard</div>
      <h1>{esc(brand['name'])}</h1>
      <p>{esc(report.get('executive_summary') or report['brand_profile'].get('brand_summary', ''))}</p>
      <div class='kpi-grid'>
        <div class='kpi'><div class='kpi-label'>Overall GEO score</div><div class='kpi-value'>{report['overall_score']}</div><div class='kpi-sub'>{esc(report['overall_level'])}</div></div>
        <div class='kpi'><div class='kpi-label'>Gap to leading peer</div><div class='kpi-value'>{round(report.get('competitive_gap',0),1)}</div><div class='kpi-sub'>Top peer: {esc(top_peer_name)}</div></div>
        <div class='kpi'><div class='kpi-label'>Peer activation index</div><div class='kpi-value'>{round(pressure['peer_activation_index'],1)}</div><div class='kpi-sub'>Peers are already showing stronger GEO readiness</div></div>
        <div class='kpi'><div class='kpi-label'>Urgency score</div><div class='kpi-value'>{round(pressure['urgency_score'],1)}</div><div class='kpi-sub'>Higher means competitive pressure is building</div></div>
      </div>
    </div>
    <div class='hero-card'>
      <div class='section-title'>Market benchmark pulse</div>
      {metric_bar('Benchmark percentile', pressure['benchmark_percentile'], 'warning', 'Lower percentile means more peers are ahead')}
      {metric_bar('Narrative disadvantage', pressure['narrative_disadvantage'], 'danger', 'Peers appear more legible to answer engines')}
      {metric_bar('Confidence level', report['confidence_avg'], 'success', 'Benchmark consistency across dimensions')}
      <div class='footer-note'>This dashboard is designed for executive decision-making and competitive GEO prioritization.</div>
    </div>
  </div>

  <div class='grid-2'>
    <div class='card'>
      <div class='section-title'>Competitive leaderboard</div>
      <div class='table-wrap'>
        <table>
          <thead><tr><th>Brand</th><th>Stage</th><th>Overall</th><th>Visibility</th><th>Inclusion</th><th>Cognition</th><th>Outcome</th><th>Observed pressure signal</th></tr></thead>
          <tbody>{''.join(competitor_rows)}</tbody>
        </table>
      </div>
    </div>
    <div class='card'>
      <div class='section-title'>Competitive pressure narrative</div>
      <div class='narrative'>Other players in the competitive set are already showing stronger GEO activation patterns. The current brand is still under-activated in the surfaces and answer-fit behaviors that modern answer engines reward, which increases the risk of ceding narrative control to peers.</div>
      {metric_bar('Peer average score', report.get('peer_average_score', 0), 'warning')}
      {metric_bar('Client baseline', report['overall_score'], 'default')}
      {metric_bar('Leading peer', report['top_peer']['overall_score_estimate'] if report.get('top_peer') else 0, 'success')}
      <div class='list-title'>Why this matters now</div>
      <ul>{risk_items}</ul>
    </div>
  </div>

  <div class='grid-4'>{dim_rows}</div>

  <div class='grid-2'>
    <div class='card'>
      <div class='section-title'>Evidence map</div>
      <div class='grid-5'>{evidence_cards}</div>
    </div>
    <div class='card'>
      <div class='section-title'>Query universe</div>
      <div class='grid-2'>{query_type_cards}{funnel_cards}</div>
      <div class='footer-note'>Total queries: {report['query_summary']['total_queries']} | Avg importance: {report['query_summary']['avg_importance']}</div>
    </div>
  </div>

  <div class='card' style='margin-bottom:20px'>
    <div class='section-title'>Journey gap matrix</div>
    <div class='grid-5'>{journey_rows}</div>
  </div>

  <div class='grid-2'>
    <div class='card'>
      <div class='section-title'>Representative query panel</div>
      <div class='table-wrap'>
        <table>
          <thead><tr><th>Type</th><th>Stage</th><th>Query</th><th>Importance</th></tr></thead>
          <tbody>{query_table_rows}</tbody>
        </table>
      </div>
    </div>
    <div class='card'>
      <div class='section-title'>Current strengths</div>
      <ul>{strength_items}</ul>
      <div class='list-title'>Immediate action bias</div>
      <ul>
        {''.join(f"<li>{esc(a)}</li>" for dim in DIMS for a in report['dimensions'][dim]['priority_actions'][:2])}
      </ul>
    </div>
  </div>

  <div class='grid-4'>{competitor_cards}</div>

  <div class='footer-note'>Generated at {esc(report['generated_at'])}. {esc(report.get('methodology_note',''))}</div>
</div>
</body>
</html>"""


def write_outputs(output_dir: Path, report: Dict[str, Any], benchmark: Dict[str, Any]) -> None:
    ensure_dir(output_dir)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (output_dir / "dashboard.html").write_text(render_dashboard_html(report), encoding="utf-8")
    (output_dir / "benchmark.generated.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "brand": report["brand"],
        "overall_score": report["overall_score"],
        "overall_level": report["overall_level"],
        "generated_at": report["generated_at"],
        "competitive_gap": report.get("competitive_gap", 0),
        "top_peer": report.get("top_peer", {}).get("name") if report.get("top_peer") else None,
        "dimensions": {key: {"score": value["score"], "level": value["level"]} for key, value in report["dimensions"].items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def maybe_git_commit(repo_root: Path, source_dir: Path, target_dir: Path, message: str) -> Dict[str, Any]:
    ensure_dir(target_dir.parent)
    if target_dir.exists():
        subprocess.run(["rm", "-rf", str(target_dir)], check=True)
    subprocess.run(["mkdir", "-p", str(target_dir)], check=True)
    subprocess.run(["cp", "-R", f"{source_dir}/.", str(target_dir)], check=True)
    subprocess.run(["git", "add", str(target_dir)], cwd=repo_root, check=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, check=True, capture_output=True, text=True)
    if not status.stdout.strip():
        return {"committed": False, "target_dir": str(target_dir)}
    subprocess.run(["git", "commit", "-m", message], cwd=repo_root, check=True)
    return {"committed": True, "target_dir": str(target_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek-assisted GEO evaluation runner")
    parser.add_argument("--brand-config", default="config/brand.yaml")
    parser.add_argument("--manual-benchmark", default="")
    parser.add_argument("--output", default="dist/report")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--auto-benchmark", action="store_true")
    parser.add_argument("--commit-report", action="store_true")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--report-subdir", default="reports/latest")
    parser.add_argument("--commit-message", default="chore: update GEO evaluation report")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    brand_cfg = load_yaml(Path(args.brand_config))
    manual_benchmark = load_yaml(Path(args.manual_benchmark)) if args.manual_benchmark else {}

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "")
    auto_benchmark: Dict[str, Any] = {}
    if args.auto_benchmark or not manual_benchmark:
        client = DeepSeekClient(api_key, base_url, args.model)
        auto_benchmark = create_auto_benchmark(brand_cfg, client)

    merged_benchmark = merge_manual_benchmark(auto_benchmark, manual_benchmark)
    report = build_report(brand_cfg, merged_benchmark)
    output_dir = Path(args.output)
    write_outputs(output_dir, report, merged_benchmark)

    commit_info: Dict[str, Any] = {"committed": False}
    if args.commit_report:
        commit_info = maybe_git_commit(
            repo_root=repo_root,
            source_dir=output_dir.resolve(),
            target_dir=(repo_root / args.report_subdir),
            message=args.commit_message,
        )

    payload = {
        "brand": report["brand"]["name"],
        "overall_score": report["overall_score"],
        "overall_level": report["overall_level"],
        "output_dir": str(output_dir.resolve()),
        "commit": commit_info,
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise
