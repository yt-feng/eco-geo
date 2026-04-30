#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
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
    name = brand.get("name", "")
    website = brand.get("website", "")
    market = brand.get("market", "global")
    region = brand.get("region", market)
    language = brand.get("language", "zh-CN")
    category = brand.get("category", "")
    seed_competitors = brand.get("competitors", []) or []
    narrative_pillars = brand.get("narratives", []) or []

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
            {"name": "", "why_in_set": "", "confidence": 0.0}
        ],
        "query_panel": [
            {"type": "brand", "query": "", "intent": ""}
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
        f"- 品牌名: {name}",
        f"- 官网: {website}",
        f"- 市场: {market}",
        f"- 区域: {region}",
        f"- 语言: {language}",
        f"- 已知品类: {category}",
        f"- 已知竞品: {json.dumps(seed_competitors, ensure_ascii=False)}",
        f"- 希望叙事: {json.dumps(narrative_pillars, ensure_ascii=False)}",
        "",
        "要求：",
        "1. 只输出 JSON，不要 markdown。",
        "2. 不要假装联网，不要写‘最新新闻显示’之类的话。你只能基于通用知识与品牌常识做一个‘可供人工复核的初版 GEO 评估’。",
        "3. 如果不确定，降低 confidence 并把 uncertainty 写清楚。",
        "4. 竞品最多 5 个，优先给同品类、同购买决策集合里的品牌。",
        "5. query_panel 至少返回 18 个 query，分成 brand/category/problem/comparison/use_case/trust 六类。",
        "6. geo_evaluation 需要给四层打分：visibility/inclusion/cognition/outcome，范围 0-100。",
        "7. 每层必须给 metrics、rationale、confidence、priority_actions。",
        "8. 输出字段结构必须严格符合下面这个 JSON schema 示例：",
        json.dumps(output_schema, ensure_ascii=False, indent=2),
    ]
    return "\n".join(prompt_lines)


def create_auto_benchmark(brand_cfg: Dict[str, Any], client: DeepSeekClient) -> Dict[str, Any]:
    system_prompt = (
        "你是严格的品牌 GEO 研究员。"
        "你不能假装实时联网。"
        "你的任务是基于通用知识给出一个可审阅的初版基准评估，"
        "输出必须是合法 JSON，字段完整，分数要有克制。"
    )
    return client.chat_json(system_prompt, build_auto_benchmark_prompt(brand_cfg))


def normalize_dimension(raw: Dict[str, Any], key_metrics: List[str]) -> Dict[str, Any]:
    metrics = raw.get("metrics", {}) if isinstance(raw, dict) else {}
    normalized_metrics: Dict[str, float] = {}
    for name in key_metrics:
        normalized_metrics[name] = clamp_score(metrics.get(name, 0))
    return {
        "score": clamp_score(raw.get("score", 0) if isinstance(raw, dict) else 0),
        "metrics": normalized_metrics,
        "rationale": (raw.get("rationale", "") if isinstance(raw, dict) else "").strip(),
        "confidence": max(0.0, min(1.0, float(raw.get("confidence", 0) if isinstance(raw, dict) else 0))),
        "priority_actions": [str(x) for x in (raw.get("priority_actions", []) if isinstance(raw, dict) else [])][:5],
    }


def merge_manual_benchmark(auto: Dict[str, Any], manual: Dict[str, Any]) -> Dict[str, Any]:
    if not manual:
        return auto
    merged = json.loads(json.dumps(auto, ensure_ascii=False))
    for key in ["brand_profile", "competitors", "query_panel", "geo_evaluation"]:
        if key in manual and manual[key]:
            if isinstance(manual[key], dict) and isinstance(merged.get(key), dict):
                merged[key].update(manual[key])
            else:
                merged[key] = manual[key]
    return merged


def build_report(brand_cfg: Dict[str, Any], benchmark: Dict[str, Any]) -> Dict[str, Any]:
    weights = brand_cfg.get("weights", DEFAULT_WEIGHTS)
    healthy = float(brand_cfg.get("thresholds", {}).get("healthy", 75))
    warning = float(brand_cfg.get("thresholds", {}).get("warning", 55))

    geo_eval = benchmark.get("geo_evaluation", {})
    visibility = normalize_dimension(
        geo_eval.get("visibility", {}),
        [
            "brand_mention_likelihood",
            "first_party_citation_likelihood",
            "comparative_presence",
            "weighted_visibility",
        ],
    )
    inclusion = normalize_dimension(
        geo_eval.get("inclusion", {}),
        [
            "crawl_index_readiness",
            "entity_clarity",
            "structured_content_readiness",
            "knowledge_asset_completeness",
        ],
    )
    cognition = normalize_dimension(
        geo_eval.get("cognition", {}),
        [
            "definition_accuracy_likelihood",
            "attribute_recall_likelihood",
            "narrative_alignment_likelihood",
            "hallucination_resilience",
        ],
    )
    outcome = normalize_dimension(
        geo_eval.get("outcome", {}),
        [
            "visit_intent_capture",
            "conversion_readiness",
            "brand_search_lift_potential",
            "measurement_maturity",
        ],
    )
    dims = {
        "visibility": visibility,
        "inclusion": inclusion,
        "cognition": cognition,
        "outcome": outcome,
    }

    total_weight = sum(float(v) for v in weights.values()) or 1.0
    overall = 0.0
    for key, val in dims.items():
        overall += val["score"] * (float(weights.get(key, 0)) / total_weight)
    overall = round(overall, 2)

    def level(score: float) -> str:
        if score >= healthy:
            return "healthy"
        if score >= warning:
            return "watch"
        return "risk"

    return {
        "generated_at": utc_now(),
        "brand": {
            "name": brand_cfg.get("brand", {}).get("name", benchmark.get("brand_profile", {}).get("brand_name", "Unknown Brand")),
            "website": brand_cfg.get("brand", {}).get("website", benchmark.get("brand_profile", {}).get("official_website", "")),
            "market": brand_cfg.get("brand", {}).get("market", benchmark.get("brand_profile", {}).get("market", "")),
            "category": brand_cfg.get("brand", {}).get("category", benchmark.get("brand_profile", {}).get("inferred_category", "")),
        },
        "weights": weights,
        "thresholds": {"healthy": healthy, "warning": warning},
        "overall_score": overall,
        "overall_level": level(overall),
        "methodology_note": geo_eval.get("methodology_note", "DeepSeek-assisted initial benchmark; requires later validation with logs, Search Console, and query sampling."),
        "brand_profile": benchmark.get("brand_profile", {}),
        "competitors": benchmark.get("competitors", []),
        "query_panel": benchmark.get("query_panel", []),
        "dimensions": {
            key: {
                "score": val["score"],
                "level": level(val["score"]),
                "metrics": val["metrics"],
                "rationale": val["rationale"],
                "confidence": val["confidence"],
                "priority_actions": val["priority_actions"],
            }
            for key, val in dims.items()
        },
        "strengths": [str(x) for x in geo_eval.get("strengths", [])][:6],
        "risks": [str(x) for x in geo_eval.get("risks", [])][:6],
        "executive_summary": str(geo_eval.get("executive_summary", "")).strip(),
        "limitations": benchmark.get("brand_profile", {}).get("uncertainties", []),
    }


def render_markdown(report: Dict[str, Any]) -> str:
    brand = report["brand"]
    lines: List[str] = []
    lines.append(f"# {brand['name']} GEO Evaluation Report")
    lines.append("")
    lines.append(f"- Generated at: {report['generated_at']}")
    lines.append(f"- Website: {brand.get('website', '') or 'N/A'}")
    lines.append(f"- Market: {brand.get('market', '') or 'N/A'}")
    lines.append(f"- Category: {brand.get('category', '') or 'N/A'}")
    lines.append(f"- Overall GEO Score: **{report['overall_score']} / 100** ({report['overall_level']})")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(report.get("executive_summary") or "No summary generated.")
    lines.append("")
    lines.append("## Scorecard")
    lines.append("")
    lines.append("| Dimension | Score | Level | Confidence |")
    lines.append("|---|---:|---|---:|")
    for key in ["visibility", "inclusion", "cognition", "outcome"]:
        item = report["dimensions"][key]
        lines.append(f"| {key.title()} | {item['score']} | {item['level']} | {round(item['confidence'] * 100, 1)}% |")
    lines.append("")

    lines.append("## Competitor Set")
    lines.append("")
    for comp in report.get("competitors", []):
        lines.append(f"- **{comp.get('name', 'Unknown')}**: {comp.get('why_in_set', '')} (confidence {round(float(comp.get('confidence', 0)) * 100, 1)}%)")
    lines.append("")

    lines.append("## Query Panel")
    lines.append("")
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in report.get("query_panel", []):
        grouped.setdefault(str(item.get("type", "other")), []).append(item)
    for query_type in ["brand", "category", "problem", "comparison", "use_case", "trust"]:
        if query_type not in grouped:
            continue
        lines.append(f"### {query_type}")
        lines.append("")
        for item in grouped[query_type][:8]:
            lines.append(f"- {item.get('query', '')}")
        lines.append("")

    for key in ["visibility", "inclusion", "cognition", "outcome"]:
        item = report["dimensions"][key]
        lines.append(f"## {key.title()}")
        lines.append("")
        lines.append(f"Score: **{item['score']} / 100** ({item['level']})")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for metric_name, value in item["metrics"].items():
            lines.append(f"| {metric_name} | {value} |")
        lines.append("")
        lines.append(item.get("rationale", ""))
        lines.append("")
        if item.get("priority_actions"):
            lines.append("Priority actions:")
            for action in item["priority_actions"]:
                lines.append(f"- {action}")
            lines.append("")

    lines.append("## Strengths")
    lines.append("")
    for bullet in report.get("strengths", []):
        lines.append(f"- {bullet}")
    lines.append("")

    lines.append("## Risks")
    lines.append("")
    for bullet in report.get("risks", []):
        lines.append(f"- {bullet}")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(report.get("methodology_note", ""))
    for bullet in report.get("limitations", []):
        lines.append(f"- {bullet}")
    lines.append("")
    return "\n".join(lines)


def write_outputs(output_dir: Path, report: Dict[str, Any], benchmark: Dict[str, Any]) -> None:
    ensure_dir(output_dir)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (output_dir / "benchmark.generated.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "brand": report["brand"],
        "overall_score": report["overall_score"],
        "overall_level": report["overall_level"],
        "generated_at": report["generated_at"],
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
