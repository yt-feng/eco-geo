#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests
import yaml
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

DIMS = ["visibility", "inclusion", "cognition", "outcome"]
TERM_BUCKETS = ["brand_terms", "competitor_terms", "industry_terms", "category_terms", "problem_terms", "trust_terms"]
EVIDENCE_KEYS = [
    "owned_surface_strength", "entity_clarity", "content_modularity", "trust_signal_density",
    "comparison_page_readiness", "faq_readiness", "documentation_readiness", "pricing_transparency",
    "schema_readiness", "narrative_control",
]
PRESSURE_KEYS = ["peer_activation_index", "benchmark_percentile", "urgency_score", "gap_to_leading_peer", "narrative_disadvantage"]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def esc(value: Any) -> str:
    return html.escape(str(value))


def clamp(value: Any, low: float = 0.0, high: float = 100.0) -> float:
    try:
        number = float(value)
    except Exception:
        number = 0.0
    return max(low, min(high, number))


def avg(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 2) if values else 0.0


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n|\n```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in DeepSeek response")
    return json.loads(match.group(0))


class DeepSeekClient:
    def __init__(self, model: str):
        self.model = model
        self.key = os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1/chat/completions"
        self.runs: List[Dict[str, Any]] = []
        if not self.key:
            raise RuntimeError("DEEPSEEK_API_KEY is required")

    def ask(self, stage: str, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a rigorous GEO research analyst. Return valid JSON only. Be structured, conservative, and consulting-grade."},
                {"role": "user", "content": prompt},
            ],
        }
        response = requests.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            json=payload,
            timeout=240,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        self.runs.append({
            "stage": stage,
            "model": self.model,
            "timestamp": now_utc(),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "response_chars": len(content),
        })
        return extract_json(content)


def brand_from_inputs(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    brand = dict(config.get("brand", {}))
    if args.brand_name:
        brand["name"] = args.brand_name.strip()
    if args.brand_brief:
        brand["brief"] = args.brand_brief.strip()
    if args.website:
        brand["website"] = args.website.strip()
    if not brand.get("name"):
        raise RuntimeError("brand_name is required")
    if not brand.get("brief"):
        brand["brief"] = brand.get("category") or "Minimal brand brief provided; infer cautiously."
    brand.setdefault("market", "Global")
    brand.setdefault("language", "zh-CN")
    return brand


def schema_profile() -> Dict[str, Any]:
    return {
        "brand_name": "", "one_line_brief": "", "official_website": "", "category": "", "market": "", "language": "",
        "positioning_summary": "", "icp": [""], "buying_committee": [""], "core_offerings": [""],
        "must_own_topics": [""], "competitor_candidates": [{"name": "", "reason": "", "confidence": 0}],
        "uncertainties": [""], "confidence": 0,
    }


def schema_keywords() -> Dict[str, Any]:
    return {
        "term_buckets": {bucket: [""] for bucket in TERM_BUCKETS},
        "topic_cloud": [{"term": "", "weight": 0, "bucket": "category_terms"}],
        "semantic_clusters": [{"cluster": "", "terms": [""], "geo_value": 0, "competitive_intensity": 0}],
        "taxonomy_summary": "",
    }


def schema_questions() -> Dict[str, Any]:
    return {
        "query_panel": [{"type": "brand", "query": "", "intent": "", "funnel_stage": "Consider", "importance": 0, "risk_level": "medium"}],
        "question_families": [{"family": "", "purpose": "", "query_count": 0, "representative_queries": [""], "recommended_engine_runs": 0}],
        "monitoring_design": {"planned_query_count": 0, "planned_deepseek_runs": 0, "sampling_notes": ""},
    }


def schema_competitors() -> Dict[str, Any]:
    return {
        "competitors": [{
            "name": "", "why_in_set": "", "geo_maturity_stage": "Active", "overall_score_estimate": 0,
            "dimension_scores": {dim: 0 for dim in DIMS}, "visible_strengths": [""],
            "geo_moves_likely_underway": [""], "evidence_signals": [""], "confidence": 0,
        }],
        "market_pressure": {key: 0 for key in PRESSURE_KEYS},
        "competitive_narrative": "",
    }


def schema_final() -> Dict[str, Any]:
    return {
        "geo_evaluation": {dim: {"score": 0, "metrics": {}, "rationale": "", "confidence": 0, "priority_actions": [""]} for dim in DIMS},
        "evidence_map": {key: 0 for key in EVIDENCE_KEYS},
        "journey_gap_matrix": [{"stage": "Discover", "current_strength": 0, "competitor_pressure": 0, "opportunity": 0, "notes": ""}],
        "executive_summary": "", "strengths": [""], "risks": [""], "methodology_note": "",
    }


def prompt_profile(brand: Dict[str, Any]) -> str:
    return "\n".join([
        "Stage 1: infer a brand GEO profile from minimal input. Output JSON only.",
        f"Brand name: {brand.get('name', '')}",
        f"One-line brief: {brand.get('brief', '')}",
        f"Optional website: {brand.get('website', '')}",
        f"Market: {brand.get('market', 'Global')}",
        "Infer website/category/ICP/competitors cautiously. Do not pretend to browse the live web.",
        json.dumps(schema_profile(), ensure_ascii=False, indent=2),
    ])


def prompt_keywords(profile: Dict[str, Any]) -> str:
    return "\n".join([
        "Stage 2: build a rich GEO keyword taxonomy. Output JSON only.",
        "Create 15-30 terms for important buckets. Terms should be useful for GEO query design and competitive analysis.",
        json.dumps(profile, ensure_ascii=False, indent=2),
        json.dumps(schema_keywords(), ensure_ascii=False, indent=2),
    ])


def prompt_questions(profile: Dict[str, Any], keywords: Dict[str, Any]) -> str:
    return "\n".join([
        "Stage 3: design a private GEO question universe. Output JSON only.",
        "Generate at least 72 queries across brand, competitor, industry, category, problem, comparison, use_case, and trust.",
        "Exact queries are internal and should not be shown in customer dashboard, but must be present in internal audit files.",
        json.dumps({"profile": profile, "keywords": keywords}, ensure_ascii=False, indent=2),
        json.dumps(schema_questions(), ensure_ascii=False, indent=2),
    ])


def prompt_competitors(profile: Dict[str, Any], keywords: Dict[str, Any], questions: Dict[str, Any]) -> str:
    return "\n".join([
        "Stage 4: create a competitive GEO benchmark. Output JSON only.",
        "Show where peers appear more GEO-active and why the target brand has urgency to move. Be credible, not sensational.",
        json.dumps({
            "profile": profile,
            "semantic_clusters": keywords.get("semantic_clusters", []),
            "question_families": questions.get("question_families", []),
        }, ensure_ascii=False, indent=2),
        json.dumps(schema_competitors(), ensure_ascii=False, indent=2),
    ])


def prompt_final(profile: Dict[str, Any], keywords: Dict[str, Any], questions: Dict[str, Any], competitors: Dict[str, Any]) -> str:
    return "\n".join([
        "Stage 5: final GEO scorecard and dashboard narrative. Output JSON only.",
        "Use previous layers. Include at least 4 metrics per dimension. Make it data-driven, competitive, and dashboard-ready.",
        json.dumps({
            "profile": profile,
            "keywords_summary": keywords.get("taxonomy_summary", ""),
            "topic_cloud": keywords.get("topic_cloud", [])[:40],
            "monitoring_design": questions.get("monitoring_design", {}),
            "competitor_benchmark": competitors,
        }, ensure_ascii=False, indent=2),
        json.dumps(schema_final(), ensure_ascii=False, indent=2),
    ])


def run_research(brand: Dict[str, Any], model: str) -> Dict[str, Any]:
    client = DeepSeekClient(model)
    profile = client.ask("01_profile_deep_dive", prompt_profile(brand))
    keywords = client.ask("02_keyword_taxonomy", prompt_keywords(profile))
    questions = client.ask("03_private_question_universe", prompt_questions(profile, keywords))
    competitors = client.ask("04_competitive_benchmark", prompt_competitors(profile, keywords, questions))
    final = client.ask("05_final_scorecard", prompt_final(profile, keywords, questions, competitors))
    return {
        "generated_at": now_utc(), "brand_input": brand, "profile": profile, "keywords": keywords,
        "questions": questions, "competitors": competitors, "final": final, "deepseek_runs": client.runs,
    }


def metric_bar(label: str, value: Any, tone: str = "default") -> str:
    color = {"default": "#60a5fa", "good": "#34d399", "warn": "#f59e0b", "bad": "#fb7185"}.get(tone, "#60a5fa")
    score = clamp(value)
    return f"<div class='bar'><div><span>{esc(label)}</span><b>{round(score, 1)}</b></div><p><i style='width:{score}%;background:{color}'></i></p></div>"


def render_topic_cloud(items: List[Dict[str, Any]]) -> str:
    if not isinstance(items, list) or not items:
        return "<p class='muted'>No topic cloud generated.</p>"
    weights = [clamp(item.get("weight", item.get("count", 1)), 1, 100) for item in items[:50] if isinstance(item, dict)]
    max_weight = max(weights) if weights else 1
    spans = []
    for item in items[:50]:
        if not isinstance(item, dict):
            continue
        weight = clamp(item.get("weight", item.get("count", 1)), 1, 100)
        size = 12 + int(30 * weight / max_weight)
        spans.append(f"<span style='font-size:{size}px'>{esc(item.get('term', ''))}</span>")
    return "<div class='cloud'>" + "".join(spans) + "</div>"


def render_query_mix(query_panel: List[Dict[str, Any]]) -> str:
    counts = Counter(str(q.get("type", "other")) for q in query_panel if isinstance(q, dict))
    if not counts:
        return "<p class='muted'>No query mix generated.</p>"
    pairs = list(counts.items())[:14]
    max_count = max(v for _, v in pairs) or 1
    rows = []
    height = 34 + len(pairs) * 26
    for i, (label, value) in enumerate(pairs):
        y = 24 + i * 26
        width = 500 * value / max_count
        rows.append(
            f"<text x='8' y='{y+13}' fill='#b9c7dd' font-size='12'>{esc(label)}</text>"
            f"<rect x='155' y='{y}' width='{width:.1f}' height='16' rx='7' fill='url(#g)'/>"
            f"<text x='{162+width:.1f}' y='{y+13}' fill='#e8eef9' font-size='12'>{value}</text>"
        )
    return f"<svg viewBox='0 0 720 {height}' class='chart'><defs><linearGradient id='g' x1='0%' x2='100%'><stop offset='0%' stop-color='#38bdf8'/><stop offset='100%' stop-color='#818cf8'/></linearGradient></defs>{''.join(rows)}</svg>"


def render_dashboard(research: Dict[str, Any], output_dir: Path) -> None:
    brand = research["brand_input"]
    keywords = research.get("keywords", {})
    questions = research.get("questions", {})
    competitors = research.get("competitors", {})
    final = research.get("final", {})
    geo = final.get("geo_evaluation", {}) if isinstance(final.get("geo_evaluation", {}), dict) else {}
    pressure = competitors.get("market_pressure", {}) if isinstance(competitors.get("market_pressure", {}), dict) else {}
    query_panel = questions.get("query_panel", []) if isinstance(questions.get("query_panel", []), list) else []
    competitor_list = competitors.get("competitors", []) if isinstance(competitors.get("competitors", []), list) else []
    question_families = questions.get("question_families", []) if isinstance(questions.get("question_families", []), list) else []

    comp_rows = ""
    for comp in competitor_list[:10]:
        if not isinstance(comp, dict):
            continue
        ds = comp.get("dimension_scores", {}) if isinstance(comp.get("dimension_scores", {}), dict) else {}
        comp_rows += (
            "<tr>"
            f"<td><strong>{esc(comp.get('name', ''))}</strong></td>"
            f"<td>{esc(comp.get('geo_maturity_stage', ''))}</td>"
            f"<td>{esc(comp.get('overall_score_estimate', ''))}</td>"
            f"<td>{esc(ds.get('visibility', ''))}</td><td>{esc(ds.get('inclusion', ''))}</td><td>{esc(ds.get('cognition', ''))}</td><td>{esc(ds.get('outcome', ''))}</td>"
            f"<td>{esc('; '.join([str(x) for x in comp.get('evidence_signals', [])[:2]]))}</td>"
            "</tr>"
        )

    dim_cards = ""
    for dim in DIMS:
        item = geo.get(dim, {}) if isinstance(geo.get(dim, {}), dict) else {}
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
        actions = item.get("priority_actions", []) if isinstance(item.get("priority_actions", []), list) else []
        dim_cards += (
            f"<section class='card'><h3>{dim.title()}</h3>"
            f"{metric_bar('Score', item.get('score', 0))}"
            + "".join(metric_bar(k.replace("_", " ").title(), v) for k, v in list(metrics.items())[:6])
            + f"<p class='muted'>{esc(item.get('rationale', ''))}</p>"
            + "<ul>" + "".join(f"<li>{esc(a)}</li>" for a in actions[:4]) + "</ul></section>"
        )

    html_text = f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{esc(brand.get('name',''))} GEO v3 Research Dashboard</title>
<style>
body{{margin:0;background:#07111f;color:#e7eefb;font-family:Inter,Segoe UI,Arial,sans-serif}}.wrap{{max-width:1500px;margin:auto;padding:24px}}
.hero,.grid2,.grid4{{display:grid;gap:18px}}.hero{{grid-template-columns:2fr 1fr}}.grid2{{grid-template-columns:1.15fr .85fr}}.grid4{{grid-template-columns:repeat(4,1fr)}}
.card,.kpi{{background:#0f1b2d;border:1px solid #22324a;border-radius:18px;padding:18px;margin-bottom:18px;box-shadow:0 12px 28px rgba(0,0,0,.18)}}
.kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:16px}}.kpi h2{{font-size:30px;margin:6px 0}}.muted,small{{color:#93a4bb;line-height:1.55}}
.bar div{{display:flex;justify-content:space-between;font-size:13px}}.bar p{{height:10px;background:#0b1424;border:1px solid #24344d;border-radius:999px;overflow:hidden}}.bar i{{display:block;height:100%}}
table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{border-bottom:1px solid #22324a;padding:10px;text-align:left;vertical-align:top}}th{{color:#b9c7dd}}
.cloud{{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end}}.cloud span{{background:#111f34;border:1px solid #22324a;border-radius:999px;padding:6px 10px}}.chart{{width:100%;height:auto}}
@media(max-width:1100px){{.hero,.grid2,.grid4,.kpis{{grid-template-columns:1fr}}}}
</style></head><body><div class='wrap'>
<section class='hero'><div class='card'><small>GEO v3 Research Dashboard</small><h1>{esc(brand.get('name',''))}</h1><p class='muted'>{esc(brand.get('brief',''))}</p><p>{esc(final.get('executive_summary',''))}</p>
<div class='kpis'><div class='kpi'><small>Actual DeepSeek calls</small><h2>{len(research.get('deepseek_runs', []))}</h2></div><div class='kpi'><small>Private query universe</small><h2>{len(query_panel)}</h2></div><div class='kpi'><small>Question families</small><h2>{len(question_families)}</h2></div><div class='kpi'><small>Competitors benchmarked</small><h2>{len(competitor_list)}</h2></div><div class='kpi'><small>Topic clusters</small><h2>{len(keywords.get('semantic_clusters', []))}</h2></div></div></div>
<div class='card'><h3>Market Pressure</h3>{metric_bar('Peer activation index', pressure.get('peer_activation_index', 0), 'warn')}{metric_bar('Urgency score', pressure.get('urgency_score', 0), 'bad')}{metric_bar('Gap to leading peer', pressure.get('gap_to_leading_peer', 0), 'bad')}{metric_bar('Narrative disadvantage', pressure.get('narrative_disadvantage', 0), 'bad')}<p class='muted'>{esc(competitors.get('competitive_narrative', ''))}</p></div></section>
<section class='grid2'><div class='card'><h3>Question Universe Mix</h3>{render_query_mix(query_panel)}</div><div class='card'><h3>Research Topic Cloud</h3>{render_topic_cloud(keywords.get('topic_cloud', []))}</div></section>
<section class='card'><h3>Competitor GEO Leaderboard</h3><table><tr><th>Competitor</th><th>Stage</th><th>Overall</th><th>Visibility</th><th>Inclusion</th><th>Cognition</th><th>Outcome</th><th>Evidence Signal</th></tr>{comp_rows}</table></section>
<section class='grid4'>{dim_cards}</section>
</div></body></html>"""
    (output_dir / "dashboard.html").write_text(html_text, encoding="utf-8")


def render_internal_audit(research: Dict[str, Any], output_dir: Path) -> None:
    brand = research["brand_input"]
    keywords = research.get("keywords", {})
    questions = research.get("questions", {})
    term_buckets = keywords.get("term_buckets", {}) if isinstance(keywords.get("term_buckets", {}), dict) else {}
    families = questions.get("question_families", []) if isinstance(questions.get("question_families", []), list) else []

    cards = "".join(
        f"<section class='card'><h3>{esc(k.replace('_',' ').title())}</h3><p>{esc(', '.join([str(x) for x in v]))}</p></section>"
        for k, v in term_buckets.items() if isinstance(v, list)
    )
    family_rows = ""
    for family in families[:30]:
        if not isinstance(family, dict):
            continue
        family_rows += (
            f"<tr><td>{esc(family.get('family',''))}</td><td>{esc(family.get('purpose',''))}</td>"
            f"<td>{esc(family.get('query_count',''))}</td><td>{esc(family.get('recommended_engine_runs',''))}</td>"
            f"<td>{esc('; '.join([str(x) for x in family.get('representative_queries', [])[:5]]))}</td></tr>"
        )
    run_rows = "".join(
        f"<tr><td>{esc(run.get('stage',''))}</td><td>{esc(run.get('model',''))}</td><td>{esc(run.get('total_tokens',''))}</td><td>{esc(run.get('response_chars',''))}</td></tr>"
        for run in research.get("deepseek_runs", [])
    )
    html_text = f"""<html><head><meta charset='utf-8'><style>body{{background:#07111f;color:#e7eefb;font-family:Inter,Segoe UI,Arial,sans-serif;padding:24px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{background:#0f1b2d;border:1px solid #22324a;border-radius:16px;padding:16px;margin-bottom:16px}}table{{width:100%;border-collapse:collapse}}td,th{{border-bottom:1px solid #22324a;padding:9px;text-align:left;vertical-align:top}}@media(max-width:1100px){{.grid{{grid-template-columns:1fr}}}}</style></head><body><h1>{esc(brand.get('name',''))} Internal GEO Research Audit</h1><p>Actual DeepSeek pipeline calls: <b>{len(research.get('deepseek_runs', []))}</b></p><div class='grid'>{cards}</div><section class='card'><h2>Question Families and Representative Queries</h2><table><tr><th>Family</th><th>Purpose</th><th>Queries</th><th>Runs</th><th>Representative Queries</th></tr>{family_rows}</table></section><section class='card'><h2>DeepSeek Pipeline Calls</h2><table><tr><th>Stage</th><th>Model</th><th>Total Tokens</th><th>Response Chars</th></tr>{run_rows}</table></section></body></html>"""
    (output_dir / "internal_audit.html").write_text(html_text, encoding="utf-8")
    (output_dir / "internal_audit.json").write_text(json.dumps(research, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "internal_audit.md").write_text(json.dumps({
        "term_buckets": term_buckets,
        "question_families": families,
        "deepseek_runs": research.get("deepseek_runs", []),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def build_pdf(path: Path, title: str, research: Dict[str, Any], internal: bool = False) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    story: List[Any] = [Paragraph(esc(title), styles["Title"]), Spacer(1, 8)]
    if internal:
        rows = [["Stage", "Model", "Tokens", "Chars"]] + [
            [run.get("stage", ""), run.get("model", ""), run.get("total_tokens", ""), run.get("response_chars", "")]
            for run in research.get("deepseek_runs", [])
        ]
        story.append(Paragraph("DeepSeek Pipeline Calls", styles["Heading2"]))
        story.append(Table(rows, repeatRows=1))
        term_buckets = research.get("keywords", {}).get("term_buckets", {})
        for key, values in term_buckets.items():
            if isinstance(values, list):
                story.append(Spacer(1, 6))
                story.append(Paragraph(key.replace("_", " ").title(), styles["Heading2"]))
                story.append(Paragraph(esc(", ".join([str(x) for x in values[:30]])), styles["Small"]))
    else:
        story.append(Paragraph("Executive Summary", styles["Heading2"]))
        story.append(Paragraph(esc(research.get("final", {}).get("executive_summary", "")), styles["Small"]))
        rows = [["Metric", "Value"], ["DeepSeek calls", len(research.get("deepseek_runs", []))], ["Private queries", len(research.get("questions", {}).get("query_panel", []))], ["Question families", len(research.get("questions", {}).get("question_families", []))], ["Competitors", len(research.get("competitors", {}).get("competitors", []))]]
        story.append(Spacer(1, 8))
        story.append(Table(rows, repeatRows=1))
    for item in story:
        if isinstance(item, Table):
            item.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e6fb")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=12*mm, rightMargin=12*mm, topMargin=12*mm, bottomMargin=12*mm).build(story)


def commit_report(repo_root: Path, output_dir: Path, report_subdir: str, message: str) -> Dict[str, Any]:
    target = repo_root / report_subdir
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output_dir, target)
    subprocess.run(["git", "add", str(target)], cwd=repo_root, check=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, check=True, capture_output=True, text=True)
    if not status.stdout.strip():
        return {"committed": False, "target": str(target)}
    subprocess.run(["git", "commit", "-m", message], cwd=repo_root, check=True)
    return {"committed": True, "target": str(target)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v3 minimal-input GEO research pipeline")
    parser.add_argument("--brand-config", default="config/brand.yaml")
    parser.add_argument("--brand-name", default="")
    parser.add_argument("--brand-brief", default="")
    parser.add_argument("--website", default="")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--output", default="dist/report")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--report-subdir", default="reports/latest")
    parser.add_argument("--commit-message", default="chore: update GEO v3 research dashboard")
    parser.add_argument("--commit-report", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    brand = brand_from_inputs(args, load_yaml(repo_root / args.brand_config))

    research = run_research(brand, args.model)
    (output_dir / "research_layers.json").write_text(json.dumps(research, ensure_ascii=False, indent=2), encoding="utf-8")
    render_dashboard(research, output_dir)
    render_internal_audit(research, output_dir)
    build_pdf(output_dir / "dashboard.pdf", f"{brand.get('name')} GEO v3 Research Dashboard", research, internal=False)
    build_pdf(output_dir / "internal_audit.pdf", f"{brand.get('name')} Internal GEO Audit", research, internal=True)

    summary = {
        "brand": brand,
        "generated_at": research.get("generated_at"),
        "actual_deepseek_pipeline_calls": len(research.get("deepseek_runs", [])),
        "private_query_count": len(research.get("questions", {}).get("query_panel", [])),
        "question_family_count": len(research.get("questions", {}).get("question_families", [])),
        "competitor_count": len(research.get("competitors", {}).get("competitors", [])),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text("# GEO v3 Research Dashboard\n\n" + json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    commit_info = {"committed": False}
    if args.commit_report:
        commit_info = commit_report(repo_root, output_dir, args.report_subdir, args.commit_message)
    print(json.dumps({"brand": brand.get("name"), "deepseek_calls": summary["actual_deepseek_pipeline_calls"], "private_queries": summary["private_query_count"], "commit": commit_info}, ensure_ascii=False))


if __name__ == "__main__":
    main()
