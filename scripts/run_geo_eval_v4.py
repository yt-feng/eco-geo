#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

DIMS = ["visibility", "inclusion", "cognition", "outcome"]
QUERY_TYPES = ["brand", "competitor", "industry", "category", "problem", "comparison", "use_case", "trust"]
FUNNELS = ["Discover", "Consider", "Validate", "Select", "Expand"]


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
        return {"_raw_text": text}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {"_raw_text": text}


class DeepSeekMonitor:
    def __init__(self, model: str):
        self.model = model
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1/chat/completions"
        self.calls: List[Dict[str, Any]] = []
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required")

    def call_json(self, stage: str, prompt: str, *, question: str = "", query_id: str = "", call_type: str = "research", max_retries: int = 2) -> Dict[str, Any]:
        call_id = f"call_{len(self.calls) + 1:04d}"
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a rigorous GEO monitoring analyst. Return valid JSON only. Be structured, conservative, and explicit."},
                {"role": "user", "content": prompt},
            ],
        }
        started = time.time()
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=180,
                )
                response.raise_for_status()
                data = response.json()
                response_text = data["choices"][0]["message"]["content"]
                parsed = extract_json(response_text)
                usage = data.get("usage", {})
                record = {
                    "call_id": call_id,
                    "stage": stage,
                    "call_type": call_type,
                    "query_id": query_id,
                    "question": question,
                    "model": self.model,
                    "timestamp": now_utc(),
                    "duration_ms": int((time.time() - started) * 1000),
                    "success": True,
                    "attempts": attempt + 1,
                    "prompt": prompt,
                    "response_text": response_text,
                    "response_json": parsed,
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    },
                }
                self.calls.append(record)
                return parsed
            except Exception as exc:
                last_error = str(exc)
                time.sleep(min(2 + attempt, 5))
        record = {
            "call_id": call_id,
            "stage": stage,
            "call_type": call_type,
            "query_id": query_id,
            "question": question,
            "model": self.model,
            "timestamp": now_utc(),
            "duration_ms": int((time.time() - started) * 1000),
            "success": False,
            "attempts": max_retries + 1,
            "prompt": prompt,
            "response_text": "",
            "response_json": {},
            "error": last_error,
            "usage": {},
        }
        self.calls.append(record)
        return {"error": last_error, "question": question}


def setup_prompt(brand_name: str, brand_brief: str, website: str, monitor_runs: int) -> str:
    schema = {
        "brand_profile": {
            "brand_name": "", "brief": "", "official_website": "", "category": "", "market": "", "positioning_summary": "",
            "ideal_customer_profiles": [""], "buying_committee": [""], "core_offerings": [""], "must_own_topics": [""], "uncertainties": [""],
        },
        "term_buckets": {
            "brand_terms": [""], "competitor_terms": [""], "industry_terms": [""], "category_terms": [""], "problem_terms": [""], "trust_terms": [""],
        },
        "topic_cloud": [{"term": "", "weight": 0, "bucket": "category_terms"}],
        "competitors": [{"name": "", "why_in_set": "", "expected_geo_strength": 0, "likely_advantages": [""], "confidence": 0}],
        "semantic_clusters": [{"cluster": "", "terms": [""], "geo_value": 0, "competitive_intensity": 0}],
        "query_plan": [{"query_id": "q001", "type": "brand", "funnel_stage": "Consider", "question": "", "intent": "", "importance": 0, "risk_level": "medium"}],
        "monitoring_method": {"planned_deepseek_calls": 0, "sampling_logic": "", "evaluation_dimensions": [""]},
    }
    return "\n".join([
        "Build a complete GEO monitoring plan from minimal brand input. Output valid JSON only.",
        f"Target brand: {brand_name}",
        f"One-line brand brief: {brand_brief}",
        f"Optional official website: {website}",
        f"Required number of concrete monitoring questions: {monitor_runs}",
        "Generate exactly the requested number of query_plan items. Each item must contain a concrete question that can be sent to DeepSeek as a GEO probe.",
        "Cover brand, competitor, industry, category, problem, comparison, use_case, and trust questions. Make the plan feel operationally serious and suitable for a client-facing audit.",
        "Do not claim to browse the live web. Infer cautiously from general knowledge and the brand brief.",
        json.dumps(schema, ensure_ascii=False, indent=2),
    ])


def monitoring_prompt(brand_name: str, brand_brief: str, question_item: Dict[str, Any], competitors: List[str]) -> str:
    schema = {
        "question": "",
        "answer": "",
        "target_brand_mentioned": False,
        "target_brand_role": "not_mentioned | cited | recommended | compared | criticized | incidental",
        "target_brand_sentiment": "positive | neutral | negative | not_mentioned",
        "target_brand_recommendation_strength": 0,
        "competitors_mentioned": [""],
        "recommended_brands": [""],
        "best_answer_owner": "",
        "first_party_source_needed": False,
        "citation_likelihood_score": 0,
        "answer_confidence": 0,
        "factual_risk_flags": [""],
        "geo_gap_observed": "",
        "summary_takeaway": "",
    }
    return "\n".join([
        "Run one DeepSeek GEO monitoring probe. Output valid JSON only.",
        "Answer the user's question like a generative answer engine would, then evaluate whether the target brand is visible, credible, and competitive in that answer.",
        f"Target brand: {brand_name}",
        f"Brand brief: {brand_brief}",
        f"Known competitor set: {json.dumps(competitors, ensure_ascii=False)}",
        f"Query metadata: {json.dumps(question_item, ensure_ascii=False)}",
        f"User question to answer: {question_item.get('question', '')}",
        "Be explicit. If the target brand is absent while competitors appear, mark the GEO gap clearly.",
        json.dumps(schema, ensure_ascii=False, indent=2),
    ])


def synthesis_prompt(brand_name: str, setup: Dict[str, Any], aggregate: Dict[str, Any], sample_results: List[Dict[str, Any]]) -> str:
    schema = {
        "executive_summary": "",
        "dimension_scores": {"visibility": 0, "inclusion": 0, "cognition": 0, "outcome": 0},
        "evidence_map": {"answer_visibility": 0, "competitor_pressure": 0, "first_party_citation_need": 0, "narrative_control": 0, "trust_signal_gap": 0, "recommendation_strength": 0},
        "top_findings": [""],
        "priority_actions": [""],
        "client_message": "",
        "methodology_note": "",
    }
    return "\n".join([
        "Synthesize the completed GEO monitoring run into a dashboard-ready executive assessment. Output valid JSON only.",
        f"Target brand: {brand_name}",
        "Setup plan summary:",
        json.dumps({"profile": setup.get("brand_profile", {}), "competitors": setup.get("competitors", [])[:10], "monitoring_method": setup.get("monitoring_method", {})}, ensure_ascii=False, indent=2),
        "Aggregate monitoring metrics:",
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        "Sample probe results:",
        json.dumps(sample_results[:12], ensure_ascii=False, indent=2),
        "Make clear that GEO monitoring is a systematic measurement program, not a one-off prompt test.",
        json.dumps(schema, ensure_ascii=False, indent=2),
    ])


def normalize_plan(setup: Dict[str, Any], requested_runs: int) -> List[Dict[str, Any]]:
    plan = setup.get("query_plan", []) if isinstance(setup.get("query_plan", []), list) else []
    normalized: List[Dict[str, Any]] = []
    for i, item in enumerate(plan[:requested_runs], start=1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        normalized.append({
            "query_id": str(item.get("query_id") or f"q{i:03d}"),
            "type": str(item.get("type") or "other"),
            "funnel_stage": str(item.get("funnel_stage") or "Consider"),
            "question": question,
            "intent": str(item.get("intent") or ""),
            "importance": clamp(item.get("importance", 50)),
            "risk_level": str(item.get("risk_level") or "medium"),
        })
    return normalized


def compute_aggregate(brand_name: str, query_plan: List[Dict[str, Any]], probe_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = [call.get("response_json", {}) for call in probe_calls if call.get("success")]
    total = len(results)
    mentioned = [r for r in results if r.get("target_brand_mentioned") is True]
    recommended = [r for r in results if str(r.get("target_brand_role", "")).lower() == "recommended" or clamp(r.get("target_brand_recommendation_strength", 0)) >= 70]
    first_party_needed = [r for r in results if r.get("first_party_source_needed") is True]
    risk_flags = []
    for r in results:
        flags = r.get("factual_risk_flags", [])
        if isinstance(flags, list):
            risk_flags.extend([f for f in flags if f])
    competitors = Counter()
    recommended_brands = Counter()
    sentiment = Counter()
    answer_owner = Counter()
    for r in results:
        sentiment[str(r.get("target_brand_sentiment", "unknown"))] += 1
        answer_owner[str(r.get("best_answer_owner", "unknown"))] += 1
        for c in r.get("competitors_mentioned", []) if isinstance(r.get("competitors_mentioned", []), list) else []:
            if c:
                competitors[str(c)] += 1
        for c in r.get("recommended_brands", []) if isinstance(r.get("recommended_brands", []), list) else []:
            if c:
                recommended_brands[str(c)] += 1
    query_type_counts = Counter(q.get("type", "other") for q in query_plan)
    funnel_counts = Counter(q.get("funnel_stage", "unknown") for q in query_plan)
    visibility_rate = round(100 * len(mentioned) / total, 2) if total else 0.0
    recommendation_rate = round(100 * len(recommended) / total, 2) if total else 0.0
    avg_citation = avg([clamp(r.get("citation_likelihood_score", 0)) for r in results])
    avg_confidence = avg([clamp(r.get("answer_confidence", 0)) for r in results])
    dimension_scores = {
        "visibility": round(visibility_rate * 0.55 + recommendation_rate * 0.25 + min(20, avg_citation * 0.2), 2),
        "inclusion": round(avg_citation * 0.55 + (100 - (100 * len(first_party_needed) / total if total else 0)) * 0.25 + avg_confidence * 0.20, 2),
        "cognition": round((100 - min(100, len(risk_flags) * 5)) * 0.45 + (sentiment.get("positive", 0) / total * 100 if total else 0) * 0.35 + avg_confidence * 0.20, 2),
        "outcome": round(recommendation_rate * 0.65 + visibility_rate * 0.20 + avg_citation * 0.15, 2),
    }
    return {
        "brand_name": brand_name,
        "monitoring_questions_planned": len(query_plan),
        "monitoring_calls_successful": total,
        "monitoring_calls_failed": len([c for c in probe_calls if not c.get("success")]),
        "brand_mention_rate": visibility_rate,
        "brand_recommendation_rate": recommendation_rate,
        "first_party_source_needed_rate": round(100 * len(first_party_needed) / total, 2) if total else 0.0,
        "avg_citation_likelihood": avg_citation,
        "avg_answer_confidence": avg_confidence,
        "factual_risk_flag_count": len(risk_flags),
        "query_type_distribution": dict(query_type_counts),
        "funnel_stage_distribution": dict(funnel_counts),
        "competitor_mention_counts": dict(competitors.most_common(20)),
        "recommended_brand_counts": dict(recommended_brands.most_common(20)),
        "sentiment_distribution": dict(sentiment),
        "best_answer_owner_distribution": dict(answer_owner.most_common(20)),
        "dimension_scores": dimension_scores,
        "risk_flag_samples": risk_flags[:30],
    }


def svg_bar(items: Dict[str, Any], title_width: int = 210) -> str:
    pairs = [(str(k), int(clamp(v, 0, 100000))) for k, v in list(items.items())[:18]]
    if not pairs:
        return "<p class='muted'>No data.</p>"
    max_value = max(v for _, v in pairs) or 1
    height = 34 + len(pairs) * 28
    rows = []
    for i, (label, value) in enumerate(pairs):
        y = 24 + i * 28
        width = 520 * value / max_value
        rows.append(f"<text x='8' y='{y+14}' fill='#b9c7dd' font-size='12'>{esc(label[:32])}</text><rect x='{title_width}' y='{y}' width='{width:.1f}' height='17' rx='7' fill='url(#g)'/><text x='{title_width + width + 8:.1f}' y='{y+14}' fill='#e8eef9' font-size='12'>{value}</text>")
    return f"<svg viewBox='0 0 850 {height}' class='chart'><defs><linearGradient id='g' x1='0%' x2='100%'><stop offset='0%' stop-color='#38bdf8'/><stop offset='100%' stop-color='#818cf8'/></linearGradient></defs>{''.join(rows)}</svg>"


def metric_bar(label: str, value: Any, tone: str = "default") -> str:
    color = {"default": "#60a5fa", "good": "#34d399", "warn": "#f59e0b", "bad": "#fb7185"}.get(tone, "#60a5fa")
    score = clamp(value)
    return f"<div class='bar'><div><span>{esc(label)}</span><b>{round(score,1)}</b></div><p><i style='width:{score}%;background:{color}'></i></p></div>"


def term_cloud(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "<p class='muted'>No topic cloud.</p>"
    max_weight = max(clamp(item.get("weight", item.get("count", 1)), 1, 1000) for item in items[:80]) or 1
    tags = []
    for item in items[:80]:
        weight = clamp(item.get("weight", item.get("count", 1)), 1, 1000)
        size = 12 + int(32 * weight / max_weight)
        tags.append(f"<span style='font-size:{size}px'>{esc(item.get('term',''))}</span>")
    return "<div class='cloud'>" + "".join(tags) + "</div>"


def render_dashboard(output_dir: Path, brand_name: str, brand_brief: str, setup: Dict[str, Any], aggregate: Dict[str, Any], synthesis: Dict[str, Any], calls: List[Dict[str, Any]]) -> None:
    profile = setup.get("brand_profile", {})
    term_buckets = setup.get("term_buckets", {}) if isinstance(setup.get("term_buckets", {}), dict) else {}
    topic_cloud = setup.get("topic_cloud", []) if isinstance(setup.get("topic_cloud", []), list) else []
    competitors = setup.get("competitors", []) if isinstance(setup.get("competitors", []), list) else []
    query_plan = setup.get("query_plan", []) if isinstance(setup.get("query_plan", []), list) else []
    monitoring_calls = [c for c in calls if c.get("call_type") == "monitoring"]
    total_tokens = sum(int((c.get("usage") or {}).get("total_tokens") or 0) for c in calls)
    prompt_tokens = sum(int((c.get("usage") or {}).get("prompt_tokens") or 0) for c in calls)
    completion_tokens = sum(int((c.get("usage") or {}).get("completion_tokens") or 0) for c in calls)
    data_inventory = {
        "Total DeepSeek API calls": len(calls),
        "Monitoring probes": len(monitoring_calls),
        "Concrete questions": len(query_plan),
        "Competitors mapped": len(competitors),
        "Term buckets": len(term_buckets),
        "Total keyword terms": sum(len(v) for v in term_buckets.values() if isinstance(v, list)),
        "Prompt tokens": prompt_tokens,
        "Completion tokens": completion_tokens,
        "Total tokens": total_tokens,
        "Failed calls": len([c for c in calls if not c.get("success")]),
    }
    inventory_cards = "".join(f"<div class='kpi'><small>{esc(k)}</small><h2>{esc(v)}</h2></div>" for k, v in data_inventory.items())
    dim_scores = synthesis.get("dimension_scores") or aggregate.get("dimension_scores", {})
    dim_cards = "".join(f"<section class='card'><h3>{dim.title()}</h3>{metric_bar('Score', dim_scores.get(dim, 0))}</section>" for dim in DIMS)
    evidence = synthesis.get("evidence_map", {}) if isinstance(synthesis.get("evidence_map", {}), dict) else {}
    evidence_cards = "".join(f"<div class='mini'><small>{esc(k.replace('_',' ').title())}</small>{metric_bar('', v, 'good' if clamp(v)>=70 else 'warn' if clamp(v)>=45 else 'bad')}</div>" for k, v in evidence.items())
    competitor_rows = "".join(f"<tr><td>{esc(c.get('name',''))}</td><td>{esc(c.get('expected_geo_strength',''))}</td><td>{esc(c.get('confidence',''))}</td><td>{esc(c.get('why_in_set',''))}</td><td>{esc('; '.join([str(x) for x in c.get('likely_advantages', [])[:3]]))}</td></tr>" for c in competitors[:20] if isinstance(c, dict))
    call_rows = []
    for call in calls:
        response_summary = call.get("response_json", {}).get("summary_takeaway") or call.get("response_json", {}).get("executive_summary") or call.get("response_json", {}).get("positioning_summary") or ""
        call_rows.append(f"<tr><td>{esc(call.get('call_id'))}</td><td>{esc(call.get('stage'))}</td><td>{esc(call.get('call_type'))}</td><td>{esc(call.get('query_id'))}</td><td>{esc(call.get('success'))}</td><td>{esc((call.get('usage') or {}).get('total_tokens',''))}</td><td>{esc(call.get('duration_ms',''))}</td><td>{esc(call.get('question',''))}</td><td>{esc(str(response_summary)[:280])}</td></tr>")
    detail_blocks = []
    for call in calls:
        detail_blocks.append(f"<details class='call-detail'><summary>{esc(call.get('call_id'))} · {esc(call.get('stage'))} · {esc(call.get('question') or 'research call')}</summary><h4>Prompt</h4><pre>{esc(call.get('prompt',''))}</pre><h4>Response</h4><pre>{esc(call.get('response_text',''))}</pre></details>")
    term_sections = "".join(f"<section class='mini'><h4>{esc(k.replace('_',' ').title())}</h4><p>{esc(', '.join([str(x) for x in v[:80]]))}</p></section>" for k, v in term_buckets.items() if isinstance(v, list))
    result_cards = []
    for call in monitoring_calls:
        r = call.get("response_json", {})
        result_cards.append(f"<details class='call-detail'><summary>{esc(call.get('query_id'))} · {esc(call.get('question'))}</summary><p><b>Mentioned:</b> {esc(r.get('target_brand_mentioned'))} · <b>Role:</b> {esc(r.get('target_brand_role'))} · <b>Sentiment:</b> {esc(r.get('target_brand_sentiment'))}</p><p><b>Competitors:</b> {esc(', '.join([str(x) for x in r.get('competitors_mentioned', [])]))}</p><p><b>Answer:</b></p><pre>{esc(r.get('answer',''))}</pre><p><b>Takeaway:</b> {esc(r.get('summary_takeaway',''))}</p></details>")
    html_text = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{esc(brand_name)} Full GEO Monitoring Dashboard</title><style>body{{margin:0;background:#07111f;color:#e7eefb;font-family:Inter,Segoe UI,Arial,sans-serif}}.wrap{{max-width:1680px;margin:auto;padding:24px}}.hero,.grid2,.grid3,.grid4,.grid5{{display:grid;gap:18px}}.hero{{grid-template-columns:1.7fr 1fr}}.grid2{{grid-template-columns:1.1fr .9fr}}.grid3{{grid-template-columns:repeat(3,1fr)}}.grid4{{grid-template-columns:repeat(4,1fr)}}.grid5{{grid-template-columns:repeat(5,1fr)}}.card,.kpi,.mini,.call-detail{{background:#0f1b2d;border:1px solid #22324a;border-radius:18px;padding:18px;margin-bottom:18px;box-shadow:0 12px 28px rgba(0,0,0,.18)}}.kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:16px}}.kpi h2{{font-size:28px;margin:6px 0}}.muted,small{{color:#93a4bb;line-height:1.55}}.bar div{{display:flex;justify-content:space-between;font-size:13px}}.bar p{{height:10px;background:#0b1424;border:1px solid #24344d;border-radius:999px;overflow:hidden}}.bar i{{display:block;height:100%}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{border-bottom:1px solid #22324a;padding:9px;text-align:left;vertical-align:top}}th{{color:#b9c7dd}}pre{{white-space:pre-wrap;word-break:break-word;background:#08111f;border:1px solid #22324a;border-radius:12px;padding:12px;max-height:520px;overflow:auto}}.cloud{{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end}}.cloud span{{background:#111f34;border:1px solid #22324a;border-radius:999px;padding:6px 10px}}.chart{{width:100%;height:auto}}summary{{cursor:pointer;font-weight:700}}@media(max-width:1100px){{.hero,.grid2,.grid3,.grid4,.grid5,.kpis{{grid-template-columns:1fr}}}}</style></head><body><div class='wrap'><section class='hero'><div class='card'><small>Full GEO Monitoring System</small><h1>{esc(brand_name)}</h1><p class='muted'>{esc(brand_brief)}</p><p>{esc(synthesis.get('executive_summary',''))}</p><div class='kpis'>{inventory_cards}</div></div><div class='card'><h3>System Workflow</h3>{metric_bar('Brand profile setup', 100, 'good')}{metric_bar('Question universe generation', 100, 'good')}{metric_bar('DeepSeek monitoring probes', min(100, len(monitoring_calls)), 'warn')}{metric_bar('Result aggregation', 100, 'good')}{metric_bar('Dashboard synthesis', 100, 'good')}</div></section><section class='grid2'><div class='card'><h3>Question Universe Distribution</h3>{svg_bar(aggregate.get('query_type_distribution', {}))}</div><div class='card'><h3>Funnel Stage Distribution</h3>{svg_bar(aggregate.get('funnel_stage_distribution', {}))}</div></section><section class='grid2'><div class='card'><h3>Competitor Mention Frequency</h3>{svg_bar(aggregate.get('competitor_mention_counts', {}), 250)}</div><div class='card'><h3>Recommended Brand Frequency</h3>{svg_bar(aggregate.get('recommended_brand_counts', {}), 250)}</div></section><section class='grid4'>{dim_cards}</section><section class='grid2'><div class='card'><h3>Evidence Map</h3><div class='grid3'>{evidence_cards}</div></div><div class='card'><h3>Research Topic Cloud</h3>{term_cloud(topic_cloud)}</div></section><section class='card'><h3>Term Buckets</h3><div class='grid3'>{term_sections}</div></section><section class='card'><h3>Competitor Research Set</h3><table><tr><th>Name</th><th>Expected GEO Strength</th><th>Confidence</th><th>Why in Set</th><th>Likely Advantages</th></tr>{competitor_rows}</table></section><section class='card'><h3>Complete API Call Ledger</h3><p class='muted'>Every DeepSeek call is shown here with stage, question, token usage and response summary. Full prompts and full responses are available below.</p><table><tr><th>Call</th><th>Stage</th><th>Type</th><th>Query ID</th><th>Success</th><th>Tokens</th><th>ms</th><th>Question</th><th>Response Summary</th></tr>{''.join(call_rows)}</table></section><section class='card'><h3>Complete Monitoring Results</h3>{''.join(result_cards)}</section><section class='card'><h3>Full Prompt and Response Trace</h3>{''.join(detail_blocks)}</section></div></body></html>"""
    (output_dir / "dashboard.html").write_text(html_text, encoding="utf-8")


def build_pdf(path: Path, brand_name: str, aggregate: Dict[str, Any], synthesis: Dict[str, Any], calls: List[Dict[str, Any]]) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    story: List[Any] = [Paragraph(f"{esc(brand_name)} Full GEO Monitoring Dashboard", styles["Title"]), Spacer(1, 8)]
    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    story.append(Paragraph(esc(synthesis.get("executive_summary", "")), styles["Small"]))
    rows = [["Metric", "Value"], ["Total DeepSeek calls", len(calls)], ["Monitoring probes", len([c for c in calls if c.get("call_type") == "monitoring")],], ["Successful probes", aggregate.get("monitoring_calls_successful")], ["Brand mention rate", aggregate.get("brand_mention_rate")], ["Brand recommendation rate", aggregate.get("brand_recommendation_rate")], ["Avg citation likelihood", aggregate.get("avg_citation_likelihood")]]
    story.append(Spacer(1, 8))
    story.append(Table(rows, repeatRows=1))
    story.append(Paragraph("Call Ledger", styles["Heading2"]))
    ledger = [["Call", "Stage", "Type", "Query", "Tokens", "Success"]]
    for call in calls[:120]:
        ledger.append([call.get("call_id"), call.get("stage"), call.get("call_type"), str(call.get("question", ""))[:80], (call.get("usage") or {}).get("total_tokens", ""), call.get("success")])
    story.append(Table(ledger, repeatRows=1))
    for item in story:
        if isinstance(item, Table):
            item.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e6fb")), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 6), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=10*mm, rightMargin=10*mm, topMargin=10*mm, bottomMargin=10*mm).build(story)


def write_outputs(output_dir: Path, brand_name: str, brand_brief: str, setup: Dict[str, Any], aggregate: Dict[str, Any], synthesis: Dict[str, Any], calls: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "deepseek_calls.json").write_text(json.dumps(calls, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "raw_runs.jsonl").open("w", encoding="utf-8") as f:
        for call in calls:
            f.write(json.dumps(call, ensure_ascii=False) + "\n")
    (output_dir / "research_setup.json").write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "synthesis.json").write_text(json.dumps(synthesis, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"brand": brand_name, "generated_at": now_utc(), "total_deepseek_calls": len(calls), "monitoring_probes": len([c for c in calls if c.get("call_type") == "monitoring"]), "brand_mention_rate": aggregate.get("brand_mention_rate"), "brand_recommendation_rate": aggregate.get("brand_recommendation_rate"), "dimension_scores": synthesis.get("dimension_scores") or aggregate.get("dimension_scores")}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text("# Full GEO Monitoring Dashboard\n\n" + json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    render_dashboard(output_dir, brand_name, brand_brief, setup, aggregate, synthesis, calls)
    build_pdf(output_dir / "dashboard.pdf", brand_name, aggregate, synthesis, calls)


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
    parser = argparse.ArgumentParser(description="Run full-detail GEO monitoring system")
    parser.add_argument("--brand-name", required=True)
    parser.add_argument("--brand-brief", required=True)
    parser.add_argument("--website", default="")
    parser.add_argument("--monitor-runs", type=int, default=120)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--output", default="dist/report")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--report-subdir", default="reports/latest")
    parser.add_argument("--commit-message", default="chore: update full GEO monitoring dashboard")
    parser.add_argument("--commit-report", action="store_true")
    args = parser.parse_args()
    monitor_runs = int(clamp(args.monitor_runs, 5, 300))
    output_dir = Path(args.output).resolve()
    repo_root = Path(args.repo_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    client = DeepSeekMonitor(args.model)
    setup = client.call_json("01_monitoring_plan_generation", setup_prompt(args.brand_name, args.brand_brief, args.website, monitor_runs), call_type="setup")
    query_plan = normalize_plan(setup, monitor_runs)
    competitor_names = [str(c.get("name")) for c in setup.get("competitors", []) if isinstance(c, dict) and c.get("name")]
    probe_calls_before = len(client.calls)
    for item in query_plan:
        client.call_json("02_deepseek_geo_probe", monitoring_prompt(args.brand_name, args.brand_brief, item, competitor_names), question=item["question"], query_id=item["query_id"], call_type="monitoring")
    probe_calls = client.calls[probe_calls_before:]
    aggregate = compute_aggregate(args.brand_name, query_plan, probe_calls)
    sample_results = [call.get("response_json", {}) for call in probe_calls if call.get("success")][:20]
    synthesis = client.call_json("03_aggregate_synthesis", synthesis_prompt(args.brand_name, setup, aggregate, sample_results), call_type="synthesis")
    write_outputs(output_dir, args.brand_name, args.brand_brief, setup, aggregate, synthesis, client.calls)
    commit_info = commit_report(repo_root, output_dir, args.report_subdir, args.commit_message) if args.commit_report else {"committed": False}
    print(json.dumps({"brand": args.brand_name, "total_deepseek_calls": len(client.calls), "monitoring_probes": len(probe_calls), "commit": commit_info}, ensure_ascii=False))


if __name__ == "__main__":
    main()
