#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree as ET

import requests
import yaml
from bs4 import BeautifulSoup

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
except Exception:
    colors = None

DIMS = ["visibility", "inclusion", "cognition", "outcome"]
DEFAULT_WEIGHTS = {"visibility": 35, "inclusion": 25, "cognition": 25, "outcome": 15}
EVIDENCE_KEYS = [
    "owned_surface_strength", "entity_clarity", "content_modularity", "trust_signal_density",
    "comparison_page_readiness", "faq_readiness", "documentation_readiness", "pricing_transparency",
    "schema_readiness", "narrative_control",
]
MARKET_PRESSURE_KEYS = ["peer_activation_index", "benchmark_percentile", "urgency_score", "gap_to_leading_peer", "narrative_disadvantage"]
JOURNEY_STAGES = ["Discover", "Consider", "Validate", "Select", "Expand"]
STOPWORDS = set("a an and are as at be by can for from has have in into is it its of on or our the their this to we with you your about product products solution solutions technology technologies storage data enterprise more learn page site kioxia".split())
SURFACE_PATTERNS = {
    "about": r"about|company|profile|who-we-are",
    "product": r"product|ssd|flash|memory|storage|drive|nand|ufs|emmc",
    "enterprise": r"enterprise|data-center|datacenter|server|cloud",
    "support": r"support|download|faq|help|contact|manual",
    "trust": r"quality|security|privacy|compliance|warranty|sustainability|environment",
    "comparison": r"compare|comparison|versus|vs|benchmark",
    "pricing": r"price|pricing|quote|buy|where-to-buy",
    "docs": r"whitepaper|document|resource|spec|technical|developer",
    "news": r"news|press|release|blog|insight",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clamp_score(value: Any) -> float:
    try:
        n = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(100.0, n))


def clamp_ratio(value: Any) -> float:
    try:
        n = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, n))


def avg(values: List[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
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


def safe_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "brand"


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: Optional[str], model: str):
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required")
        self.api_key = api_key
        self.base_url = (base_url or "https://api.deepseek.com/v1/chat/completions").strip()
        self.model = model
        self.audit: List[Dict[str, Any]] = []

    def chat_json(self, label: str, system_prompt: str, user_prompt: str, timeout: int = 180) -> Dict[str, Any]:
        prompt_hash = hashlib.sha256((system_prompt + "\n" + user_prompt).encode("utf-8")).hexdigest()[:12]
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        started_at = now_utc()
        resp = requests.post(self.base_url, headers=headers, json=payload, timeout=timeout)
        status_code = resp.status_code
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        data = extract_json(content)
        self.audit.append({
            "label": label,
            "model": self.model,
            "started_at": started_at,
            "status_code": status_code,
            "prompt_hash": prompt_hash,
            "response_chars": len(content),
        })
        return data


def auto_benchmark_schema() -> Dict[str, Any]:
    return {
        "brand_profile": {"brand_name": "", "official_website": "", "market": "", "region": "", "language": "", "inferred_category": "", "brand_summary": "", "confidence": 0.0, "uncertainties": [""]},
        "competitors": [{"name": "", "why_in_set": "", "confidence": 0.0, "geo_maturity_stage": "", "overall_score_estimate": 0, "dimension_scores": {"visibility": 0, "inclusion": 0, "cognition": 0, "outcome": 0}, "strengths": [""], "evidence_signals": [""]}],
        "query_panel": [{"type": "brand", "query": "", "intent": "", "funnel_stage": "Consider", "importance": 0}],
        "keyword_taxonomy": {"brand_terms": [""], "competitor_terms": [""], "category_terms": [""], "industry_terms": [""], "problem_terms": [""]},
        "evidence_map": {k: 0 for k in EVIDENCE_KEYS},
        "market_pressure": {k: 0 for k in MARKET_PRESSURE_KEYS},
        "journey_gap_matrix": [{"stage": "Discover", "current_strength": 0, "competitor_pressure": 0, "opportunity": 0, "notes": ""}],
        "geo_evaluation": {
            "methodology_note": "",
            "visibility": {"score": 0, "metrics": {"brand_mention_likelihood": 0, "first_party_citation_likelihood": 0, "comparative_presence": 0, "weighted_visibility": 0}, "rationale": "", "confidence": 0.0, "priority_actions": [""]},
            "inclusion": {"score": 0, "metrics": {"crawl_index_readiness": 0, "entity_clarity": 0, "structured_content_readiness": 0, "knowledge_asset_completeness": 0}, "rationale": "", "confidence": 0.0, "priority_actions": [""]},
            "cognition": {"score": 0, "metrics": {"definition_accuracy_likelihood": 0, "attribute_recall_likelihood": 0, "narrative_alignment_likelihood": 0, "hallucination_resilience": 0}, "rationale": "", "confidence": 0.0, "priority_actions": [""]},
            "outcome": {"score": 0, "metrics": {"visit_intent_capture": 0, "conversion_readiness": 0, "brand_search_lift_potential": 0, "measurement_maturity": 0}, "rationale": "", "confidence": 0.0, "priority_actions": [""]},
            "strengths": [""], "risks": [""], "executive_summary": "",
        },
    }


def create_auto_benchmark(brand_cfg: Dict[str, Any], client: DeepSeekClient) -> Dict[str, Any]:
    brand = brand_cfg.get("brand", {})
    prompt = "\n".join([
        "Generate a rigorous GEO benchmark JSON for this brand. Do not claim real-time web access.",
        f"Brand: {brand.get('name', '')}",
        f"Website: {brand.get('website', '')}",
        f"Market: {brand.get('market', 'Global')}",
        f"Region: {brand.get('region', brand.get('market', 'Global'))}",
        f"Language: {brand.get('language', 'zh-CN')}",
        f"Category: {brand.get('category', '')}",
        f"Known competitors: {json.dumps(brand.get('competitors', []) or [], ensure_ascii=False)}",
        f"Narratives: {json.dumps(brand.get('narratives', []) or [], ensure_ascii=False)}",
        "Requirements:",
        "1. Output JSON only.",
        "2. Produce at least 30 private query_panel items across brand, category, problem, comparison, use_case, trust.",
        "3. Include keyword_taxonomy with brand_terms, competitor_terms, category_terms, industry_terms, problem_terms.",
        "4. Include competitors with GEO maturity, dimension scores, strengths, and visible evidence signals.",
        "5. Include evidence_map, market_pressure, journey_gap_matrix, and four-dimension GEO evaluation.",
        "6. Make competitive pressure explicit: peers are already more activated; the target has gaps.",
        "7. Scores must be conservative and not all high.",
        "Schema example:",
        json.dumps(auto_benchmark_schema(), ensure_ascii=False, indent=2),
    ])
    system = "You are a strict GEO strategy analyst. Return complete JSON only. Do not expose methodology prompts."
    return client.chat_json("benchmark_generation", system, prompt)


def discover_urls(root_url: str, max_pages: int) -> Tuple[List[str], Dict[str, Any]]:
    urls: List[str] = []
    meta = {"robots_status": None, "sitemap_status": None, "discovery_errors": []}
    parsed = urlparse(root_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    candidates = [root_url.rstrip("/"), urljoin(base, "/robots.txt"), urljoin(base, "/sitemap.xml")]
    session = requests.Session()
    session.headers.update({"User-Agent": "eco-geo-bot/1.0 (+https://github.com/yt-feng/eco-geo)"})
    urls.append(root_url.rstrip("/"))
    try:
        r = session.get(urljoin(base, "/robots.txt"), timeout=15)
        meta["robots_status"] = r.status_code
        if r.ok:
            for line in r.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    candidates.append(line.split(":", 1)[1].strip())
    except Exception as exc:
        meta["discovery_errors"].append(f"robots: {exc}")
    try:
        r = session.get(urljoin(base, "/sitemap.xml"), timeout=20)
        meta["sitemap_status"] = r.status_code
        if r.ok:
            try:
                xml_root = ET.fromstring(r.content)
                ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
                locs = [el.text for el in xml_root.findall(".//sm:loc", ns) if el.text] or [el.text for el in xml_root.findall(".//loc") if el.text]
                for loc in locs[: max_pages * 3]:
                    if urlparse(loc).netloc == parsed.netloc:
                        urls.append(loc)
            except Exception as exc:
                meta["discovery_errors"].append(f"sitemap_parse: {exc}")
    except Exception as exc:
        meta["discovery_errors"].append(f"sitemap: {exc}")
    try:
        r = session.get(root_url, timeout=20)
        if r.ok:
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = urljoin(root_url, a["href"]).split("#")[0]
                if urlparse(href).netloc == parsed.netloc:
                    urls.append(href)
    except Exception as exc:
        meta["discovery_errors"].append(f"homepage_links: {exc}")
    cleaned: List[str] = []
    seen = set()
    for u in urls:
        u = u.split("#")[0].rstrip("/")
        if u and u not in seen and len(cleaned) < max_pages:
            seen.add(u)
            cleaned.append(u)
    return cleaned, meta


def visible_text_from_soup(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))[:200000]


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9+\-]{2,}", text.lower())
    return [w.strip("-+") for w in words if w not in STOPWORDS and len(w) >= 3]


def analyze_site(root_url: str, max_pages: int) -> Dict[str, Any]:
    if not root_url:
        return {"enabled": False, "pages": [], "metrics": {}, "word_cloud": []}
    urls, discovery_meta = discover_urls(root_url, max_pages)
    session = requests.Session()
    session.headers.update({"User-Agent": "eco-geo-bot/1.0 (+https://github.com/yt-feng/eco-geo)"})
    pages: List[Dict[str, Any]] = []
    term_counter: Counter = Counter()
    schema_counter: Counter = Counter()
    surface_counts: Counter = Counter()
    success = 0
    for url in urls:
        page: Dict[str, Any] = {"url": url, "status": None, "title": "", "h1": [], "meta_description": "", "schema_types": [], "surfaces": []}
        try:
            r = session.get(url, timeout=20, allow_redirects=True)
            page["status"] = r.status_code
            page["final_url"] = r.url
            if r.ok and "text/html" in r.headers.get("content-type", ""):
                success += 1
                soup = BeautifulSoup(r.text, "html.parser")
                page["title"] = (soup.title.get_text(" ", strip=True) if soup.title else "")[:180]
                page["h1"] = [h.get_text(" ", strip=True)[:140] for h in soup.find_all("h1")[:3]]
                meta_desc = soup.find("meta", attrs={"name": re.compile("description", re.I)})
                page["meta_description"] = meta_desc.get("content", "")[:260] if meta_desc else ""
                for script in soup.find_all("script", attrs={"type": re.compile("ld\+json", re.I)}):
                    try:
                        data = json.loads(script.string or "{}")
                        items = data if isinstance(data, list) else [data]
                        for item in items:
                            typ = item.get("@type") if isinstance(item, dict) else None
                            if isinstance(typ, list):
                                for t in typ:
                                    schema_counter[str(t)] += 1
                                    page["schema_types"].append(str(t))
                            elif typ:
                                schema_counter[str(typ)] += 1
                                page["schema_types"].append(str(typ))
                    except Exception:
                        pass
                text = visible_text_from_soup(soup)
                page["word_count"] = len(text.split())
                page["internal_link_count"] = len([a for a in soup.find_all("a", href=True) if urlparse(urljoin(url, a["href"])).netloc == urlparse(root_url).netloc])
                tokens = tokenize(text + " " + url)
                term_counter.update(tokens)
                haystack = (url + " " + page["title"] + " " + " ".join(page["h1"]) + " " + page["meta_description"]).lower()
                for surface, pattern in SURFACE_PATTERNS.items():
                    if re.search(pattern, haystack):
                        surface_counts[surface] += 1
                        page["surfaces"].append(surface)
            pages.append(page)
        except Exception as exc:
            page["error"] = str(exc)
            pages.append(page)
    total = len(urls) or 1
    pages_with_title = sum(1 for p in pages if p.get("title"))
    pages_with_h1 = sum(1 for p in pages if p.get("h1"))
    pages_with_meta = sum(1 for p in pages if p.get("meta_description"))
    pages_with_schema = sum(1 for p in pages if p.get("schema_types"))
    word_cloud = [{"term": term, "weight": count} for term, count in term_counter.most_common(60)]
    return {
        "enabled": True,
        "root_url": root_url,
        "fetched_at": now_utc(),
        "discovery": discovery_meta,
        "metrics": {
            "discovered_urls": len(urls),
            "pages_attempted": len(urls),
            "pages_successful": success,
            "fetch_success_rate": round(success / total * 100, 2),
            "title_coverage": round(pages_with_title / total * 100, 2),
            "h1_coverage": round(pages_with_h1 / total * 100, 2),
            "meta_description_coverage": round(pages_with_meta / total * 100, 2),
            "schema_coverage": round(pages_with_schema / total * 100, 2),
        },
        "surface_counts": dict(surface_counts),
        "schema_types": dict(schema_counter),
        "word_cloud": word_cloud,
        "pages": pages,
    }


def normalize_dimension(raw: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    metrics = raw.get("metrics", {}) if isinstance(raw, dict) else {}
    return {
        "score": clamp_score(raw.get("score", 0) if isinstance(raw, dict) else 0),
        "metrics": {k: clamp_score(metrics.get(k, 0)) for k in keys},
        "rationale": str(raw.get("rationale", "") if isinstance(raw, dict) else "").strip(),
        "confidence": clamp_ratio(raw.get("confidence", 0) if isinstance(raw, dict) else 0),
        "priority_actions": [str(x) for x in (raw.get("priority_actions", []) if isinstance(raw, dict) else [])][:5],
    }


def normalize_competitors(items: Any) -> List[Dict[str, Any]]:
    result = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        raw_dims = item.get("dimension_scores", {}) if isinstance(item.get("dimension_scores"), dict) else {}
        dim_scores = {d: clamp_score(raw_dims.get(d, item.get(d, 0))) for d in DIMS}
        overall = clamp_score(item.get("overall_score_estimate", avg(list(dim_scores.values()))))
        result.append({
            "name": str(item.get("name", "Unknown competitor")),
            "why_in_set": str(item.get("why_in_set", "")),
            "confidence": clamp_ratio(item.get("confidence", 0)),
            "geo_maturity_stage": str(item.get("geo_maturity_stage", infer_stage(overall))) or infer_stage(overall),
            "overall_score_estimate": overall,
            "dimension_scores": dim_scores,
            "strengths": [str(x) for x in item.get("strengths", [])][:4],
            "evidence_signals": [str(x) for x in item.get("evidence_signals", [])][:5],
        })
    return sorted(result, key=lambda x: x["overall_score_estimate"], reverse=True)[:6]


def normalize_query_panel(items: Any) -> List[Dict[str, Any]]:
    panel = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        panel.append({
            "type": str(item.get("type", "other")),
            "query": query,
            "intent": str(item.get("intent", "")),
            "funnel_stage": str(item.get("funnel_stage", "Consider")),
            "importance": clamp_score(item.get("importance", 50)),
        })
    return panel[:80]


def normalize_numeric_map(raw: Any, keys: List[str]) -> Dict[str, float]:
    raw = raw if isinstance(raw, dict) else {}
    return {k: clamp_score(raw.get(k, 0)) for k in keys}


def normalize_journey(items: Any) -> List[Dict[str, Any]]:
    rows = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage", "Other"))
        rows[stage] = {"stage": stage, "current_strength": clamp_score(item.get("current_strength", 0)), "competitor_pressure": clamp_score(item.get("competitor_pressure", 0)), "opportunity": clamp_score(item.get("opportunity", 0)), "notes": str(item.get("notes", ""))}
    for stage in JOURNEY_STAGES:
        rows.setdefault(stage, {"stage": stage, "current_strength": 0, "competitor_pressure": 0, "opportunity": 0, "notes": ""})
    return [rows[s] for s in JOURNEY_STAGES]


def run_answer_probes(client: DeepSeekClient, brand_name: str, competitors: List[Dict[str, Any]], query_panel: List[Dict[str, Any]], limit: int) -> Dict[str, Any]:
    selected = sorted(query_panel, key=lambda q: q.get("importance", 0), reverse=True)[:limit]
    comp_names = [c["name"] for c in competitors]
    results = []
    for idx, item in enumerate(selected, 1):
        schema = {"answer_summary": "", "mentioned_target": False, "mentioned_competitors": [], "dominant_entity": "", "narrative_terms": [], "confidence": 0.0}
        prompt = "\n".join([
            "Answer the following market question as a concise answer-engine response, then return JSON only.",
            f"Target brand: {brand_name}",
            f"Competitors to watch: {json.dumps(comp_names, ensure_ascii=False)}",
            f"Question: {item['query']}",
            "Return whether the target brand and competitors appear in your answer.",
            json.dumps(schema, ensure_ascii=False),
        ])
        try:
            data = client.chat_json(f"answer_probe_{idx:02d}", "Return JSON only. Be concise and do not invent citations.", prompt, timeout=120)
        except Exception as exc:
            data = {"error": str(exc), "mentioned_target": False, "mentioned_competitors": [], "confidence": 0.0}
        results.append({"query": item, "probe": data})
    mention_rate = round(sum(1 for r in results if r["probe"].get("mentioned_target")) / max(1, len(results)) * 100, 2)
    comp_counter = Counter()
    narrative_counter = Counter()
    for r in results:
        for c in r["probe"].get("mentioned_competitors", []) if isinstance(r["probe"].get("mentioned_competitors", []), list) else []:
            comp_counter[str(c)] += 1
        for t in r["probe"].get("narrative_terms", []) if isinstance(r["probe"].get("narrative_terms", []), list) else []:
            narrative_counter[str(t)] += 1
    return {"enabled": True, "probe_limit": limit, "total_probes": len(results), "target_mention_rate": mention_rate, "competitor_mentions": dict(comp_counter), "narrative_terms": dict(narrative_counter), "results": results}


def build_report(brand_cfg: Dict[str, Any], benchmark: Dict[str, Any], site: Dict[str, Any], probes: Dict[str, Any], api_audit: List[Dict[str, Any]]) -> Dict[str, Any]:
    brand_cfg_inner = brand_cfg.get("brand", {})
    geo = benchmark.get("geo_evaluation", {}) if isinstance(benchmark.get("geo_evaluation"), dict) else {}
    brand_profile = benchmark.get("brand_profile", {}) if isinstance(benchmark.get("brand_profile"), dict) else {}
    dims = {
        "visibility": normalize_dimension(geo.get("visibility", {}), ["brand_mention_likelihood", "first_party_citation_likelihood", "comparative_presence", "weighted_visibility"]),
        "inclusion": normalize_dimension(geo.get("inclusion", {}), ["crawl_index_readiness", "entity_clarity", "structured_content_readiness", "knowledge_asset_completeness"]),
        "cognition": normalize_dimension(geo.get("cognition", {}), ["definition_accuracy_likelihood", "attribute_recall_likelihood", "narrative_alignment_likelihood", "hallucination_resilience"]),
        "outcome": normalize_dimension(geo.get("outcome", {}), ["visit_intent_capture", "conversion_readiness", "brand_search_lift_potential", "measurement_maturity"]),
    }
    weights = brand_cfg.get("weights", DEFAULT_WEIGHTS)
    overall = round(sum(dims[d]["score"] * (float(weights.get(d, 0)) / max(1, sum(float(v) for v in weights.values()))) for d in DIMS), 2)
    competitors = normalize_competitors(benchmark.get("competitors", []))
    query_panel = normalize_query_panel(benchmark.get("query_panel", []))
    evidence = normalize_numeric_map(benchmark.get("evidence_map", {}), EVIDENCE_KEYS)
    market_pressure = normalize_numeric_map(benchmark.get("market_pressure", {}), MARKET_PRESSURE_KEYS)
    journey = normalize_journey(benchmark.get("journey_gap_matrix", []))
    top_peer = competitors[0] if competitors else None
    competitive_gap = round((top_peer["overall_score_estimate"] - overall) if top_peer else 0, 2)
    if market_pressure["peer_activation_index"] == 0 and competitors:
        market_pressure["peer_activation_index"] = avg([c["overall_score_estimate"] for c in competitors])
    if market_pressure["gap_to_leading_peer"] == 0:
        market_pressure["gap_to_leading_peer"] = max(0, competitive_gap)
    if market_pressure["urgency_score"] == 0:
        market_pressure["urgency_score"] = max(0, min(100, 100 - overall + max(0, competitive_gap) / 2))
    taxonomy = benchmark.get("keyword_taxonomy", {}) if isinstance(benchmark.get("keyword_taxonomy"), dict) else {}
    for key in ["brand_terms", "competitor_terms", "category_terms", "industry_terms", "problem_terms"]:
        taxonomy.setdefault(key, [])
    site_terms = site.get("word_cloud", [])[:40]
    query_counts = Counter(q["type"] for q in query_panel)
    funnel_counts = Counter(q["funnel_stage"] for q in query_panel)
    return {
        "generated_at": now_utc(),
        "brand": {"name": brand_cfg_inner.get("name", brand_profile.get("brand_name", "Unknown Brand")), "website": brand_cfg_inner.get("website", brand_profile.get("official_website", "")), "market": brand_cfg_inner.get("market", brand_profile.get("market", "")), "category": brand_cfg_inner.get("category", brand_profile.get("inferred_category", ""))},
        "overall_score": overall,
        "overall_level": infer_stage(overall),
        "dimensions": {d: {**dims[d], "stage": infer_stage(dims[d]["score"])} for d in DIMS},
        "competitors": competitors,
        "top_peer": top_peer,
        "competitive_gap": competitive_gap,
        "market_pressure": market_pressure,
        "query_panel": query_panel,
        "query_summary": {"total_queries": len(query_panel), "types": dict(query_counts), "funnel_stages": dict(funnel_counts), "avg_importance": avg([q["importance"] for q in query_panel])},
        "keyword_taxonomy": taxonomy,
        "site_snapshot": site,
        "answer_probes": probes,
        "monitoring_summary": {"deepseek_api_calls": len(api_audit), "answer_probe_calls": probes.get("total_probes", 0), "private_query_candidates": len(query_panel), "site_pages_attempted": site.get("metrics", {}).get("pages_attempted", 0), "site_pages_successful": site.get("metrics", {}).get("pages_successful", 0)},
        "evidence_map": evidence,
        "journey_gap_matrix": journey,
        "strengths": [str(x) for x in geo.get("strengths", [])][:6],
        "risks": [str(x) for x in geo.get("risks", [])][:6],
        "executive_summary": str(geo.get("executive_summary", "")).strip(),
        "methodology_note": str(geo.get("methodology_note", "Daily GEO snapshot combining live owned-site crawl, answer-engine probes, competitive benchmark, and private query taxonomy.")),
    }


def esc(x: Any) -> str:
    return html.escape(str(x))


def bar(label: str, value: float, tone: str = "blue") -> str:
    colors_map = {"blue": "#60a5fa", "green": "#34d399", "orange": "#f59e0b", "red": "#fb7185", "purple": "#a78bfa"}
    c = colors_map.get(tone, "#60a5fa")
    return f"<div class='bar'><div class='bar-meta'><span>{esc(label)}</span><b>{round(value,1)}</b></div><div class='rail'><div class='fill' style='width:{max(0,min(100,value))}%;background:{c}'></div></div></div>"


def word_cloud_html(terms: List[Dict[str, Any]], limit: int = 50) -> str:
    if not terms:
        return "<div class='muted'>No terms extracted.</div>"
    weights = [int(t.get("weight", 1)) for t in terms[:limit]]
    min_w, max_w = min(weights), max(weights)
    spans = []
    for item in terms[:limit]:
        w = int(item.get("weight", 1))
        size = 13 + (24 * (w - min_w) / max(1, max_w - min_w))
        spans.append(f"<span style='font-size:{round(size,1)}px'>{esc(item.get('term',''))}</span>")
    return "<div class='word-cloud'>" + " ".join(spans) + "</div>"


def render_dashboard_html(report: Dict[str, Any]) -> str:
    brand = report["brand"]
    metrics = report["site_snapshot"].get("metrics", {})
    competitor_rows = []
    competitor_rows.append(f"<tr><td><b>{esc(brand['name'])}</b></td><td>{esc(report['overall_level'])}</td><td>{report['overall_score']}</td><td>{report['dimensions']['visibility']['score']}</td><td>{report['dimensions']['inclusion']['score']}</td><td>{report['dimensions']['cognition']['score']}</td><td>{report['dimensions']['outcome']['score']}</td><td>Client baseline</td></tr>")
    for c in report["competitors"]:
        ds = c["dimension_scores"]
        competitor_rows.append(f"<tr><td><b>{esc(c['name'])}</b></td><td>{esc(c['geo_maturity_stage'])}</td><td>{c['overall_score_estimate']}</td><td>{ds['visibility']}</td><td>{ds['inclusion']}</td><td>{ds['cognition']}</td><td>{ds['outcome']}</td><td>{esc(', '.join(c.get('evidence_signals', [])[:2]) or c.get('why_in_set',''))}</td></tr>")
    dim_cards = "".join(f"<div class='card'><h3>{d.title()}</h3>{bar('Score', report['dimensions'][d]['score'], 'blue')}" + "".join(bar(k.replace('_',' ').title(), v, 'purple') for k,v in report['dimensions'][d]['metrics'].items()) + f"<p class='muted'>{esc(report['dimensions'][d]['rationale'])}</p></div>" for d in DIMS)
    evidence_cards = "".join(f"<div class='mini'><div class='label'>{esc(k.replace('_',' ').title())}</div>{bar('', v, 'green' if v>=70 else 'orange' if v>=45 else 'red')}</div>" for k,v in report['evidence_map'].items())
    taxonomy_cards = "".join(f"<div class='mini'><div class='label'>{esc(k.replace('_',' ').title())}</div><p>{esc(', '.join([str(x) for x in v[:12]]))}</p></div>" for k,v in report['keyword_taxonomy'].items())
    surface_cards = "".join(f"<div class='mini'><div class='label'>{esc(k)}</div><div class='big'>{v}</div></div>" for k,v in report['site_snapshot'].get('surface_counts', {}).items())
    query_cards = "".join(f"<div class='mini'><div class='label'>{esc(k)}</div><div class='big'>{v}</div></div>" for k,v in report['query_summary']['types'].items())
    journey_cards = "".join(f"<div class='card'><h3>{esc(j['stage'])}</h3>{bar('Current strength', j['current_strength'], 'blue')}{bar('Competitor pressure', j['competitor_pressure'], 'red')}{bar('Opportunity', j['opportunity'], 'green')}<p class='muted'>{esc(j['notes'])}</p></div>" for j in report['journey_gap_matrix'])
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{esc(brand['name'])} GEO Dashboard</title><style>
body{{margin:0;background:#08111f;color:#e8eef9;font-family:Inter,Arial,sans-serif}}.wrap{{max-width:1500px;margin:auto;padding:28px}}.hero,.grid2{{display:grid;grid-template-columns:1.4fr .9fr;gap:18px}}.grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.grid5{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px}}.card,.mini{{background:#101d31;border:1px solid #243550;border-radius:18px;padding:18px;box-shadow:0 10px 30px rgba(0,0,0,.22)}}h1{{font-size:38px;margin:0 0 8px}}h2{{margin-top:28px}}h3{{margin:0 0 12px}}.muted{{color:#9aabc3;line-height:1.55}}.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:16px}}.label{{color:#93a4bb;font-size:12px;text-transform:uppercase;letter-spacing:.06em}}.big{{font-size:30px;font-weight:800;margin-top:6px}}.bar{{margin:9px 0}}.bar-meta{{display:flex;justify-content:space-between;font-size:13px;margin-bottom:5px}}.rail{{height:9px;background:#07101d;border-radius:999px;border:1px solid #25364e;overflow:hidden}}.fill{{height:100%;border-radius:999px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #26364d;text-align:left;vertical-align:top}}th{{color:#bdd0ea}}.word-cloud span{{display:inline-block;margin:7px 9px;color:#dbeafe}}.section{{margin:22px 0}}@media(max-width:1100px){{.hero,.grid2,.grid4,.grid5,.kpis{{grid-template-columns:1fr}}}}</style></head><body><div class='wrap'>
<div class='hero'><div class='card'><div class='label'>Daily GEO Command Center</div><h1>{esc(brand['name'])}</h1><p class='muted'>{esc(report['executive_summary'])}</p><div class='kpis'><div class='mini'><div class='label'>Overall</div><div class='big'>{report['overall_score']}</div></div><div class='mini'><div class='label'>Gap to leader</div><div class='big'>{report['competitive_gap']}</div></div><div class='mini'><div class='label'>DeepSeek calls</div><div class='big'>{report['monitoring_summary']['deepseek_api_calls']}</div></div><div class='mini'><div class='label'>Pages crawled</div><div class='big'>{report['monitoring_summary']['site_pages_successful']}</div></div></div></div><div class='card'><h3>Live owned-site snapshot</h3>{bar('Fetch success rate', metrics.get('fetch_success_rate',0),'green')}{bar('Schema coverage', metrics.get('schema_coverage',0),'orange')}{bar('H1 coverage', metrics.get('h1_coverage',0),'blue')}{bar('Meta description coverage', metrics.get('meta_description_coverage',0),'purple')}</div></div>
<h2>Competitive leaderboard</h2><div class='card'><table><thead><tr><th>Brand</th><th>Stage</th><th>Overall</th><th>Visibility</th><th>Inclusion</th><th>Cognition</th><th>Outcome</th><th>Signals</th></tr></thead><tbody>{''.join(competitor_rows)}</tbody></table></div>
<h2>Private query universe summary</h2><div class='grid2'><div class='card'><h3>Question family mix</h3><div class='grid5'>{query_cards}</div><p class='muted'>Exact private questions are withheld from the client dashboard; this view shows only aggregate families and topic distribution.</p></div><div class='card'><h3>Answer-engine probe snapshot</h3>{bar('Target mention rate', report['answer_probes'].get('target_mention_rate',0),'blue')}<p class='muted'>DeepSeek answer probes run today: {report['answer_probes'].get('total_probes',0)}. Competitor mentions observed: {esc(json.dumps(report['answer_probes'].get('competitor_mentions',{}),ensure_ascii=False))}</p></div></div>
<h2>Topic cloud from live site and query taxonomy</h2><div class='card'>{word_cloud_html(report['site_snapshot'].get('word_cloud', []), 60)}</div>
<h2>Keyword taxonomy, internal summary</h2><div class='grid5'>{taxonomy_cards}</div>
<h2>Owned surface coverage</h2><div class='grid5'>{surface_cards}</div>
<h2>GEO dimension decomposition</h2><div class='grid4'>{dim_cards}</div>
<h2>Evidence map</h2><div class='grid5'>{evidence_cards}</div>
<h2>Journey gap matrix</h2><div class='grid5'>{journey_cards}</div>
<p class='muted'>Generated at {esc(report['generated_at'])}. {esc(report['methodology_note'])}</p>
</div></body></html>"""


def render_report_md(report: Dict[str, Any]) -> str:
    lines = [f"# {report['brand']['name']} GEO Daily Snapshot", "", f"Generated at: {report['generated_at']}", "", f"Overall score: **{report['overall_score']}**", f"DeepSeek calls: **{report['monitoring_summary']['deepseek_api_calls']}**", f"Private query candidates: **{report['monitoring_summary']['private_query_candidates']}**", f"Site pages crawled: **{report['monitoring_summary']['site_pages_successful']} / {report['monitoring_summary']['site_pages_attempted']}**", "", "## Competitor Leaderboard", "", "| Brand | Overall | Visibility | Inclusion | Cognition | Outcome |", "|---|---:|---:|---:|---:|---:|", f"| {report['brand']['name']} | {report['overall_score']} | {report['dimensions']['visibility']['score']} | {report['dimensions']['inclusion']['score']} | {report['dimensions']['cognition']['score']} | {report['dimensions']['outcome']['score']} |"]
    for c in report['competitors']:
        ds = c['dimension_scores']
        lines.append(f"| {c['name']} | {c['overall_score_estimate']} | {ds['visibility']} | {ds['inclusion']} | {ds['cognition']} | {ds['outcome']} |")
    lines += ["", "## Query Family Summary"]
    for k, v in report['query_summary']['types'].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Live Site Snapshot"]
    for k, v in report['site_snapshot'].get('metrics', {}).items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def render_internal_audit_md(report: Dict[str, Any], api_audit: List[Dict[str, Any]]) -> str:
    lines = [f"# Internal GEO Audit - {report['brand']['name']}", "", "This file is for operator review. Do not share as the client dashboard.", "", "## Monitoring Footprint", ""]
    for k, v in report['monitoring_summary'].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## DeepSeek API Calls", "", "| # | Label | Model | Status | Prompt Hash | Response Chars |", "|---:|---|---|---:|---|---:|"]
    for i, item in enumerate(api_audit, 1):
        lines.append(f"| {i} | {item['label']} | {item['model']} | {item['status_code']} | {item['prompt_hash']} | {item['response_chars']} |")
    lines += ["", "## Keyword Taxonomy"]
    for k, vals in report['keyword_taxonomy'].items():
        lines.append(f"### {k}\n" + ", ".join([str(v) for v in vals]))
    lines += ["", "## Private Query Panel"]
    grouped = defaultdict(list)
    for q in report['query_panel']:
        grouped[q['type']].append(q)
    for qtype, items in grouped.items():
        lines.append(f"### {qtype}")
        for q in items:
            lines.append(f"- [{q['funnel_stage']}; importance {q['importance']}] {q['query']} — {q['intent']}")
    lines += ["", "## DeepSeek Answer Probes"]
    for r in report['answer_probes'].get('results', []):
        q = r['query']; p = r['probe']
        lines.append(f"- Q: {q['query']}")
        lines.append(f"  - target mentioned: {p.get('mentioned_target')} | competitors: {p.get('mentioned_competitors')} | confidence: {p.get('confidence')}")
        if p.get('answer_summary'):
            lines.append(f"  - summary: {p.get('answer_summary')}")
    lines += ["", "## Crawled Pages"]
    for p in report['site_snapshot'].get('pages', []):
        lines.append(f"- {p.get('status')} {p.get('url')} | title: {p.get('title','')} | surfaces: {', '.join(p.get('surfaces', []))}")
    return "\n".join(lines)


def render_pdf(report: Dict[str, Any], output_path: Path) -> None:
    if colors is None:
        return
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    styles = getSampleStyleSheet()
    base = ParagraphStyle('base-cn', parent=styles['BodyText'], fontName='STSong-Light', fontSize=9, leading=12)
    title = ParagraphStyle('title-cn', parent=styles['Title'], fontName='STSong-Light', fontSize=20, leading=24)
    h2 = ParagraphStyle('h2-cn', parent=styles['Heading2'], fontName='STSong-Light', fontSize=14, leading=18)
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=0.55*inch, leftMargin=0.55*inch, topMargin=0.55*inch, bottomMargin=0.55*inch)
    story: List[Any] = [Paragraph(f"{report['brand']['name']} GEO Daily Dashboard", title), Spacer(1, 10), Paragraph(f"Generated at {report['generated_at']}", base), Spacer(1, 10)]
    kpi = [["Overall", "Gap to leader", "DeepSeek calls", "Pages crawled"], [str(report['overall_score']), str(report['competitive_gap']), str(report['monitoring_summary']['deepseek_api_calls']), f"{report['monitoring_summary']['site_pages_successful']} / {report['monitoring_summary']['site_pages_attempted']}"]]
    table = Table(kpi, colWidths=[1.6*inch]*4)
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dbeafe')),('GRID',(0,0),(-1,-1),0.3,colors.grey),('FONTNAME',(0,0),(-1,-1),'STSong-Light'),('ALIGN',(0,0),(-1,-1),'CENTER')]))
    story += [table, Spacer(1, 14), Paragraph('Executive Summary', h2), Paragraph(report.get('executive_summary',''), base), Spacer(1, 12), Paragraph('Competitor Leaderboard', h2)]
    rows = [["Brand", "Overall", "Visibility", "Inclusion", "Cognition", "Outcome"], [report['brand']['name'], report['overall_score'], report['dimensions']['visibility']['score'], report['dimensions']['inclusion']['score'], report['dimensions']['cognition']['score'], report['dimensions']['outcome']['score']]]
    for c in report['competitors'][:6]:
        ds = c['dimension_scores']; rows.append([c['name'], c['overall_score_estimate'], ds['visibility'], ds['inclusion'], ds['cognition'], ds['outcome']])
    t = Table(rows, repeatRows=1, colWidths=[1.45*inch, .75*inch, .75*inch, .75*inch, .75*inch, .75*inch])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dbeafe')),('GRID',(0,0),(-1,-1),0.25,colors.grey),('FONTNAME',(0,0),(-1,-1),'STSong-Light'),('FONTSIZE',(0,0),(-1,-1),8)]))
    story += [t, Spacer(1, 12), Paragraph('Query Family Summary', h2), Paragraph(json.dumps(report['query_summary']['types'], ensure_ascii=False), base), Spacer(1, 10), Paragraph('Live Site Snapshot', h2), Paragraph(json.dumps(report['site_snapshot'].get('metrics',{}), ensure_ascii=False), base), PageBreak(), Paragraph('Keyword Taxonomy', h2)]
    for k, vals in report['keyword_taxonomy'].items():
        story += [Paragraph(f"<b>{k}</b>: " + ', '.join([str(v) for v in vals[:18]]), base), Spacer(1, 6)]
    story += [Paragraph('Top Site Terms', h2), Paragraph(', '.join([x['term'] for x in report['site_snapshot'].get('word_cloud', [])[:40]]), base)]
    doc.build(story)


def write_outputs(output_dir: Path, report: Dict[str, Any], benchmark: Dict[str, Any], api_audit: List[Dict[str, Any]]) -> None:
    ensure_dir(output_dir)
    (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir / 'benchmark.generated.json').write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir / 'dashboard.html').write_text(render_dashboard_html(report), encoding='utf-8')
    (output_dir / 'report.md').write_text(render_report_md(report), encoding='utf-8')
    (output_dir / 'internal_audit.md').write_text(render_internal_audit_md(report, api_audit), encoding='utf-8')
    (output_dir / 'internal_audit.json').write_text(json.dumps({"api_calls": api_audit, "query_panel": report['query_panel'], "keyword_taxonomy": report['keyword_taxonomy'], "answer_probes": report['answer_probes'], "site_pages": report['site_snapshot'].get('pages', [])}, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir / 'summary.json').write_text(json.dumps({"brand": report['brand'], "overall_score": report['overall_score'], "competitive_gap": report['competitive_gap'], "deepseek_api_calls": report['monitoring_summary']['deepseek_api_calls'], "generated_at": report['generated_at']}, ensure_ascii=False, indent=2), encoding='utf-8')
    render_pdf(report, output_dir / 'dashboard.pdf')


def maybe_git_commit(repo_root: Path, source_dir: Path, target_dir: Path, message: str) -> Dict[str, Any]:
    ensure_dir(target_dir.parent)
    if target_dir.exists():
        subprocess.run(['rm', '-rf', str(target_dir)], check=True)
    subprocess.run(['mkdir', '-p', str(target_dir)], check=True)
    subprocess.run(['cp', '-R', f'{source_dir}/.', str(target_dir)], check=True)
    subprocess.run(['git', 'add', str(target_dir)], cwd=repo_root, check=True)
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=repo_root, capture_output=True, text=True, check=True)
    if not status.stdout.strip():
        return {"committed": False, "target_dir": str(target_dir)}
    subprocess.run(['git', 'commit', '-m', message], cwd=repo_root, check=True)
    return {"committed": True, "target_dir": str(target_dir)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--brand-config', default='config/brand.yaml')
    parser.add_argument('--manual-benchmark', default='')
    parser.add_argument('--output', default='dist/report')
    parser.add_argument('--model', default='deepseek-chat')
    parser.add_argument('--probe-limit', type=int, default=12)
    parser.add_argument('--site-max-pages', type=int, default=25)
    parser.add_argument('--commit-report', action='store_true')
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--report-subdir', default='reports/latest')
    parser.add_argument('--commit-message', default='chore: update GEO daily dashboard')
    args = parser.parse_args()
    brand_cfg = load_yaml(Path(args.brand_config))
    brand = brand_cfg.get('brand', {})
    client = DeepSeekClient(os.getenv('DEEPSEEK_API_KEY', ''), os.getenv('DEEPSEEK_BASE_URL', ''), args.model)
    benchmark = create_auto_benchmark(brand_cfg, client)
    manual = load_yaml(Path(args.manual_benchmark)) if args.manual_benchmark else {}
    if manual:
        benchmark.update(manual)
    site = analyze_site(brand.get('website', ''), args.site_max_pages)
    competitors = normalize_competitors(benchmark.get('competitors', []))
    query_panel = normalize_query_panel(benchmark.get('query_panel', []))
    probes = run_answer_probes(client, brand.get('name', ''), competitors, query_panel, args.probe_limit)
    report = build_report(brand_cfg, benchmark, site, probes, client.audit)
    out = Path(args.output)
    write_outputs(out, report, benchmark, client.audit)
    commit_info = {"committed": False}
    if args.commit_report:
        commit_info = maybe_git_commit(Path(args.repo_root).resolve(), out.resolve(), Path(args.repo_root).resolve() / args.report_subdir, args.commit_message)
    print(json.dumps({"brand": report['brand']['name'], "overall_score": report['overall_score'], "deepseek_api_calls": report['monitoring_summary']['deepseek_api_calls'], "output": str(out.resolve()), "commit": commit_info}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
