#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urljoin, urlparse, urldefrag

import yaml
import requests
from bs4 import BeautifulSoup

import run_geo_eval_v3 as base

DIMS = ["visibility", "inclusion", "cognition", "outcome"]
STOPWORDS = set("a an and are as at be by can for from has have in into is it its of on or our the their this to we with you your about product products solution solutions technology technologies storage data enterprise more learn page site global company support contact home overview services service news press release".split())


def esc(v: Any) -> str:
    return html.escape(str(v))


def clamp(v: Any, low: float = 0.0, high: float = 100.0) -> float:
    try:
        n = float(v)
    except Exception:
        n = 0.0
    return max(low, min(high, n))


def avg(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 2) if values else 0.0


def metric_bar(label: str, value: Any, tone: str = "default") -> str:
    color = {"default": "#60a5fa", "good": "#34d399", "warn": "#f59e0b", "bad": "#fb7185"}.get(tone, "#60a5fa")
    score = clamp(value)
    return f"<div class='bar'><div><span>{esc(label)}</span><b>{round(score, 1)}</b></div><p><i style='width:{score}%;background:{color}'></i></p></div>"


def svg_bar(items: Dict[str, Any], width: int = 760) -> str:
    pairs = [(str(k), int(clamp(v, 0, 10000))) for k, v in list(items.items())[:14]]
    if not pairs:
        return "<p class='muted'>No data.</p>"
    max_count = max(v for _, v in pairs) or 1
    rows, height = [], 34 + len(pairs) * 28
    for i, (label, value) in enumerate(pairs):
        y = 24 + i * 28
        w = (width - 260) * value / max_count
        rows.append(f"<text x='8' y='{y+14}' fill='#b9c7dd' font-size='12'>{esc(label)}</text><rect x='190' y='{y}' width='{w:.1f}' height='17' rx='7' fill='url(#g)'/><text x='{198+w:.1f}' y='{y+14}' fill='#e8eef9' font-size='12'>{value}</text>")
    return f"<svg viewBox='0 0 {width} {height}' class='chart'><defs><linearGradient id='g' x1='0%' x2='100%'><stop offset='0%' stop-color='#38bdf8'/><stop offset='100%' stop-color='#818cf8'/></linearGradient></defs>{''.join(rows)}</svg>"


def topic_cloud(items: List[Dict[str, Any]]) -> str:
    if not isinstance(items, list) or not items:
        return "<p class='muted'>No topic cloud generated.</p>"
    weights = [clamp(x.get("weight", x.get("count", 1)), 1, 100) for x in items[:60] if isinstance(x, dict)]
    mx = max(weights) if weights else 1
    spans = []
    for x in items[:60]:
        if not isinstance(x, dict):
            continue
        weight = clamp(x.get("weight", x.get("count", 1)), 1, 100)
        size = 12 + int(30 * weight / mx)
        spans.append(f"<span style='font-size:{size}px'>{esc(x.get('term', ''))}</span>")
    return "<div class='cloud'>" + "".join(spans) + "</div>"


def heatmap(items: Dict[str, Any]) -> str:
    cells = []
    for k, v in items.items():
        val = clamp(v)
        color = f"rgba(96,165,250,{0.18 + val/140:.2f})"
        cells.append(f"<div class='heat' style='background:{color}'><small>{esc(k.replace('_',' ').title())}</small><b>{round(val,1)}</b></div>")
    if not cells:
        return "<p class='muted'>No heatmap generated.</p>"
    return "<div class='heatgrid'>" + "".join(cells) + "</div>"


def crawl_site(root_url: str, max_pages: int = 18) -> Dict[str, Any]:
    root_url = (root_url or "").strip()
    if not root_url:
        return {"available": False, "pages_crawled": 0, "topic_cloud": [], "page_rows": [], "signals": {}, "page_type_counts": {}}
    if not root_url.startswith(("http://", "https://")):
        root_url = "https://" + root_url
    root_url = urldefrag(root_url)[0].rstrip("/")
    domain = urlparse(root_url).netloc.lower()
    session = requests.Session()
    session.headers.update({"User-Agent": "eco-geo-snapshot/3.0"})
    queue = [root_url]
    seen, discovered = set(), {root_url}
    pages, errors = [], []
    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            resp = session.get(url, timeout=18)
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.get_text(" ", strip=True)[:160] if soup.title else ""
            headings = [h.get_text(" ", strip=True)[:140] for h in soup.find_all(["h1", "h2", "h3"])[:8]]
            clone = BeautifulSoup(str(soup), "html.parser")
            for tag in clone(["script", "style", "noscript", "svg"]):
                tag.decompose()
            text = re.sub(r"\s+", " ", clone.get_text(" ", strip=True))
            low = (url + " " + title + " " + " ".join(headings) + " " + text[:2000]).lower()
            page_type = "general"
            for name, pattern in {"about": r"about|company|profile|corporate", "product": r"product|ssd|flash|memory|storage|nand", "support": r"support|download|faq|manual", "trust": r"security|privacy|compliance|quality|warranty|sustainability", "comparison": r"compare|comparison|versus|\bvs\b", "pricing": r"pricing|price|quote|where-to-buy", "docs": r"whitepaper|document|spec|technical|developer", "news": r"news|press|release|blog"}.items():
                if re.search(pattern, low):
                    page_type = name
                    break
            json_ld = soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", re.I)})
            pages.append({"url": url, "title": title, "page_type": page_type, "word_count": len(re.findall(r"[A-Za-z][A-Za-z0-9+\-]{2,}", text)), "json_ld": bool(json_ld), "faq_schema": "faqpage" in low, "pricing_signal": bool(re.search(r"pricing|price|quote|where to buy|buy now", low)), "comparison_signal": bool(re.search(r"compare|comparison|versus|\bvs\b", low)), "trust_signal": bool(re.search(r"security|privacy|compliance|quality|certification|warranty|sustainability", low)), "sample": title + " " + " ".join(headings) + " " + text[:2000]})
            for a in soup.select("a[href]"):
                href = urljoin(url, a.get("href", ""))
                parsed = urlparse(href)
                if parsed.scheme in {"http", "https"} and parsed.netloc.lower() == domain and not re.search(r"\.(pdf|png|jpg|jpeg|svg|zip|mp4|webp)$", parsed.path, re.I):
                    link = urldefrag(href)[0].rstrip("/")
                    if link not in seen and link not in discovered and len(discovered) < max_pages * 5:
                        discovered.add(link)
                        queue.append(link)
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})
    terms = Counter()
    for p in pages:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+\-]{2,}", p.get("sample", "")):
            token = token.lower().strip("-+")
            if token and token not in STOPWORDS and len(token) >= 3:
                terms[token] += 1
    return {"available": True, "root_url": root_url, "pages_crawled": len(pages), "page_rows": [{k: v for k, v in p.items() if k != "sample"} for p in pages], "topic_cloud": [{"term": k, "count": v} for k, v in terms.most_common(50)], "page_type_counts": dict(Counter(p["page_type"] for p in pages)), "signals": {"json_ld_pages": sum(1 for p in pages if p.get("json_ld")), "faq_schema_pages": sum(1 for p in pages if p.get("faq_schema")), "pricing_signal_pages": sum(1 for p in pages if p.get("pricing_signal")), "comparison_signal_pages": sum(1 for p in pages if p.get("comparison_signal")), "trust_signal_pages": sum(1 for p in pages if p.get("trust_signal")), "avg_word_count": avg([p.get("word_count", 0) for p in pages])}, "errors": errors}


def render_rich_dashboard(research: Dict[str, Any], output_dir: Path) -> None:
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
    semantic_clusters = keywords.get("semantic_clusters", []) if isinstance(keywords.get("semantic_clusters", []), list) else []
    website = brand.get("website") or research.get("profile", {}).get("official_website", "")
    site_snapshot = crawl_site(website, 18)
    research["site_snapshot"] = site_snapshot
    data_inventory = {"Research passes": len(research.get("deepseek_runs", [])), "Private queries": len(query_panel), "Question families": len(question_families), "Keyword terms": sum(len(v) for v in keywords.get("term_buckets", {}).values() if isinstance(v, list)), "Semantic clusters": len(semantic_clusters), "Competitors": len(competitor_list), "Dashboard metrics": sum(len((geo.get(d, {}) or {}).get("metrics", {})) for d in DIMS), "Website pages scanned": site_snapshot.get("pages_crawled", 0)}
    inventory_cards = "".join(f"<div class='kpi'><small>{esc(k)}</small><h2>{esc(v)}</h2></div>" for k, v in data_inventory.items())
    query_counts = Counter(str(q.get("type", "other")) for q in query_panel if isinstance(q, dict))
    funnel_counts = Counter(str(q.get("funnel_stage", "unknown")) for q in query_panel if isinstance(q, dict))
    comp_rows = "".join(f"<tr><td><strong>{esc(c.get('name',''))}</strong></td><td>{esc(c.get('geo_maturity_stage',''))}</td><td>{esc(c.get('overall_score_estimate',''))}</td><td>{esc((c.get('dimension_scores') or {}).get('visibility',''))}</td><td>{esc((c.get('dimension_scores') or {}).get('inclusion',''))}</td><td>{esc((c.get('dimension_scores') or {}).get('cognition',''))}</td><td>{esc((c.get('dimension_scores') or {}).get('outcome',''))}</td><td>{esc('; '.join([str(x) for x in c.get('evidence_signals', [])[:2]]))}</td></tr>" for c in competitor_list[:10] if isinstance(c, dict))
    dim_cards = ""
    for dim in DIMS:
        item = geo.get(dim, {}) if isinstance(geo.get(dim, {}), dict) else {}
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
        actions = item.get("priority_actions", []) if isinstance(item.get("priority_actions", []), list) else []
        dim_cards += f"<section class='card'><h3>{dim.title()}</h3>{metric_bar('Score', item.get('score', 0))}{''.join(metric_bar(k.replace('_',' ').title(), v) for k, v in list(metrics.items())[:6])}<p class='muted'>{esc(item.get('rationale', ''))}</p><ul>{''.join(f'<li>{esc(a)}</li>' for a in actions[:4])}</ul></section>"
    evidence_cards = "".join(f"<div class='mini'><small>{esc(k.replace('_',' ').title())}</small>{metric_bar('', v, 'good' if clamp(v) >= 70 else 'warn' if clamp(v) >= 45 else 'bad')}</div>" for k, v in final.get("evidence_map", {}).items())
    site_signal_cards = "".join(f"<div class='mini'><small>{esc(k.replace('_',' ').title())}</small><h2>{esc(v)}</h2></div>" for k, v in site_snapshot.get("signals", {}).items())
    site_rows = "".join(f"<tr><td>{esc(p.get('page_type',''))}</td><td>{esc(p.get('title',''))}</td><td>{esc(p.get('word_count',''))}</td><td>{'Y' if p.get('json_ld') else '-'}</td><td>{'Y' if p.get('trust_signal') else '-'}</td><td>{esc(p.get('url',''))}</td></tr>" for p in site_snapshot.get("page_rows", [])[:18])
    journey_cards = "".join(f"<div class='mini'><small>{esc(j.get('stage',''))}</small>{metric_bar('Current', j.get('current_strength',0))}{metric_bar('Peer pressure', j.get('competitor_pressure',0),'bad')}{metric_bar('Opportunity', j.get('opportunity',0),'good')}<p class='muted'>{esc(j.get('notes',''))}</p></div>" for j in final.get("journey_gap_matrix", []) if isinstance(j, dict))
    html_text = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{esc(brand.get('name',''))} GEO Research Dashboard</title><style>body{{margin:0;background:#07111f;color:#e7eefb;font-family:Inter,Segoe UI,Arial,sans-serif}}.wrap{{max-width:1520px;margin:auto;padding:24px}}.hero,.grid2,.grid3,.grid4{{display:grid;gap:18px}}.hero{{grid-template-columns:2fr 1fr}}.grid2{{grid-template-columns:1.15fr .85fr}}.grid3{{grid-template-columns:repeat(3,1fr)}}.grid4{{grid-template-columns:repeat(4,1fr)}}.card,.kpi,.mini{{background:#0f1b2d;border:1px solid #22324a;border-radius:18px;padding:18px;margin-bottom:18px;box-shadow:0 12px 28px rgba(0,0,0,.18)}}.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:16px}}.kpi h2,.mini h2{{font-size:30px;margin:6px 0}}.muted,small{{color:#93a4bb;line-height:1.55}}.bar div{{display:flex;justify-content:space-between;font-size:13px}}.bar p{{height:10px;background:#0b1424;border:1px solid #24344d;border-radius:999px;overflow:hidden}}.bar i{{display:block;height:100%}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{border-bottom:1px solid #22324a;padding:10px;text-align:left;vertical-align:top}}th{{color:#b9c7dd}}.cloud{{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end}}.cloud span{{background:#111f34;border:1px solid #22324a;border-radius:999px;padding:6px 10px}}.chart{{width:100%;height:auto}}@media(max-width:1100px){{.hero,.grid2,.grid3,.grid4,.kpis{{grid-template-columns:1fr}}}}</style></head><body><div class='wrap'><section class='hero'><div class='card'><small>GEO Research Command Center</small><h1>{esc(brand.get('name',''))}</h1><p class='muted'>{esc(brand.get('brief',''))}</p><p>{esc(final.get('executive_summary',''))}</p><div class='kpis'>{inventory_cards}</div></div><div class='card'><h3>Market Pressure</h3>{metric_bar('Peer activation index', pressure.get('peer_activation_index',0), 'warn')}{metric_bar('Urgency score', pressure.get('urgency_score',0), 'bad')}{metric_bar('Gap to leading peer', pressure.get('gap_to_leading_peer',0), 'bad')}{metric_bar('Narrative disadvantage', pressure.get('narrative_disadvantage',0), 'bad')}<p class='muted'>{esc(competitors.get('competitive_narrative',''))}</p></div></section><section class='grid2'><div class='card'><h3>Query Universe Mix</h3>{svg_bar(dict(query_counts))}<p class='muted'>Exact monitoring questions are kept in the internal audit. This view shows the evaluated universe by intent class.</p></div><div class='card'><h3>Funnel Stage Mix</h3>{svg_bar(dict(funnel_counts))}</div></section><section class='grid2'><div class='card'><h3>Research Topic Cloud</h3>{topic_cloud(keywords.get('topic_cloud', []))}</div><div class='card'><h3>Owned-Surface Topic Cloud</h3>{topic_cloud(site_snapshot.get('topic_cloud', []))}</div></section><section class='grid2'><div class='card'><h3>Evidence Map</h3><div class='grid3'>{evidence_cards}</div></div><div class='card'><h3>Owned-Surface Signal Snapshot</h3><div class='grid3'>{site_signal_cards}</div><p class='muted'>Website page scan is included when an official site is supplied or confidently inferred.</p></div></section><section class='card'><h3>Competitor GEO Leaderboard</h3><table><tr><th>Competitor</th><th>Stage</th><th>Overall</th><th>Visibility</th><th>Inclusion</th><th>Cognition</th><th>Outcome</th><th>Evidence Signal</th></tr>{comp_rows}</table></section><section class='grid4'>{dim_cards}</section><section class='grid2'><div class='card'><h3>Journey Gap Matrix</h3><div class='grid3'>{journey_cards}</div></div><div class='card'><h3>Public Page Sample</h3><table><tr><th>Type</th><th>Title</th><th>Words</th><th>JSON-LD</th><th>Trust</th><th>URL</th></tr>{site_rows}</table></div></section></div></body></html>"""
    (output_dir / "dashboard.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rich v3 GEO research pipeline")
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
    base.clamp = clamp
    config = base.load_yaml(repo_root / args.brand_config)
    brand = base.brand_from_args(args, config)
    research = base.run_research(brand, args.model)
    (output_dir / "research_layers.json").write_text(json.dumps(research, ensure_ascii=False, indent=2), encoding="utf-8")
    render_rich_dashboard(research, output_dir)
    base.render_audit(research, output_dir)
    base.make_pdf(output_dir / "dashboard.pdf", f"{brand.get('name')} GEO Research Dashboard", research, internal=False)
    base.make_pdf(output_dir / "internal_audit.pdf", f"{brand.get('name')} Internal GEO Audit", research, internal=True)
    summary = {"brand": brand, "generated_at": research.get("generated_at"), "pipeline_calls": len(research.get("deepseek_runs", [])), "private_query_count": len(research.get("questions", {}).get("query_panel", [])), "question_family_count": len(research.get("questions", {}).get("question_families", [])), "competitor_count": len(research.get("competitors", {}).get("competitors", [])), "website_pages_scanned": research.get("site_snapshot", {}).get("pages_crawled", 0)}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text("# GEO v3 Research Dashboard\n\n" + json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    commit_info = base.commit(repo_root, output_dir, args.report_subdir, args.commit_message) if args.commit_report else {"committed": False}
    print(json.dumps({"brand": brand.get("name"), "pipeline_calls": summary["pipeline_calls"], "private_queries": summary["private_query_count"], "commit": commit_info}, ensure_ascii=False))


if __name__ == "__main__":
    main()
