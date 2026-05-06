from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Dict, List, Tuple

from .theme import load_theme

THEME = load_theme()
PALETTE = THEME["palette"]
BRAND_NAME = THEME["brand_name"]
REPORT_LABEL = THEME.get("report_label", "Deep Research Report")
FONT_FAMILY = THEME.get("font_family", "Trebuchet MS, Aptos, Arial, sans-serif")

PAGE_FORMAT = os.getenv("REPORT_PAGE_FORMAT", "B5").upper()
if PAGE_FORMAT == "A4":
    PAGE_W, PAGE_H = 8.27, 11.69
    PAD_X, PAD_TOP, PAD_BOTTOM = 0.58, 0.56, 0.50
    BASE_FONT = 12.2
    H2_SIZE = 18.4
    LEAD_SIZE = 14.6
else:
    # A compact portrait canvas closer to a client-ready consulting booklet.
    # Matches pdf_renderer.py's default 176mm x 250mm page size.
    PAGE_W, PAGE_H = 6.93, 9.84
    PAD_X, PAD_TOP, PAD_BOTTOM = 0.42, 0.42, 0.38
    BASE_FONT = 10.9
    H2_SIZE = 15.9
    LEAD_SIZE = 12.4

CONTENT_W = PAGE_W - 2 * PAD_X

CSS = f"""
:root {{ --accent:{PALETTE['accent']}; --accent2:{PALETTE.get('bright_blue', PALETTE['accent'])}; --ink:{PALETTE['ink']}; --muted:{PALETTE['subtle']}; --line:{PALETTE['line']}; --paper:{PALETTE['paper']}; --bg:{PALETTE['panel']}; --lightblue:{PALETTE.get('light_blue_fill', '#EBF5FF')}; }}
* {{ box-sizing:border-box; }}
html, body {{ margin:0; padding:0; }}
body {{ font-family:{FONT_FAMILY}; background:#fff; color:var(--ink); font-size:{BASE_FONT}px; line-height:1.42; }}
.page {{ width:{PAGE_W}in; height:{PAGE_H}in; margin:0 auto; background:var(--paper); position:relative; padding:{PAD_TOP}in {PAD_X}in {PAD_BOTTOM}in {PAD_X}in; page-break-after:always; overflow:hidden; }}
.cover {{ padding:0; background-size:cover; background-position:center; color:white; }}
.cover::after {{ content:""; position:absolute; inset:0; background:linear-gradient(90deg, rgba(5,28,44,.90) 0%, rgba(5,28,44,.68) 44%, rgba(5,28,44,.10) 100%); }}
.cover-panel {{ position:absolute; left:.42in; top:.56in; width:{min(4.90, CONTENT_W - 0.35):.2f}in; min-height:2.85in; background:rgba(255,255,255,.96); color:var(--ink); padding:.26in .30in; z-index:2; border-top:.06in solid var(--accent2); }}
.cover-panel .eyebrow {{ font-size:8pt; color:var(--accent); font-weight:bold; letter-spacing:.04em; text-transform:uppercase; }}
.cover-panel h1 {{ font-size:{22 if PAGE_FORMAT != 'A4' else 25}pt; line-height:1.08; font-weight:400; margin:.18in 0 .17in; }}
.cover-date {{ font-size:8pt; color:#555; font-weight:bold; }}
.logo-fixed {{ position:absolute; top:.17in; right:.34in; width:.58in; z-index:10; }}
.page-header {{ position:absolute; top:.16in; left:{PAD_X}in; right:1.05in; height:.18in; color:#9aa0a6; font-size:5.8pt; text-transform:uppercase; letter-spacing:.04em; }}
.page-footer {{ position:absolute; bottom:.13in; left:{PAD_X}in; right:{PAD_X}in; display:flex; justify-content:space-between; color:#a7adb3; font-size:5.8pt; white-space:nowrap; }}
.kicker {{ color:var(--accent); font-size:7.0pt; font-weight:bold; letter-spacing:.08em; text-transform:uppercase; margin-bottom:.06in; }}
h1, h2, h3 {{ margin:0; }}
h1 {{ font-size:25pt; line-height:1.08; font-weight:400; color:var(--ink); }}
h2 {{ font-size:{H2_SIZE}pt; line-height:1.14; font-weight:400; color:var(--ink); margin-bottom:.12in; }}
h3 {{ font-size:11.5pt; line-height:1.16; color:var(--accent); margin:.06in 0 .08in; }}
.lead {{ font-size:{LEAD_SIZE}pt; line-height:1.22; color:var(--accent); font-weight:400; margin:.08in 0 .14in; max-width:{CONTENT_W:.2f}in; }}
.two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:.22in; }}
p {{ margin:0 0 .075in; }}
ul {{ margin:.02in 0 .05in .15in; padding:0; }}
li {{ margin-bottom:.035in; }}
.contents-list {{ margin-top:.18in; font-size:10pt; line-height:1.42; }}
.contents-list li {{ margin-bottom:.075in; }}
.highlight-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:.10in .13in; margin-top:.14in; }}
.highlight-card {{ border-left:3px solid var(--accent); background:#fff; padding:.065in .08in; min-height:.58in; box-shadow:0 0 0 1px var(--line); }}
.highlight-card .num {{ color:var(--accent); font-size:8.5pt; font-weight:bold; margin-bottom:.025in; }}
.highlight-card .text {{ color:var(--ink); font-size:7.8pt; line-height:1.22; }}
.hero-image {{ width:100%; height:2.65in; object-fit:cover; display:block; margin:-.02in 0 .18in; }}
.inline-image {{ width:100%; height:2.15in; object-fit:cover; display:block; margin:.05in 0 .10in; }}
.chart-inline {{ width:100%; max-height:{3.15 if PAGE_FORMAT != 'A4' else 3.80}in; object-fit:contain; border:none; margin:.08in 0 .08in; page-break-inside:avoid; }}
.chart-page-img {{ width:100%; max-height:{6.35 if PAGE_FORMAT != 'A4' else 7.80}in; object-fit:contain; display:block; margin:.06in 0 .08in; }}
.takeaway {{ border-left:3px solid var(--accent); background:#f7fbfd; padding:.075in .095in; margin:.10in 0 .09in; page-break-inside:avoid; font-size:7.8pt; line-height:1.22; }}
.takeaway strong {{ color:var(--ink); display:block; margin-bottom:.025in; }}
.reference-note {{ color:var(--muted); font-size:6.3pt; border-top:1px solid var(--line); padding-top:.07in; margin-top:.10in; }}
.disclaimer-text, .small-note {{ color:var(--muted); font-size:8.4pt; line-height:1.42; max-width:{CONTENT_W:.2f}in; }}
.section-note {{ color:var(--muted); font-size:7.2pt; margin:.03in 0 .08in; }}
@media print {{ body {{ background:#fff; }} .page {{ margin:0; box-shadow:none; }} }}
"""

LABELS = {
    "zh": {"lang": "zh-CN", "hero": REPORT_LABEL, "topic": "选题", "prepared_by": "出品方", "summary": "执行摘要", "toc": "目录", "disclaimer": "免责声明", "takeaways": "本页要点", "charts": "Exhibit", "reference_note": "本报告参考了以下机构或平台的公开研究与数据资料：", "formal_note": "完整底稿与来源备份已归档于 backup 文件夹。", "disclaimer_text": "本文件为管理咨询与研究分析材料，仅供战略讨论、行业研判与管理决策参考，不构成投资建议、证券建议、法律意见、税务意见或审计意见。"},
    "en": {"lang": "en", "hero": REPORT_LABEL, "topic": "Topic", "prepared_by": "Prepared by", "summary": "Key Highlights", "toc": "Contents", "disclaimer": "Disclaimer", "takeaways": "Takeaways", "charts": "Exhibit", "reference_note": "This report was informed by public research and data from:", "formal_note": "The full source backup is archived in the backup folder.", "disclaimer_text": "This document is a management consulting and research analysis deliverable for strategy discussion only and does not constitute investment, legal, tax, or audit advice."},
}


def _labels(language: str) -> Dict[str, str]:
    return LABELS["en"] if str(language).lower().startswith("en") else LABELS["zh"]


def _page_header(parts: List[str], logo_path: str, page_no: int) -> None:
    if logo_path:
        parts.append(f"<img class='logo-fixed' src='{html.escape(logo_path)}' alt='brand logo' />")
    parts.append(f"<div class='page-header'>{html.escape(BRAND_NAME)} | CONFIDENTIAL</div>")
    parts.append(f"<div class='page-footer'><span>{html.escape(BRAND_NAME)} | Confidential</span><span>{page_no}</span></div>")


def _clean_summary_item(item: str) -> str:
    item = str(item or "").strip()
    if "：" in item:
        head, rest = item.split("：", 1)
        if rest.strip().startswith(head.strip()):
            item = head + "：" + rest.strip()[len(head.strip()):].lstrip("：: ，,。")
    return item


def _chart_keys(assets: Dict[str, str]) -> List[str]:
    def sort_key(k: str) -> Tuple[int, str]:
        try:
            return int(k.split("-", 1)[1]), k
        except Exception:
            return 999, k
    return sorted([k for k in assets if k.startswith("chart-")], key=sort_key)


def _resolve_visual(section: Dict, section_idx: int, assets: Dict[str, str], chart_ids: List[str], used_charts: set[str]) -> str:
    hint = str(section.get("visual_hint", "") or "")
    if hint in assets:
        return hint
    image_key = f"image-{section_idx}"
    if image_key in assets:
        return image_key
    for chart_key in chart_ids:
        if chart_key not in used_charts:
            return chart_key
    return ""


def _next_unused_chart(chart_ids: List[str], used_charts: set[str]) -> str:
    for chart_key in chart_ids:
        if chart_key not in used_charts:
            return chart_key
    return ""


def _render_chart_page(parts: List[str], chart_key: str, chart_path: str, labels: Dict[str, str], logo_path: str, page_no: int) -> int:
    parts.append("<section class='page content-page'>")
    _page_header(parts, logo_path, page_no)
    exhibit_no = chart_key.replace("chart-", "")
    parts.append(f"<div class='kicker'>{html.escape(labels['charts'])} {html.escape(exhibit_no)}</div>")
    parts.append(f"<img class='chart-page-img' src='{html.escape(chart_path)}' alt='{html.escape(chart_key)}' />")
    parts.append("</section>")
    return page_no + 1


def render_report_html(report: Dict, assets: Dict[str, str], output_file: Path, topic: str, language: str = "zh") -> Path:
    labels = _labels(language)
    sections = report.get("sections", [])
    institutions = report.get("reference_institutions", [])
    logo_path = assets.get("brand-logo", "")
    cover_path = assets.get("cover-background", "")
    title = report.get("report_title", topic)
    page_no = 1
    chart_ids = _chart_keys(assets)
    used_charts: set[str] = set()

    parts: List[str] = ["<!DOCTYPE html>", f"<html lang='{labels['lang']}'>", "<head>", "<meta charset='utf-8' />", "<meta name='viewport' content='width=device-width, initial-scale=1' />", f"<title>{html.escape(title)}</title>", f"<style>{CSS}</style>", "</head>", "<body>"]

    parts.append(f"<section class='page cover' style=\"background-image:url('{html.escape(cover_path)}');\"><div class='cover-panel'><div class='eyebrow'>{html.escape(BRAND_NAME)}</div><div class='eyebrow'>{html.escape(labels['hero'])}</div><h1>{html.escape(title)}</h1><div class='cover-date'>{html.escape(topic)}</div></div></section>")
    page_no += 1

    parts.append("<section class='page content-page'>")
    _page_header(parts, logo_path, page_no)
    parts.append(f"<div class='kicker'>{html.escape(labels['summary'])}</div><h2>The analysis points to a focused set of management priorities</h2>")
    summary = [_clean_summary_item(x) for x in report.get("executive_summary", [])[:8]]
    parts.append("<div class='highlight-grid'>")
    for idx, item in enumerate(summary, start=1):
        parts.append(f"<div class='highlight-card'><div class='num'>{idx:02d}</div><div class='text'>{html.escape(item)}</div></div>")
    parts.append("</div></section>")
    page_no += 1

    parts.append("<section class='page content-page'>")
    _page_header(parts, logo_path, page_no)
    parts.append(f"<div class='kicker'>{html.escape(labels['toc'])}</div><h2>{html.escape(labels['toc'])}</h2><ol class='contents-list'>")
    for section in sections:
        parts.append(f"<li>{html.escape(section.get('title', 'Section'))}</li>")
    parts.append("</ol></section>")
    page_no += 1

    parts.append("<section class='page content-page'>")
    _page_header(parts, logo_path, page_no)
    parts.append(f"<div class='kicker'>{html.escape(labels['disclaimer'])}</div><h2>{html.escape(labels['disclaimer'])}</h2><p class='disclaimer-text'>{html.escape(labels['disclaimer_text'])}</p>")
    if institutions:
        parts.append(f"<p class='small-note'>{html.escape(labels['reference_note'])} {html.escape(', '.join(institutions))}.</p>")
    parts.append(f"<p class='small-note'>{html.escape(labels['formal_note'])}</p></section>")
    page_no += 1

    for idx, section in enumerate(sections, start=1):
        paragraphs = section.get("paragraphs", [])
        visual_key = _resolve_visual(section, idx, assets, chart_ids, used_charts)
        is_chart = visual_key.startswith("chart-")
        if is_chart:
            used_charts.add(visual_key)

        first = paragraphs[:4]
        rest = paragraphs[4:]
        if rest and len(rest) <= 1:
            first += rest
            rest = []

        parts.append("<section class='page content-page'>")
        _page_header(parts, logo_path, page_no)
        parts.append(f"<h2>{html.escape(section.get('title', 'Section'))}</h2>")
        lead = section.get("lead", "")
        if lead:
            parts.append(f"<div class='lead'>{html.escape(lead)}</div>")

        if visual_key and not is_chart and visual_key in assets:
            parts.append(f"<img class='hero-image' src='{html.escape(assets[visual_key])}' alt='{html.escape(visual_key)}' />")

        parts.append("<div class='two-col'><div>")
        for p in first[:2]:
            parts.append(f"<p>{html.escape(p)}</p>")
        takeaways = section.get("key_takeaways", [])[:3]
        if takeaways:
            parts.append(f"<div class='takeaway'><strong>{html.escape(labels['takeaways'])}</strong><ul>")
            for item in takeaways:
                parts.append(f"<li>{html.escape(item)}</li>")
            parts.append("</ul></div>")
        parts.append("</div><div>")
        for p in first[2:4]:
            parts.append(f"<p>{html.escape(p)}</p>")
        parts.append("</div></div>")
        if is_chart and visual_key in assets:
            parts.append(f"<img class='chart-inline' src='{html.escape(assets[visual_key])}' alt='{html.escape(visual_key)}' />")
        parts.append("</section>")
        page_no += 1

        # If the section used a photograph, interleave the next data exhibit right after it.
        if not is_chart:
            chart_key = _next_unused_chart(chart_ids, used_charts)
            if chart_key:
                used_charts.add(chart_key)
                page_no = _render_chart_page(parts, chart_key, assets[chart_key], labels, logo_path, page_no)

        if rest:
            parts.append("<section class='page content-page'>")
            _page_header(parts, logo_path, page_no)
            parts.append(f"<h2>{html.escape(section.get('title', 'Section'))}</h2><div class='section-note'>Evidence and implications</div>")
            parts.append("<div class='two-col'><div>")
            midpoint = (len(rest) + 1) // 2
            for p in rest[:midpoint]:
                parts.append(f"<p>{html.escape(p)}</p>")
            parts.append("</div><div>")
            for p in rest[midpoint:]:
                parts.append(f"<p>{html.escape(p)}</p>")
            parts.append("</div></div></section>")
            page_no += 1

    # Only include genuinely unused exhibits; this avoids the old end-of-report duplicate exhibit dump.
    for chart_key in chart_ids:
        if chart_key not in used_charts:
            used_charts.add(chart_key)
            page_no = _render_chart_page(parts, chart_key, assets[chart_key], labels, logo_path, page_no)

    if institutions:
        parts.append("<section class='page content-page'>")
        _page_header(parts, logo_path, page_no)
        parts.append(f"<div class='reference-note'>{html.escape(labels['reference_note'])} {html.escape(', '.join(institutions))}. {html.escape(labels['formal_note'])}</div>")
        parts.append("</section>")

    parts.append("</body></html>")
    output_file.write_text("\n".join(parts), encoding="utf-8")
    return output_file


def render_report_markdown(report: Dict, assets: Dict[str, str], output_file: Path, topic: str, language: str = "zh") -> Path:
    labels = _labels(language)
    institutions = report.get("reference_institutions", [])
    lines: List[str] = [f"# {report.get('report_title', topic)}", "", f"**{labels['prepared_by']}**: {BRAND_NAME}", "", f"**{labels['topic']}**: {topic}", ""]
    lines.extend([f"## {labels['summary']}", ""])
    for item in report.get("executive_summary", []):
        lines.append(f"- {_clean_summary_item(item)}")
    lines.extend(["", f"## {labels['toc']}", ""])
    for section in report.get("sections", []):
        lines.append(f"- {section.get('title', 'Section')}")
    lines.extend(["", f"## {labels['disclaimer']}", "", labels['disclaimer_text'], ""])
    for section in report.get("sections", []):
        lines.extend([f"## {section.get('title', 'Section')}", ""])
        if section.get("lead"):
            lines.extend([f"> {section.get('lead')}", ""])
        for paragraph in section.get("paragraphs", []):
            lines.extend([paragraph, ""])
        if section.get("key_takeaways"):
            lines.extend([f"**{labels['takeaways']}**", ""])
            for item in section.get("key_takeaways", []):
                lines.append(f"- {item}")
            lines.append("")
    if institutions:
        lines.extend([f"> {labels['reference_note']} {', '.join(institutions)}.", "", f"> {labels['formal_note']}", ""])
    output_file.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output_file
