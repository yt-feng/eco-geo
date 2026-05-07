#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PAGE_W, _ = A4
LATIN_FONT = "Helvetica"
try:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    CJK_FONT = "STSong-Light"
except Exception:
    CJK_FONT = LATIN_FONT

BRAND_BLUE = colors.HexColor("#1f4e79")
DEEP_NAVY = colors.HexColor("#0f172a")
MID_GRAY = colors.HexColor("#6b7280")
ACCENT_BLUE = colors.HexColor("#2f80ed")
GOOD_GREEN = colors.HexColor("#16a34a")
WARN_ORANGE = colors.HexColor("#f59e0b")
RISK_RED = colors.HexColor("#dc2626")
DIM_ORDER = ["visibility", "inclusion", "cognition", "outcome"]


def has_cjk(value: Any) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", str(value or "")))


def font_for(value: Any) -> str:
    return CJK_FONT if has_cjk(value) else LATIN_FONT


def redact_text(text: str) -> str:
    if not text:
        return ""
    rules = [
        (r"sk-[A-Za-z0-9_\-]{16,}", "[REDACTED_API_KEY]"),
        (r"Bearer\s+[A-Za-z0-9_\.\-]{16,}", "Bearer [REDACTED]"),
        (r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]"),
        (r"\b1[3-9]\d{9}\b", "[REDACTED_PHONE]"),
    ]
    for pattern, repl in rules:
        text = re.sub(pattern, repl, text)
    return text


def safe_text(value: Any, limit: int | None = None) -> str:
    text = "" if value is None else str(value)
    text = redact_text(text).replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if limit and len(text) > limit:
        return text[: limit - 1].rstrip() + "..."
    return text


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def as_num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def clamp(value: Any, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, as_num(value)))


def pct(value: Any) -> str:
    return f"{as_num(value):.1f}%"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


BASE = getSampleStyleSheet()
STYLES = {
    "Title": ParagraphStyle("Title", parent=BASE["Title"], fontName=LATIN_FONT, fontSize=24, leading=30, alignment=TA_LEFT, textColor=DEEP_NAVY, spaceAfter=10),
    "SubTitle": ParagraphStyle("SubTitle", parent=BASE["BodyText"], fontName=LATIN_FONT, fontSize=11, leading=16, textColor=MID_GRAY, spaceAfter=8),
    "H1": ParagraphStyle("H1", parent=BASE["Heading1"], fontName=LATIN_FONT, fontSize=17, leading=22, textColor=BRAND_BLUE, spaceBefore=8, spaceAfter=8),
    "H2": ParagraphStyle("H2", parent=BASE["Heading2"], fontName=LATIN_FONT, fontSize=12, leading=16, textColor=DEEP_NAVY, spaceBefore=6, spaceAfter=6),
    "Body": ParagraphStyle("Body", parent=BASE["BodyText"], fontName=LATIN_FONT, fontSize=9.2, leading=13.2, textColor=DEEP_NAVY, spaceAfter=5),
    "Small": ParagraphStyle("Small", parent=BASE["BodyText"], fontName=LATIN_FONT, fontSize=7.5, leading=10, textColor=DEEP_NAVY),
    "Muted": ParagraphStyle("Muted", parent=BASE["BodyText"], fontName=LATIN_FONT, fontSize=8.2, leading=11.5, textColor=MID_GRAY, spaceAfter=4),
}


def para(value: Any, style: str = "Body") -> Paragraph:
    raw = safe_text(value)
    text = raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    base = STYLES[style]
    chosen = font_for(raw)
    if base.fontName != chosen:
        base = ParagraphStyle(f"{style}_{chosen}", parent=base, fontName=chosen)
    return Paragraph(text, base)


class Bar(Flowable):
    def __init__(self, value: float, width: float = 76 * mm, height: float = 4.2 * mm, color: colors.Color = ACCENT_BLUE):
        super().__init__()
        self.value = clamp(value)
        self.width = width
        self.height = height
        self.color = color

    def wrap(self, avail_width: float, avail_height: float) -> Tuple[float, float]:
        return min(self.width, avail_width), self.height

    def draw(self) -> None:
        self.canv.setFillColor(colors.HexColor("#e5e7eb"))
        self.canv.roundRect(0, 0, self.width, self.height, self.height / 2, stroke=0, fill=1)
        self.canv.setFillColor(self.color)
        self.canv.roundRect(0, 0, self.width * self.value / 100.0, self.height, self.height / 2, stroke=0, fill=1)


def table_style(header_bg: colors.Color = BRAND_BLUE, font_size: float = 7.5) -> TableStyle:
    return TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), LATIN_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])


def metric_grid(metrics: List[Tuple[str, Any, str]]) -> Table:
    cells = []
    for label, value, note in metrics:
        cells.append([
            para(label, "Muted"),
            Paragraph(f"<font name='{LATIN_FONT}' size='17' color='#0f172a'><b>{safe_text(value)}</b></font>", STYLES["Body"]),
            para(note, "Muted"),
        ])
    while len(cells) < 6:
        cells.append([para("", "Muted"), para("", "Muted"), para("", "Muted")])
    t = Table([cells[:3], cells[3:6]], colWidths=[56 * mm, 56 * mm, 56 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dbeafe")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dbeafe")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def bars_table(items: Dict[str, Any], title_a: str, title_b: str, max_rows: int = 12) -> Table:
    pairs = sorted([(safe_text(k, 36), as_num(v)) for k, v in (items or {}).items()], key=lambda x: x[1], reverse=True)[:max_rows]
    mx = max([v for _, v in pairs] or [1])
    rows: List[List[Any]] = [[para(title_a, "Small"), para(title_b, "Small"), para("Distribution", "Small")]]
    for label, value in pairs:
        rows.append([para(label, "Small"), para(str(int(value) if float(value).is_integer() else round(value, 1)), "Small"), Bar(100 * value / mx, width=76 * mm)])
    if len(rows) == 1:
        rows.append([para("No data", "Muted"), para("-", "Muted"), para("-", "Muted")])
    t = Table(rows, colWidths=[55 * mm, 20 * mm, 84 * mm])
    t.setStyle(table_style(header_bg=colors.HexColor("#334155"), font_size=7.2))
    return t


def scorecard(scores: Dict[str, Any]) -> Table:
    labels = {
        "visibility": "Visibility - answer presence",
        "inclusion": "Inclusion - source readiness",
        "cognition": "Cognition - narrative accuracy",
        "outcome": "Outcome - recommendation capture",
    }
    rows = [[para("Dimension", "Small"), para("Score", "Small"), para("Status", "Small"), para("Visual", "Small")]]
    for dim in DIM_ORDER:
        val = clamp(scores.get(dim, 0))
        status = "Healthy" if val >= 70 else "Watch" if val >= 40 else "Risk"
        color = GOOD_GREEN if val >= 70 else WARN_ORANGE if val >= 40 else RISK_RED
        rows.append([para(labels.get(dim, dim), "Small"), para(f"{val:.1f}", "Small"), para(status, "Small"), Bar(val, color=color)])
    t = Table(rows, colWidths=[64 * mm, 22 * mm, 26 * mm, 58 * mm])
    t.setStyle(table_style())
    return t


def bullet_list(items: Iterable[Any], limit: int = 6) -> List[Paragraph]:
    rows = [para("- " + safe_text(x, 240), "Body") for x in list(items or [])[:limit]]
    return rows or [para("- No item available from this run.", "Muted")]


def calls_summary(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    durations = [as_num(c.get("duration_ms")) for c in calls if c.get("duration_ms") is not None]
    tokens = [as_num((c.get("usage") or {}).get("total_tokens")) for c in calls]
    return {
        "total_calls": len(calls),
        "monitoring_calls": len([c for c in calls if c.get("call_type") == "monitoring"]),
        "successful_calls": len([c for c in calls if c.get("success") is True]),
        "failed_calls": len([c for c in calls if c.get("success") is not True]),
        "total_tokens": int(sum(tokens)),
        "avg_duration_sec": round(statistics.mean(durations) / 1000.0, 1) if durations else 0,
        "p95_duration_sec": round(sorted(durations)[int(0.95 * (len(durations) - 1))] / 1000.0, 1) if durations else 0,
        "stages": dict(Counter(str(c.get("stage", "unknown")) for c in calls)),
    }


def representative_rows(calls: List[Dict[str, Any]], limit: int = 14) -> List[List[Any]]:
    rows = [[para("ID", "Small"), para("Stage", "Small"), para("Question", "Small"), para("Visibility", "Small"), para("Takeaway", "Small")]]
    for call in [c for c in calls if c.get("call_type") == "monitoring"][:limit]:
        result = call.get("response_json") or {}
        visibility = f"{result.get('target_brand_mentioned', '')} / {result.get('target_brand_role', '')}"
        takeaway = result.get("summary_takeaway") or result.get("geo_gap_observed") or ""
        rows.append([
            para(call.get("query_id") or call.get("call_id"), "Small"),
            para(call.get("stage", ""), "Small"),
            para(call.get("question", ""), "Small"),
            para(visibility, "Small"),
            para(takeaway, "Small"),
        ])
    return rows


def tex_escape(value: Any) -> str:
    s = safe_text(value)
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(repl.get(ch, ch) for ch in s)


def build_tex(path: Path, brand: str, aggregate: Dict[str, Any], synthesis: Dict[str, Any], summary: Dict[str, Any]) -> None:
    dims = synthesis.get("dimension_scores") or aggregate.get("dimension_scores", {}) or {}
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.72in]{geometry}",
        r"\usepackage{booktabs,longtable,hyperref}",
        r"\title{GEO Monitoring Consulting Report}",
        r"\begin{document}",
        r"\maketitle",
        rf"\section*{{{tex_escape(brand)}}}",
        rf"\textbf{{Generated:}} {tex_escape(now_utc())}\\",
        rf"\textbf{{Total DeepSeek calls:}} {summary.get('total_calls', 0)}\\",
        rf"\textbf{{Monitoring probes:}} {summary.get('monitoring_calls', 0)}\\",
        rf"\textbf{{Brand mention rate:}} {tex_escape(aggregate.get('brand_mention_rate', ''))}\\",
        rf"\textbf{{Brand recommendation rate:}} {tex_escape(aggregate.get('brand_recommendation_rate', ''))}\\",
        r"\section*{Executive Summary}",
        tex_escape(synthesis.get("executive_summary", "")),
        r"\section*{Dimension Scorecard}",
        r"\begin{tabular}{lr}\toprule Dimension & Score\\\midrule",
    ]
    for dim in DIM_ORDER:
        lines.append(rf"{tex_escape(dim.title())} & {as_num(dims.get(dim)):.1f}\\")
    lines.extend([r"\bottomrule\end{tabular}", r"\section*{Top Findings}", r"\begin{itemize}"])
    for item in (synthesis.get("top_findings") or [])[:8]:
        lines.append(rf"\item {tex_escape(item)}")
    lines.extend([r"\end{itemize}", r"\section*{90-Day Action Roadmap}", r"\begin{itemize}"])
    for item in (synthesis.get("priority_actions") or [])[:8]:
        lines.append(rf"\item {tex_escape(item)}")
    lines.extend([r"\end{itemize}", r"\section*{Methodology Note}", tex_escape(synthesis.get("methodology_note", "")), r"\end{document}"])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_pdf(path: Path, brand: str, setup: Dict[str, Any], aggregate: Dict[str, Any], synthesis: Dict[str, Any], calls: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=15 * mm, bottomMargin=14 * mm)
    summary = calls_summary(calls)
    dims = synthesis.get("dimension_scores") or aggregate.get("dimension_scores", {}) or {}
    profile = setup.get("brand_profile", {}) if isinstance(setup, dict) else {}
    story: List[Any] = []

    story.append(para("Generative Engine Optimization", "SubTitle"))
    story.append(para("Consulting-Grade Monitoring Report", "Title"))
    story.append(Paragraph(safe_text(brand), ParagraphStyle("Brand", parent=STYLES["Title"], fontName=font_for(brand), fontSize=19, leading=24, textColor=BRAND_BLUE)))
    story.append(Spacer(1, 8))
    story.append(para(profile.get("brief") or profile.get("positioning_summary") or "Automated GEO monitoring run", "SubTitle"))
    story.append(Spacer(1, 10))
    story.append(metric_grid([
        ("DeepSeek calls", summary["total_calls"], "complete API call ledger"),
        ("Monitoring probes", summary["monitoring_calls"], "query-level answer tests"),
        ("Success rate", pct(100 * summary["successful_calls"] / max(1, summary["total_calls"])), "operational health"),
        ("Mention rate", pct(aggregate.get("brand_mention_rate")), "brand appears in answers"),
        ("Recommendation rate", pct(aggregate.get("brand_recommendation_rate")), "brand is recommended"),
        ("Total tokens", summary["total_tokens"], "prompt + completion usage"),
    ]))
    story.append(Spacer(1, 16))
    story.append(para("Confidential - generated from structured DeepSeek monitoring logs, aggregate metrics, and synthesis outputs. Full raw prompts and responses stay in JSON/JSONL trace files; this PDF shows redacted, executive-safe summaries.", "Muted"))
    story.append(PageBreak())

    story.append(para("1. Executive Summary", "H1"))
    story.append(para(synthesis.get("executive_summary") or "No executive summary was generated for this run."))
    story.append(Spacer(1, 8))
    story.append(scorecard(dims))
    story.append(Spacer(1, 8))
    story.append(para("Management readout", "H2"))
    if as_num(aggregate.get("brand_mention_rate")) > 0 and as_num(aggregate.get("brand_recommendation_rate")) <= 1:
        story.append(para("The brand is not fully invisible, but it is not capturing the recommendation layer. This pattern usually indicates weak first-party evidence, unclear answer-engine positioning, or stronger competitor narratives."))
    else:
        story.append(para("The monitoring run should be read as a baseline for answer-engine visibility, evidence readiness, and competitive recommendation capture."))
    story.append(PageBreak())

    story.append(para("2. Query Universe Coverage", "H1"))
    story.append(para("The monitoring panel is designed to prove coverage, not just produce one-off prompt examples. It spans brand, competitor, industry, category, problem, comparison, use-case, and trust-oriented questions."))
    story.append(bars_table(aggregate.get("query_type_distribution", {}), "Query type", "Count"))
    story.append(Spacer(1, 10))
    story.append(bars_table(aggregate.get("funnel_stage_distribution", {}), "Funnel stage", "Count"))
    story.append(PageBreak())

    story.append(para("3. Competitive Pressure", "H1"))
    story.append(para("Competitor pressure is evaluated from answer appearances, recommendation ownership, and best-answer-owner patterns across monitoring probes."))
    story.append(para("Competitor mention frequency", "H2"))
    story.append(bars_table(aggregate.get("competitor_mention_counts", {}), "Competitor", "Mentions"))
    story.append(Spacer(1, 10))
    story.append(para("Recommended brand frequency", "H2"))
    story.append(bars_table(aggregate.get("recommended_brand_counts", {}), "Recommended brand", "Count"))
    story.append(PageBreak())

    story.append(para("4. Evidence Gaps and Risk", "H1"))
    story.append(metric_grid([
        ("First-party source needed", pct(aggregate.get("first_party_source_needed_rate")), "answers need official support"),
        ("Citation likelihood", round(as_num(aggregate.get("avg_citation_likelihood")), 2), "average observed score"),
        ("Answer confidence", round(as_num(aggregate.get("avg_answer_confidence")), 2), "average observed score"),
        ("Risk flags", aggregate.get("factual_risk_flag_count", 0), "fact gaps detected"),
        ("Failed calls", summary["failed_calls"], "execution reliability"),
        ("Avg duration", f"{summary['avg_duration_sec']}s", "per API call"),
    ]))
    story.append(Spacer(1, 10))
    story.append(para("Risk flag samples", "H2"))
    story.extend(bullet_list(aggregate.get("risk_flag_samples", []), limit=10))
    story.append(PageBreak())

    story.append(para("5. Strategic Findings", "H1"))
    story.extend(bullet_list(synthesis.get("top_findings", []), limit=8))
    story.append(Spacer(1, 10))
    story.append(para("90-Day Action Roadmap", "H1"))
    actions = synthesis.get("priority_actions", []) or []
    fallback = [
        "Build authoritative first-party answer assets: brand fact page, FAQ, pricing/range detail, trust credentials, and comparison pages.",
        "Convert monitoring gaps into content briefs mapped by query type and funnel stage; prioritize competitor-owned recommendation questions.",
        "Re-run monitoring and measure improvement in mention rate, recommendation rate, risk flags, and best-answer-owner share.",
    ]
    rows = [[para("Phase", "Small"), para("Action focus", "Small")]]
    for idx, phase in enumerate(["0-30 days", "31-60 days", "61-90 days"]):
        rows.append([para(phase, "Small"), para(actions[idx] if idx < len(actions) else fallback[idx], "Small")])
    roadmap = Table(rows, colWidths=[30 * mm, 135 * mm])
    roadmap.setStyle(table_style())
    story.append(roadmap)
    story.append(PageBreak())

    story.append(para("6. Operations Ledger", "H1"))
    story.append(para("This section summarizes execution health. Full prompt and response text is intentionally kept in deepseek_calls.json and raw_runs.jsonl, not printed here."))
    story.append(bars_table(summary.get("stages", {}), "Stage", "Calls", max_rows=8))
    story.append(Spacer(1, 8))
    ops = Table([
        [para("Metric", "Small"), para("Value", "Small")],
        [para("Total calls", "Small"), para(summary["total_calls"], "Small")],
        [para("Successful calls", "Small"), para(summary["successful_calls"], "Small")],
        [para("Failed calls", "Small"), para(summary["failed_calls"], "Small")],
        [para("P95 duration", "Small"), para(f"{summary['p95_duration_sec']}s", "Small")],
        [para("Total tokens", "Small"), para(summary["total_tokens"], "Small")],
    ], colWidths=[70 * mm, 40 * mm])
    ops.setStyle(table_style(header_bg=colors.HexColor("#334155")))
    story.append(ops)
    story.append(PageBreak())

    story.append(para("Appendix - Representative Monitoring Results", "H1"))
    rep = Table(representative_rows(calls), colWidths=[15 * mm, 30 * mm, 48 * mm, 30 * mm, 48 * mm], repeatRows=1)
    rep.setStyle(table_style(header_bg=colors.HexColor("#334155"), font_size=6.2))
    story.append(rep)
    story.append(Spacer(1, 8))
    story.append(para("Methodology", "H1"))
    story.append(para(synthesis.get("methodology_note") or "Automated monitoring run using generated query panel, repeated answer-engine probes, aggregate metrics, and executive synthesis."))

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont(font_for(brand), 7)
        canvas.setFillColor(MID_GRAY)
        canvas.drawString(16 * mm, 8 * mm, f"{safe_text(brand, 40)} GEO Monitoring Report")
        canvas.setFont(LATIN_FONT, 7)
        canvas.drawRightString(PAGE_W - 16 * mm, 8 * mm, f"Page {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a consulting-grade GEO PDF report from eco-geo JSON outputs.")
    ap.add_argument("--report-dir", default="dist/report", help="Directory containing deepseek_calls.json, aggregate_metrics.json, synthesis.json, research_setup.json")
    ap.add_argument("--calls", default="", help="Optional explicit deepseek_calls.json path")
    ap.add_argument("--aggregate", default="", help="Optional explicit aggregate_metrics.json path")
    ap.add_argument("--synthesis", default="", help="Optional explicit synthesis.json path")
    ap.add_argument("--setup", default="", help="Optional explicit research_setup.json path")
    ap.add_argument("--output", default="", help="Output PDF path. Defaults to report-dir/executive_report.pdf")
    ap.add_argument("--tex-output", default="", help="Output TeX path. Defaults to report-dir/executive_report.tex")
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    calls_path = Path(args.calls) if args.calls else report_dir / "deepseek_calls.json"
    aggregate_path = Path(args.aggregate) if args.aggregate else report_dir / "aggregate_metrics.json"
    synthesis_path = Path(args.synthesis) if args.synthesis else report_dir / "synthesis.json"
    setup_path = Path(args.setup) if args.setup else report_dir / "research_setup.json"
    out_pdf = Path(args.output) if args.output else report_dir / "executive_report.pdf"
    out_tex = Path(args.tex_output) if args.tex_output else report_dir / "executive_report.tex"

    calls = read_json(calls_path, [])
    aggregate = read_json(aggregate_path, {})
    synthesis = read_json(synthesis_path, {})
    setup = read_json(setup_path, {})
    calls = calls if isinstance(calls, list) else []
    aggregate = aggregate if isinstance(aggregate, dict) else {}
    synthesis = synthesis if isinstance(synthesis, dict) else {}
    setup = setup if isinstance(setup, dict) else {}

    brand = aggregate.get("brand_name") or (setup.get("brand_profile") or {}).get("brand_name") or "Unknown Brand"
    summary = calls_summary(calls)
    build_pdf(out_pdf, str(brand), setup, aggregate, synthesis, calls)
    build_tex(out_tex, str(brand), aggregate, synthesis, summary)
    print(json.dumps({"pdf": str(out_pdf), "tex": str(out_tex), "brand": str(brand)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
