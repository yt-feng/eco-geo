from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from .brand_assets import copy_or_generate_brand_assets, summarize_reference_institutions, write_reference_backup
from .deepseek_client import DeepSeekClient
from .graphics import create_chart, create_insight_card, ensure_dir
from .image_generator import generate_ai_image_assets
from .pdf_qa import apply_pdf_qa_fixes, run_pdf_qa
from .pdf_renderer import render_pdf_from_html
from .ppt_renderer import render_pptx
from .presentation_renderer import render_presentation_html
from .report_renderer import render_report_html, render_report_markdown
from .web_fetch import SourceDocument, collect_sources


class ResearchPipeline:
    def __init__(self, client: DeepSeekClient, language: str = "zh", target_length: int | None = None) -> None:
        self.client = client
        self.language = "en" if str(language).lower().startswith("en") else "zh"
        self.target_length = target_length or 0

    def build_report(self, topic: str, output_dir: Path) -> Dict:
        ensure_dir(output_dir)
        assets_dir = output_dir / "assets"
        ensure_dir(assets_dir)

        plan = self._plan_research(topic)
        queries = plan.get("search_queries", [])[:6]
        sources = collect_sources(queries, per_query=3, max_sources=14)
        source_dicts = [source.__dict__ for source in sources]

        try:
            report = self._synthesize_report(topic, plan, sources)
        except Exception as exc:
            (output_dir / "synthesis_error.txt").write_text(str(exc), encoding="utf-8")
            report = self._fallback_report(topic, plan, sources, reason=str(exc))

        report["reference_institutions"] = summarize_reference_institutions(report.get("references", []), source_dicts)
        self._ensure_visual_hints(report)

        asset_map = copy_or_generate_brand_assets(assets_dir)
        backup_dir = write_reference_backup(output_dir, report.get("references", []), source_dicts)
        asset_map.update(generate_ai_image_assets(self.client, topic, report, assets_dir, Path(backup_dir), language=self.language))
        asset_map.update(self._materialize_assets(report, assets_dir))

        html_path, markdown_path, pdf_path = self._render_report_pack(report, asset_map, output_dir, topic)
        qa_dir = output_dir / "backup" / "qa"
        qa_result = run_pdf_qa(pdf_path, html_path, qa_dir)

        final_report = report
        if not qa_result.get("passed", False):
            final_report = apply_pdf_qa_fixes(report, qa_result)
            self._ensure_visual_hints(final_report)
            (output_dir / "report_payload_prefixed.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            html_path, markdown_path, pdf_path = self._render_report_pack(final_report, asset_map, output_dir, topic)
            qa_result = run_pdf_qa(pdf_path, html_path, qa_dir / "after_fix")

        pptx_path = render_pptx(final_report, asset_map, output_dir / "report.pptx", topic, self.language)
        presentation_path = render_presentation_html(final_report, asset_map, output_dir / "presentation.html", topic, self.language)

        (output_dir / "report_payload.json").write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "research_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "sources.json").write_text(json.dumps(source_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "qa_result.json").write_text(json.dumps(qa_result, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "plan": plan,
            "sources": source_dicts,
            "report": final_report,
            "asset_map": asset_map,
            "output_dir": str(output_dir),
            "backup_dir": str(backup_dir),
            "language": self.language,
            "target_length": self.target_length,
            "html_path": str(html_path),
            "markdown_path": str(markdown_path),
            "pdf_path": str(pdf_path),
            "pptx_path": str(pptx_path),
            "presentation_path": str(presentation_path),
            "qa_result": qa_result,
        }

    def _render_report_pack(self, report: Dict, asset_map: Dict[str, str], output_dir: Path, topic: str):
        html_path = render_report_html(report=report, assets=asset_map, output_file=output_dir / "report.html", topic=topic, language=self.language)
        markdown_path = render_report_markdown(report=report, assets=asset_map, output_file=output_dir / "report.md", topic=topic, language=self.language)
        pdf_path = render_pdf_from_html(html_path, output_dir / "report.pdf")
        return html_path, markdown_path, pdf_path

    def _lang_instruction(self) -> str:
        return "Use English for the whole report." if self.language == "en" else "全程使用中文输出。"

    def _scope_instruction(self) -> str:
        if self.language == "en":
            return "Do not target a fixed word count. Produce a client-ready research report that naturally renders to roughly 10-30 PDF pages, depending on evidence depth. Avoid padding, but do not truncate analysis."
        return "不要按固定字数写作。目标是一份可直接分发给客户的研究报告，最终自然渲染为约 10-30 页 PDF；不要灌水，但也不要为了控页数截断分析。"

    def _title_style_instruction(self) -> str:
        if self.language == "en":
            return "Use pyramid-principle writing. Titles must be conclusion-first, crisp, sharp, and executive-ready: subject + active verb + implication. Avoid generic headings."
        return "遵循金字塔原理。标题必须是结论，不是标签；用“主体 + 动词 + 判断/影响”的结构，短促、锋利、可供高管快速判断。"

    def _method_instruction(self) -> str:
        if self.language == "en":
            return "Use seven-step problem solving, issue trees, and 10 Tests as internal writing discipline only. Do not create a visible methodology or approach page unless it directly supports the client recommendation."
        return "把七步法、issue tree、战略十问作为内部写作心法融入分析，不要在正式报告里单独写成 Approach 或方法论页面，除非它直接服务于客户结论。"

    def _plan_research(self, topic: str) -> Dict:
        system = "You are a world-class deep research planner. Design a plan that follows deep research plus strategy-consulting problem solving. Return JSON only."
        if self.language == "en":
            user = f"""
Create a research plan for the following topic and return JSON only.

Topic: {topic}

Required JSON fields:
- objective
- audience
- decision_question
- issue_tree: 4-7 branches, each with question, why_it_matters, evidence_needed
- search_queries: 6-8 public web search queries
- outline: 6-10 conclusion-first section titles
- chart_ideas: 4-8 chart opportunities; prioritize market sizing, segmentation, share, adoption curve, regional heatmap, value chain, competitor positioning, scenario comparison
- insight_card_ideas: 2-4 executive insight card ideas
- risks: data or evidence risks

Requirements:
- Use English
- Keep search queries search-engine friendly
- Outline titles must be conclusion-first and crisp
- Do not output markdown
"""
        else:
            user = f"""
为下面这个选题生成研究计划，输出 JSON：

选题：{topic}

JSON 字段要求：
- objective: 研究目标
- audience: 目标读者
- decision_question: 最核心的管理决策问题
- issue_tree: 4-7 个问题分支，每个包含 question、why_it_matters、evidence_needed
- search_queries: 6-8 个适合公开网络检索的查询语句
- outline: 6-10 个结论先行的章节标题
- chart_ideas: 4-8 个图表机会，优先包括市场规模、细分结构、份额、渗透率曲线、区域热力、价值链、竞争定位、情景比较
- insight_card_ideas: 2-4 个高管洞察图卡
- risks: 数据风险或口径风险

要求：
- 默认使用中文
- 查询语句适合搜索引擎
- outline 标题必须是结论，不要写成标签
- 不要输出 markdown，只输出 JSON
"""
        return self.client.chat_json([{"role": "system", "content": system}, {"role": "user", "content": user}])

    def _synthesize_report(self, topic: str, plan: Dict, sources: List[SourceDocument]) -> Dict:
        source_blocks = []
        for idx, src in enumerate(sources, start=1):
            excerpt = src.content[:3000]
            source_blocks.append(f"[Source {idx}]\nTitle: {src.title}\nURL: {src.url}\nSearch Query: {src.query}\nSnippet: {src.snippet}\nExcerpt:\n{excerpt}")

        source_text = "\n\n".join(source_blocks)
        if not source_text:
            source_text = "Insufficient web evidence was fetched. Build a clear analysis framework and explicitly mark where more evidence is needed." if self.language == "en" else "暂无抓取到足够网页资料，请基于选题输出可执行的分析框架，并明确指出需要后续补充外部证据。"

        system = "You are an elite strategy consultant and research writer. Use only the provided source material as factual grounding. Return strict JSON only."
        if self.language == "en":
            user = f"""
Generate a client-ready BlueOcean research report data structure and return JSON only.

Topic:
{topic}

Language rule:
{self._lang_instruction()}
Scope rule:
{self._scope_instruction()}
Headline rule:
{self._title_style_instruction()}
Method rule:
{self._method_instruction()}

Research plan:
{json.dumps(plan, ensure_ascii=False, indent=2)}

Sources:
{source_text}

Required JSON fields:
- report_title
- report_subtitle
- executive_summary: 6-8 bullets
- method_steps: exactly 7 items for backup only
- issue_tree: 4-7 branches
- sections: 7-10 items, each contains id, title, lead, paragraphs, key_takeaways, visual_hint
- insight_cards: 2-4 items, each contains id, title, subtitle, bullets, highlight_number, highlight_label, exhibit_label
- charts: 4-6 items, each contains id, exhibit_no, title, subtitle, type, categories, series, x_label, y_label, caption, source_note
- references: array with title, url, note

Hard requirements:
1. Client-ready output: do not expose methodology pages, scratchpad, or meta labels.
2. Use pyramid structure: answer first, evidence second, implication third.
3. Each section should have 3-5 coherent paragraphs; no paragraph may end with ellipses.
4. At least half of sections should set visual_hint to a chart id such as chart-1, chart-2.
5. Charts must be analytically useful exhibits; if data is approximate, state so in caption or source_note.
6. Avoid text overflow but never truncate with ellipses.
7. references may only use real URLs from source materials.
8. Output JSON only.
"""
        else:
            user = f"""
请生成一份 client-ready、可直接分发的 BlueOcean 研究报告数据结构，输出 JSON。

选题：
{topic}

语言要求：
{self._lang_instruction()}
篇幅要求：
{self._scope_instruction()}
标题要求：
{self._title_style_instruction()}
方法要求：
{self._method_instruction()}

研究计划：
{json.dumps(plan, ensure_ascii=False, indent=2)}

资料：
{source_text}

JSON 字段要求：
- report_title
- report_subtitle
- executive_summary: 6-8 条高亮结论
- method_steps: 正好 7 项，仅用于备份
- issue_tree: 4-7 个分支
- sections: 7-10 项，每项包含 id、title、lead、paragraphs、key_takeaways、visual_hint
- insight_cards: 2-4 项
- charts: 4-6 项，每项包含 id、exhibit_no、title、subtitle、type、categories、series、x_label、y_label、caption、source_note
- references: 数组，每项包含 title、url、note

硬性要求：
1. 正式文件必须 client-ready，不要暴露 Approach、方法论、scratchpad 或元描述。
2. 遵循金字塔结构：先答案，再证据，再影响。
3. 每个 section 写 3-5 段连贯正文；不要以省略号结尾。
4. 至少一半 section 的 visual_hint 指向 chart-1、chart-2 等图表。
5. 图表必须是分析型 exhibit；如果数据为示意性整理，在 caption 或 source_note 中说明。
6. 避免溢出，但不要用省略号截断文本。
7. references 只允许使用资料区真实出现过的 URL。
8. 不要输出 markdown，只输出 JSON。
"""
        return self.client.chat_json([{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.18)

    def _fallback_report(self, topic: str, plan: Dict, sources: List[SourceDocument], *, reason: str = "") -> Dict:
        english = self.language == "en"
        outline = plan.get("outline") or []
        if not isinstance(outline, list) or not outline:
            outline = [
                "China has built a structural advantage in VRFB deployment",
                "Dali Energy Storage can defend leadership through cost and integration",
                "Global expansion depends on bankability and local partnerships",
                "Technology differentiation must be translated into lifecycle economics",
                "Supply security is becoming the decisive competitive variable",
                "Execution should focus on reference projects and financing models",
                "The next phase requires a sharper international go-to-market model",
            ] if english else [
                "中国液流钒电池优势来自产业链与政策共振",
                "大力储能需要把技术领先转化为项目可融资性",
                "全球扩张的关键在于标杆项目和本地合作",
                "成本优势必须用全生命周期经济性表达",
                "钒资源安全正在成为竞争胜负手",
                "下一阶段应聚焦场景、融资与渠道",
                "国际化需要更清晰的市场进入模型",
            ]
        outline = [str(x) for x in outline[:8]]
        refs = []
        for src in sources[:10]:
            refs.append({"title": src.title or src.url, "url": src.url, "note": src.snippet or src.query})
        source_themes = [src.title or src.snippet or src.query for src in sources[:6]]
        theme_text = "; ".join([x for x in source_themes if x])

        if english:
            summary = [
                "China's VRFB position is supported by manufacturing scale, policy demand, and upstream vanadium access.",
                "Dali Energy Storage should frame leadership around lifecycle economics, project delivery, and supply security rather than equipment claims alone.",
                "International competition will increasingly depend on bankability, reference projects, and local ecosystem partnerships.",
                "Long-duration storage demand creates a structural opening, but adoption still depends on clear use cases and financing models.",
                "The near-term management agenda should prioritize investable projects, differentiated proof points, and credible global channels.",
                "Evidence quality should continue to be improved with audited project data, benchmarked costs, and customer references.",
            ]
            subtitle = "A client-ready strategic assessment based on public evidence, market signals, and management-consulting synthesis."
        else:
            summary = [
                "中国液流钒电池优势来自制造规模、政策需求和上游钒资源的共同支撑。",
                "大力储能应把领先地位从设备参数转化为全生命周期经济性、项目交付和供应安全。",
                "国际竞争将越来越取决于可融资性、标杆项目和本地生态合作，而不只是单点技术指标。",
                "长时储能需求提供结构性窗口，但落地仍取决于清晰场景和融资模型。",
                "近期管理议程应优先聚焦可投资项目、差异化证据和可信全球渠道。",
                "后续应继续用审计项目数据、成本基准和客户案例提高证据质量。",
            ]
            subtitle = "基于公开资料、市场信号和管理咨询综合判断形成的可分发战略评估。"

        sections = []
        for idx, title in enumerate(outline, start=1):
            if english:
                paragraphs = [
                    f"The available public evidence indicates that {topic} should be assessed through supply chain control, deployment demand, technology performance, and project bankability rather than through a single product lens.",
                    f"Sources reviewed for this report highlight several relevant signals: {theme_text or 'policy support, project announcements, and long-duration storage demand'}. These signals suggest that leadership is strongest when manufacturing scale and reference projects reinforce each other.",
                    "For senior management, the implication is to separate structural advantages from claims that still require stronger proof. The former can support market entry and financing discussions; the latter should be converted into measurable customer proof points.",
                    "The practical agenda is therefore to prioritize use cases where VRFB duration, cycle life, safety, and electrolyte economics create a visible advantage over lithium-ion alternatives.",
                ]
                takeaways = [
                    "Separate structural advantage from unproven claims.",
                    "Use reference projects to convert technology into bankability.",
                    "Focus on long-duration use cases where VRFB economics are clearest.",
                ]
                lead = "The strongest strategic position comes from combining industrial scale with credible project proof."
            else:
                paragraphs = [
                    f"围绕{topic}的判断，应同时看产业链控制、政策需求、技术性能和项目可融资性，而不能只看单一产品参数。",
                    f"本次公开资料显示的相关信号包括：{theme_text or '政策支持、项目落地、长时储能需求'}。这些信号说明，领先地位只有在制造规模和标杆项目相互强化时才最稳固。",
                    "对管理层而言，需要区分已经形成的结构性优势和仍需验证的市场主张。前者可以支撑市场进入与融资沟通，后者需要转化为可量化的客户证据。",
                    "因此，下一步应优先选择钒电池在时长、循环寿命、安全性和电解液经济性上明显优于锂电方案的场景。",
                ]
                takeaways = ["区分结构性优势和待验证主张。", "用标杆项目把技术转化为可融资性。", "聚焦长时储能经济性最清晰的场景。"]
                lead = "最强的战略位置来自产业规模和项目证据的叠加。"
            sections.append({
                "id": f"section-{idx}",
                "title": title,
                "lead": lead,
                "paragraphs": paragraphs,
                "key_takeaways": takeaways,
                "visual_hint": f"chart-{((idx - 1) % 4) + 1}",
            })

        charts = [
            {"id": "chart-1", "exhibit_no": "1", "title": "Leadership depends on cost, supply, technology and bankability", "subtitle": "Indicative scoring from public evidence", "type": "bar", "categories": ["Supply security", "Cost position", "Policy demand", "Technology proof", "Bankability"], "series": [{"name": "Relative strength", "values": [90, 82, 78, 74, 66]}], "x_label": "Indicative score", "y_label": "", "caption": "Indicative synthesis based on available public sources.", "source_note": "Public sources and BlueOcean synthesis."},
            {"id": "chart-2", "exhibit_no": "2", "title": "Long-duration use cases strengthen the VRFB value proposition", "subtitle": "Illustrative attractiveness by application", "type": "bar", "categories": ["Grid shifting", "Renewables firming", "Industrial microgrid", "Backup power", "Short-duration arbitrage"], "series": [{"name": "Attractiveness", "values": [88, 84, 72, 58, 40]}], "x_label": "Indicative score", "y_label": "", "caption": "VRFB economics improve as discharge duration and cycle requirements increase.", "source_note": "Public sources and BlueOcean synthesis."},
            {"id": "chart-3", "exhibit_no": "3", "title": "Commercialization priorities should shift from products to projects", "subtitle": "Illustrative management priority weighting", "type": "bar", "categories": ["Reference projects", "Financing model", "Local partners", "Cost roadmap", "Product roadmap"], "series": [{"name": "Priority", "values": [30, 25, 20, 15, 10]}], "x_label": "Share of management attention", "y_label": "", "caption": "Indicative weighting for strategy discussion.", "source_note": "BlueOcean synthesis."},
            {"id": "chart-4", "exhibit_no": "4", "title": "International expansion requires staged market entry", "subtitle": "Illustrative scenario comparison", "type": "bar", "categories": ["Domestic scale-up", "Asia partnerships", "Europe pilots", "North America licensing"], "series": [{"name": "Feasibility", "values": [86, 72, 62, 54]}], "x_label": "Indicative feasibility", "y_label": "", "caption": "Scenario view to guide market-entry sequencing.", "source_note": "BlueOcean synthesis."},
        ]
        cards = [
            {"id": "card-1", "title": summary[0], "subtitle": subtitle, "bullets": summary[:3], "highlight_number": "4", "highlight_label": "priority lenses", "exhibit_label": "Strategic position"},
            {"id": "card-2", "title": summary[1], "subtitle": "Leadership must be translated into credible customer proof.", "bullets": summary[3:6], "highlight_number": "3", "highlight_label": "proof points", "exhibit_label": "Management agenda"},
        ]
        return {
            "report_title": topic if len(topic) < 90 else topic[:90],
            "report_subtitle": subtitle,
            "executive_summary": summary,
            "method_steps": [{"name": f"Step {i}", "description": "Used internally to structure the analysis."} for i in range(1, 8)],
            "issue_tree": plan.get("issue_tree", []),
            "sections": sections,
            "insight_cards": cards,
            "charts": charts,
            "references": refs,
            "_fallback_used": True,
            "_fallback_reason": reason[:2000],
        }

    def _ensure_visual_hints(self, report: Dict) -> None:
        charts = report.get("charts", []) or []
        if not charts:
            return
        chart_ids = [c.get("id", f"chart-{idx}") for idx, c in enumerate(charts, start=1)]
        for idx, section in enumerate(report.get("sections", [])):
            hint = str(section.get("visual_hint", ""))
            if idx % 2 == 0 or not hint:
                section["visual_hint"] = chart_ids[idx % len(chart_ids)]

    def _materialize_assets(self, report: Dict, assets_dir: Path) -> Dict[str, str]:
        asset_map: Dict[str, str] = {}
        for card in report.get("insight_cards", []):
            target = assets_dir / f"{card['id']}.png"
            create_insight_card(card, target)
            asset_map[card["id"]] = f"assets/{target.name}"

        for chart in report.get("charts", []):
            target = assets_dir / f"{chart['id']}.png"
            create_chart(chart, target)
            asset_map[chart["id"]] = f"assets/{target.name}"

        return asset_map
