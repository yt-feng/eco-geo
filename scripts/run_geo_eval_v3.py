#!/usr/bin/env python3
from __future__ import annotations
import argparse, html, json, os, re, shutil, subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import requests, yaml
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
DIMS=["visibility","inclusion","cognition","outcome"]
TERM_BUCKETS=["brand_terms","competitor_terms","industry_terms","category_terms","problem_terms","trust_terms"]

def now(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
def esc(x): return html.escape(str(x))
def clamp(x):
    try: return max(0,min(100,float(x)))
    except Exception: return 0.0
def load_yaml(p:Path)->Dict[str,Any]: return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
def strip_json(t:str)->Dict[str,Any]:
    t=t.strip(); t=re.sub(r"^```[a-zA-Z0-9_-]*\n|\n```$","",t).strip()
    try: return json.loads(t)
    except Exception:
        m=re.search(r"\{.*\}",t,re.S)
        if not m: raise ValueError("No JSON in model response")
        return json.loads(m.group(0))
class DS:
    def __init__(self,model:str):
        self.model=model; self.key=os.getenv("DEEPSEEK_API_KEY",""); self.base=os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1/chat/completions"; self.runs=[]
        if not self.key: raise RuntimeError("DEEPSEEK_API_KEY required")
    def ask(self,stage:str,prompt:str)->Dict[str,Any]:
        body={"model":self.model,"temperature":0.2,"response_format":{"type":"json_object"},"messages":[{"role":"system","content":"You are a rigorous GEO research analyst. Return valid JSON only. Be structured, conservative, and consulting-grade."},{"role":"user","content":prompt}]}
        r=requests.post(self.base,headers={"Authorization":"Bearer "+self.key,"Content-Type":"application/json"},json=body,timeout=240); r.raise_for_status(); d=r.json(); c=d["choices"][0]["message"]["content"]; u=d.get("usage",{})
        self.runs.append({"stage":stage,"model":self.model,"total_tokens":u.get("total_tokens"),"prompt_tokens":u.get("prompt_tokens"),"completion_tokens":u.get("completion_tokens"),"response_chars":len(c),"timestamp":now()}); return strip_json(c)
def brand_from_args(a,cfg):
    b=dict(cfg.get("brand",{}))
    if a.brand_name: b["name"]=a.brand_name.strip()
    if a.brand_brief: b["brief"]=a.brand_brief.strip()
    if a.website: b["website"]=a.website.strip()
    if not b.get("name"): raise RuntimeError("Brand name required")
    if not b.get("brief"): b["brief"]=b.get("category") or "No brief provided; infer cautiously."
    b.setdefault("market","Global"); b.setdefault("language","zh-CN"); return b
def profile_prompt(b):
    schema={"brand_name":"","one_line_brief":"","official_website":"","category":"","market":"","language":"","positioning_summary":"","icp":[""],"buying_committee":[""],"core_offerings":[""],"must_own_topics":[""],"competitor_candidates":[{"name":"","reason":"","confidence":0}],"uncertainties":[""],"confidence":0}
    return "\n".join(["Stage 1: infer a brand GEO profile from minimal input. Output JSON only.",f"Brand name: {b.get('name','')}",f"One-line brief: {b.get('brief','')}",f"Optional website: {b.get('website','')}",f"Market: {b.get('market','Global')}","Infer official website only if reasonably confident. Do not pretend to browse the web.",json.dumps(schema,ensure_ascii=False,indent=2)])
def keywords_prompt(profile):
    schema={"term_buckets":{k:[""] for k in TERM_BUCKETS},"topic_cloud":[{"term":"","weight":0,"bucket":"category_terms"}],"semantic_clusters":[{"cluster":"","terms":[""],"geo_value":0,"competitive_intensity":0}],"taxonomy_summary":""}
    return "\n".join(["Stage 2: build a rich GEO keyword taxonomy. Output JSON only.",json.dumps(profile,ensure_ascii=False,indent=2),"Include brand terms, competitor terms, industry terms, category terms, problem terms, trust terms. 15-30 terms for important buckets.",json.dumps(schema,ensure_ascii=False,indent=2)])
def questions_prompt(profile,keywords):
    schema={"query_panel":[{"type":"brand","query":"","intent":"","funnel_stage":"Consider","importance":0,"risk_level":"medium"}],"question_families":[{"family":"","purpose":"","query_count":0,"representative_queries":[""],"recommended_engine_runs":0}],"monitoring_design":{"planned_query_count":0,"planned_deepseek_runs":0,"sampling_notes":""}}
    return "\n".join(["Stage 3: design a private GEO question universe. Output JSON only.","Generate at least 72 queries across brand, competitor, industry, category, problem, comparison, use_case, trust. Exact queries are internal.",json.dumps({"profile":profile,"keywords":keywords},ensure_ascii=False,indent=2),json.dumps(schema,ensure_ascii=False,indent=2)])
def competitors_prompt(profile,keywords,questions):
    schema={"competitors":[{"name":"","why_in_set":"","geo_maturity_stage":"Active","overall_score_estimate":0,"dimension_scores":{d:0 for d in DIMS},"visible_strengths":[""],"geo_moves_likely_underway":[""],"evidence_signals":[""],"confidence":0}],"market_pressure":{"peer_activation_index":0,"benchmark_percentile":0,"urgency_score":0,"gap_to_leading_peer":0,"narrative_disadvantage":0},"competitive_narrative":""}
    return "\n".join(["Stage 4: create a competitive GEO benchmark. Output JSON only.","Show where peers appear more GEO-active and why the target has urgency to move.",json.dumps({"profile":profile,"clusters":keywords.get("semantic_clusters",[]),"question_families":questions.get("question_families",[])},ensure_ascii=False,indent=2),json.dumps(schema,ensure_ascii=False,indent=2)])
def final_prompt(profile,keywords,questions,competitors):
    schema={"geo_evaluation":{d:{"score":0,"metrics":{},"rationale":"","confidence":0,"priority_actions":[""]} for d in DIMS},"evidence_map":{"owned_surface_strength":0,"entity_clarity":0,"content_modularity":0,"trust_signal_density":0,"comparison_page_readiness":0,"faq_readiness":0,"documentation_readiness":0,"pricing_transparency":0,"schema_readiness":0,"narrative_control":0},"journey_gap_matrix":[{"stage":"Discover","current_strength":0,"competitor_pressure":0,"opportunity":0,"notes":""}],"executive_summary":"","strengths":[""],"risks":[""],"methodology_note":""}
    return "\n".join(["Stage 5: final GEO scorecard and dashboard narrative. Output JSON only.","Use previous layers. Include at least 4 metrics per dimension. Make it data-driven, competitive, and dashboard-ready.",json.dumps({"profile":profile,"keywords":keywords,"questions_summary":questions.get("monitoring_design",{}),"competitors":competitors},ensure_ascii=False,indent=2),json.dumps(schema,ensure_ascii=False,indent=2)])
def run_research(b,model):
    ds=DS(model); profile=ds.ask("01_profile_deep_dive",profile_prompt(b)); keywords=ds.ask("02_keyword_taxonomy",keywords_prompt(profile)); questions=ds.ask("03_private_question_universe",questions_prompt(profile,keywords)); competitors=ds.ask("04_competitive_benchmark",competitors_prompt(profile,keywords,questions)); final=ds.ask("05_final_scorecard",final_prompt(profile,keywords,questions,competitors)); return {"brand_input":b,"profile":profile,"keywords":keywords,"questions":questions,"competitors":competitors,"final":final,"deepseek_runs":ds.runs}
def temp_config(out,b,r):
    p=r["profile"]; comps=[c.get("name") for c in r["competitors"].get("competitors",[]) if isinstance(c,dict) and c.get("name")]
    cfg={"brand":{"name":b.get("name"),"brief":b.get("brief"),"website":b.get("website") or p.get("official_website",""),"market":b.get("market",p.get("market","Global")),"region":b.get("region",p.get("market","Global")),"language":b.get("language",p.get("language","zh-CN")),"category":b.get("category",p.get("category","")),"narratives":p.get("must_own_topics",[])[:6],"competitors":comps[:8]},"weights":{"visibility":35,"inclusion":25,"cognition":25,"outcome":15},"thresholds":{"healthy":75,"warning":55}}
    path=out/"brand.generated.v3.yaml"; path.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding="utf-8"); return path
def run_v2(cfg,out,args):
    cmd=["python","scripts/run_geo_eval_v2.py","--brand-config",str(cfg),"--output",str(out),"--model",args.model,"--repo-root",args.repo_root,"--report-subdir",args.report_subdir,"--commit-message","chore: update GEO v2 dashboard from v3"]
    p=subprocess.run(cmd,cwd=Path(args.repo_root).resolve(),text=True,capture_output=True); return {"returncode":p.returncode,"stdout":p.stdout[-4000:],"stderr":p.stderr[-4000:]}
def topic_cloud(items):
    if not items: return "<p class='muted'>No terms.</p>"
    vals=[clamp(x.get("weight",x.get("count",1)),1,100) for x in items[:50]]; mx=max(vals) if vals else 1; spans=[]
    for x in items[:50]:
        size=12+int(30*clamp(x.get("weight",x.get("count",1)),1,100)/mx); spans.append(f"<span style='font-size:{size}px'>{esc(x.get('term',''))}</span>")
    return "<div class='cloud'>"+"".join(spans)+"</div>"
def svg_bar(items):
    if not items: return "<p class='muted'>No distribution.</p>"
    pairs=list(items.items())[:12]; mx=max(v for _,v in pairs) or 1; rows=[]; h=30+len(pairs)*24
    for i,(label,val) in enumerate(pairs):
        y=22+i*24; w=500*val/mx; rows.append(f"<text x='8' y='{y+13}' fill='#b9c7dd' font-size='12'>{esc(label)}</text><rect x='150' y='{y}' width='{w:.1f}' height='16' rx='7' fill='url(#g)'/><text x='{156+w:.1f}' y='{y+13}' fill='#e8eef9' font-size='12'>{val}</text>")
    return f"<svg viewBox='0 0 700 {h}' class='chart'><defs><linearGradient id='g' x1='0%' x2='100%'><stop offset='0%' stop-color='#38bdf8'/><stop offset='100%' stop-color='#818cf8'/></linearGradient></defs>{''.join(rows)}</svg>"
def bar(label,val,tone="default"):
    color={"default":"#60a5fa","good":"#34d399","warn":"#f59e0b","bad":"#fb7185"}.get(tone,"#60a5fa"); return f"<div class='bar'><div><span>{esc(label)}</span><b>{round(clamp(val),1)}</b></div><p><i style='width:{clamp(val)}%;background:{color}'></i></p></div>"
def render_dashboard(r,out):
    b=r["brand_input"]; final=r["final"]; qs=r["questions"]; kws=r["keywords"]; comps=r["competitors"]; geo=final.get("geo_evaluation",{}); pressure=comps.get("market_pressure",{}); qpanel=qs.get("query_panel",[]) if isinstance(qs.get("query_panel",[]),list) else []; qdist=Counter(str(q.get("type","other")) for q in qpanel if isinstance(q,dict)); comp_rows=""
    for c in comps.get("competitors",[])[:8]:
        if not isinstance(c,dict): continue
        ds=c.get("dimension_scores",{}) if isinstance(c.get("dimension_scores",{}),dict) else {}; comp_rows+=f"<tr><td>{esc(c.get('name',''))}</td><td>{esc(c.get('geo_maturity_stage',''))}</td><td>{c.get('overall_score_estimate','')}</td><td>{ds.get('visibility','')}</td><td>{ds.get('inclusion','')}</td><td>{ds.get('cognition','')}</td><td>{ds.get('outcome','')}</td><td>{esc('; '.join(c.get('evidence_signals',[])[:2]))}</td></tr>"
    cards=""
    for d in DIMS:
        item=geo.get(d,{}) if isinstance(geo.get(d,{}),dict) else {}; metrics=item.get("metrics",{}) if isinstance(item.get("metrics",{}),dict) else {}; cards+=f"<section class='card'><h3>{d.title()}</h3>{bar('Score',item.get('score',0))}{''.join(bar(k.replace('_',' ').title(),v) for k,v in list(metrics.items())[:6])}<p class='muted'>{esc(item.get('rationale',''))}</p><ul>{''.join(f'<li>{esc(a)}</li>' for a in item.get('priority_actions',[])[:4])}</ul></section>"
    html_txt=f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{esc(b.get('name',''))} GEO v3 Research Dashboard</title><style>body{{margin:0;background:#07111f;color:#e7eefb;font-family:Inter,Segoe UI,Arial,sans-serif}}.wrap{{max-width:1500px;margin:auto;padding:24px}}.hero,.grid2,.grid4{{display:grid;gap:18px}}.hero{{grid-template-columns:2fr 1fr}}.grid2{{grid-template-columns:1.15fr .85fr}}.grid4{{grid-template-columns:repeat(4,1fr)}}.card,.kpi{{background:#0f1b2d;border:1px solid #22324a;border-radius:18px;padding:18px;margin-bottom:18px;box-shadow:0 12px 28px rgba(0,0,0,.18)}}.kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:16px}}.kpi h2{{font-size:30px;margin:6px 0}}.muted,small{{color:#93a4bb}}.bar div{{display:flex;justify-content:space-between;font-size:13px}}.bar p{{height:10px;background:#0b1424;border:1px solid #24344d;border-radius:999px;overflow:hidden}}.bar i{{display:block;height:100%}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{border-bottom:1px solid #22324a;padding:10px;text-align:left;vertical-align:top}}th{{color:#b9c7dd}}.cloud{{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end}}.cloud span{{background:#111f34;border:1px solid #22324a;border-radius:999px;padding:6px 10px}}.chart{{width:100%;height:auto}}@media(max-width:1100px){{.hero,.grid2,.grid4,.kpis{{grid-template-columns:1fr}}}}</style></head><body><div class='wrap'><section class='hero'><div class='card'><small>GEO v3 Research Dashboard</small><h1>{esc(b.get('name',''))}</h1><p class='muted'>{esc(b.get('brief',''))}</p><p>{esc(final.get('executive_summary',''))}</p><div class='kpis'><div class='kpi'><small>Actual DeepSeek calls</small><h2>{len(r['deepseek_runs'])}</h2></div><div class='kpi'><small>Private query universe</small><h2>{len(qpanel)}</h2></div><div class='kpi'><small>Question families</small><h2>{len(qs.get('question_families',[]))}</h2></div><div class='kpi'><small>Competitors benchmarked</small><h2>{len(comps.get('competitors',[]))}</h2></div><div class='kpi'><small>Topic clusters</small><h2>{len(kws.get('semantic_clusters',[]))}</h2></div></div></div><div class='card'><h3>Market pressure</h3>{bar('Peer activation index',pressure.get('peer_activation_index',0),'warn')}{bar('Urgency score',pressure.get('urgency_score',0),'bad')}{bar('Gap to leading peer',pressure.get('gap_to_leading_peer',0),'bad')}{bar('Narrative disadvantage',pressure.get('narrative_disadvantage',0),'bad')}</div></section><section class='grid2'><div class='card'><h3>Question universe mix</h3>{svg_bar(dict(qdist))}</div><div class='card'><h3>Research topic cloud</h3>{topic_cloud(kws.get('topic_cloud',[]))}</div></section><section class='card'><h3>Competitor GEO leaderboard</h3><table><tr><th>Competitor</th><th>Stage</th><th>Overall</th><th>Visibility</th><th>Inclusion</th><th>Cognition</th><th>Outcome</th><th>Evidence signal</th></tr>{comp_rows}</table></section><section class='grid4'>{cards}</section></div></body></html>"""
    (out/"dashboard.html").write_text(html_txt,encoding="utf-8")
def render_audit(r,out):
    kws=r["keywords"]; qs=r["questions"]; term_cards="".join(f"<section class='card'><h3>{esc(k.replace('_',' ').title())}</h3><p>{esc(', '.join([str(x) for x in v]))}</p></section>" for k,v in kws.get("term_buckets",{}).items()); rows=""; 
    for fam in qs.get("question_families",[])[:24]:
        if isinstance(fam,dict): rows+=f"<tr><td>{esc(fam.get('family',''))}</td><td>{esc(fam.get('purpose',''))}</td><td>{fam.get('query_count','')}</td><td>{fam.get('recommended_engine_runs','')}</td><td>{esc('; '.join([str(x) for x in fam.get('representative_queries',[])[:5]]))}</td></tr>"
    run_rows="".join(f"<tr><td>{esc(x['stage'])}</td><td>{esc(x['model'])}</td><td>{x.get('total_tokens')}</td><td>{x.get('response_chars')}</td></tr>" for x in r["deepseek_runs"]); html_txt=f"<html><head><meta charset='utf-8'><style>body{{background:#07111f;color:#e7eefb;font-family:Inter,Segoe UI,Arial,sans-serif;padding:24px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{background:#0f1b2d;border:1px solid #22324a;border-radius:16px;padding:16px;margin-bottom:16px}}table{{width:100%;border-collapse:collapse}}td,th{{border-bottom:1px solid #22324a;padding:9px;text-align:left;vertical-align:top}}</style></head><body><h1>{esc(r['brand_input'].get('name',''))} Internal GEO Research Audit</h1><p>Actual DeepSeek pipeline calls: <b>{len(r['deepseek_runs'])}</b></p><div class='grid'>{term_cards}</div><section class='card'><h2>Question families and representative queries</h2><table><tr><th>Family</th><th>Purpose</th><th>Queries</th><th>Runs</th><th>Representative queries</th></tr>{rows}</table></section><section class='card'><h2>DeepSeek pipeline calls</h2><table><tr><th>Stage</th><th>Model</th><th>Total tokens</th><th>Response chars</th></tr>{run_rows}</table></section></body></html>"; (out/"internal_audit.html").write_text(html_txt,encoding="utf-8"); (out/"internal_audit.json").write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8"); (out/"internal_audit.md").write_text(json.dumps({"term_buckets":kws.get("term_buckets",{}),"question_families":qs.get("question_families",[]),"deepseek_runs":r["deepseek_runs"]},ensure_ascii=False,indent=2),encoding="utf-8")
def make_pdf(path,title,r,internal=False):
    styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name="Small",parent=styles["BodyText"],fontSize=8,leading=10)); story=[Paragraph(esc(title),styles["Title"]),Spacer(1,8)]
    if internal:
        rows=[["Stage","Model","Tokens","Chars"]]+[[x["stage"],x["model"],x.get("total_tokens"),x.get("response_chars")] for x in r["deepseek_runs"]]; story.append(Paragraph("DeepSeek Pipeline Calls",styles["Heading2"])); story.append(Table(rows,repeatRows=1));
        for k,v in r["keywords"].get("term_buckets",{}).items(): story.append(Paragraph(k.replace("_"," ").title(),styles["Heading2"])); story.append(Paragraph(esc(", ".join([str(x) for x in v[:30]])),styles["Small"]))
    else:
        story.append(Paragraph("Executive Summary",styles["Heading2"])); story.append(Paragraph(esc(r["final"].get("executive_summary","")),styles["Small"])); rows=[["Metric","Value"],["DeepSeek calls",len(r["deepseek_runs"])],["Private queries",len(r["questions"].get("query_panel",[]))],["Question families",len(r["questions"].get("question_families",[]))],["Competitors",len(r["competitors"].get("competitors",[]))]]; story.append(Spacer(1,8)); story.append(Table(rows,repeatRows=1))
    for obj in story:
        if isinstance(obj,Table): obj.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#d9e6fb")),("GRID",(0,0),(-1,-1),0.25,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=12*mm,rightMargin=12*mm,topMargin=12*mm,bottomMargin=12*mm).build(story)
def commit(repo,out,subdir,msg):
    root=Path(repo).resolve(); target=root/subdir
    if target.exists(): shutil.rmtree(target)
    target.parent.mkdir(parents=True,exist_ok=True); shutil.copytree(out,target); subprocess.run(["git","add",str(target)],cwd=root,check=True); st=subprocess.run(["git","status","--porcelain"],cwd=root,check=True,capture_output=True,text=True)
    if not st.stdout.strip(): return {"committed":False,"target":str(target)}
    subprocess.run(["git","commit","-m",msg],cwd=root,check=True); return {"committed":True,"target":str(target)}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--brand-config",default="config/brand.yaml"); p.add_argument("--brand-name",default=""); p.add_argument("--brand-brief",default=""); p.add_argument("--website",default=""); p.add_argument("--model",default="deepseek-chat"); p.add_argument("--output",default="dist/report"); p.add_argument("--repo-root",default="."); p.add_argument("--report-subdir",default="reports/latest"); p.add_argument("--commit-message",default="chore: update GEO v3 research dashboard"); p.add_argument("--commit-report",action="store_true"); a=p.parse_args(); root=Path(a.repo_root).resolve(); out=Path(a.output).resolve(); out.mkdir(parents=True,exist_ok=True); b=brand_from_args(a,load_yaml(root/a.brand_config)); research=run_research(b,a.model); (out/"research_layers.json").write_text(json.dumps(research,ensure_ascii=False,indent=2),encoding="utf-8"); cfg=temp_config(out,b,research); v2=run_v2(cfg,out,a); (out/"v2_runner_result.json").write_text(json.dumps(v2,ensure_ascii=False,indent=2),encoding="utf-8"); render_dashboard(research,out); render_audit(research,out); make_pdf(out/"dashboard.pdf",f"{b.get('name')} GEO v3 Research Dashboard",research,False); make_pdf(out/"internal_audit.pdf",f"{b.get('name')} Internal GEO Audit",research,True); summary={"brand":b,"actual_deepseek_pipeline_calls":len(research["deepseek_runs"]),"private_query_count":len(research["questions"].get("query_panel",[])),"question_family_count":len(research["questions"].get("question_families",[])),"competitor_count":len(research["competitors"].get("competitors",[])),"v2_runner_returncode":v2["returncode"]}; (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); info={"committed":False}
    if a.commit_report: info=commit(root,out,a.report_subdir,a.commit_message)
    print(json.dumps({"brand":b.get("name"),"deepseek_calls":len(research["deepseek_runs"]),"private_queries":len(research["questions"].get("query_panel",[])),"commit":info},ensure_ascii=False))
if __name__=="__main__": main()
