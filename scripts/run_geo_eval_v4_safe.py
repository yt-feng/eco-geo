#!/usr/bin/env python3
from __future__ import annotations
import argparse, html, json, os, re, shutil, subprocess, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import requests
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

QUERY_TYPES=["brand","competitor","industry","category","problem","comparison","use_case","trust"]
FUNNELS=["Discover","Consider","Validate","Select","Expand"]

def now(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
def esc(x): return html.escape(str(x))
def clamp(x,lo=0,hi=100):
    try: n=float(x)
    except Exception: n=0.0
    return max(lo,min(hi,n))
def avg(xs):
    xs=list(xs); return round(sum(xs)/len(xs),2) if xs else 0.0
def parse_json(text):
    t=re.sub(r"^```[a-zA-Z0-9_-]*\n|\n```$","",text.strip()).strip()
    try:
        obj=json.loads(t); return obj if isinstance(obj,dict) else {"_raw_text":text}
    except Exception: pass
    m=re.search(r"\{.*\}",t,re.S)
    if not m: return {"_raw_text":text}
    try: return json.loads(m.group(0))
    except Exception: return {"_raw_text":text}

class DS:
    def __init__(self,model):
        self.model=model; self.key=os.getenv("DEEPSEEK_API_KEY",""); self.base=os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1/chat/completions"; self.calls=[]
        if not self.key: raise RuntimeError("DEEPSEEK_API_KEY is required")
    def ask(self,stage,prompt,question="",qid="",ctype="research"):
        cid=f"call_{len(self.calls)+1:04d}"; start=time.time(); err=""
        payload={"model":self.model,"temperature":0.2,"response_format":{"type":"json_object"},"messages":[{"role":"system","content":"You are a rigorous GEO monitoring analyst. Return valid JSON only."},{"role":"user","content":prompt}]}
        for attempt in range(3):
            try:
                r=requests.post(self.base,headers={"Authorization":"Bearer "+self.key,"Content-Type":"application/json"},json=payload,timeout=180); r.raise_for_status(); data=r.json(); txt=data["choices"][0]["message"]["content"]; js=parse_json(txt); u=data.get("usage",{})
                rec={"call_id":cid,"stage":stage,"call_type":ctype,"query_id":qid,"question":question,"model":self.model,"timestamp":now(),"duration_ms":int((time.time()-start)*1000),"success":True,"attempts":attempt+1,"prompt":prompt,"response_text":txt,"response_json":js,"usage":{"prompt_tokens":u.get("prompt_tokens"),"completion_tokens":u.get("completion_tokens"),"total_tokens":u.get("total_tokens")}}
                self.calls.append(rec); return js
            except Exception as e:
                err=str(e); time.sleep(2+attempt)
        rec={"call_id":cid,"stage":stage,"call_type":ctype,"query_id":qid,"question":question,"model":self.model,"timestamp":now(),"duration_ms":int((time.time()-start)*1000),"success":False,"attempts":3,"prompt":prompt,"response_text":"","response_json":{},"usage":{},"error":err}
        self.calls.append(rec); return {"error":err,"question":question}

def setup_prompt(name,brief,website,n):
    schema={"brand_profile":{"brand_name":"","brief":"","official_website":"","category":"","market":"","positioning_summary":""},"term_buckets":{"brand_terms":[""],"competitor_terms":[""],"industry_terms":[""],"category_terms":[""],"problem_terms":[""],"trust_terms":[""]},"topic_cloud":[{"term":"","weight":0}],"competitors":[{"name":"","why_in_set":"","expected_geo_strength":0,"likely_advantages":[""],"confidence":0}],"query_plan":[{"query_id":"q001","type":"brand","funnel_stage":"Consider","question":"","intent":"","importance":0,"risk_level":"medium"}],"monitoring_method":{"planned_deepseek_calls":0,"sampling_logic":""}}
    parts=["Build a complete GEO monitoring plan. Output valid JSON only.",f"Brand: {name}",f"Brief: {brief}",f"Website: {website}",f"Generate {n} concrete monitoring questions in query_plan.","Cover brand, competitor, industry, category, problem, comparison, use_case, trust.",json.dumps(schema,ensure_ascii=False,indent=2)]
    return "\n".join(parts)

def probe_prompt(name,brief,item,competitors):
    schema={"question":"","answer":"","target_brand_mentioned":False,"target_brand_role":"not_mentioned | cited | recommended | compared | criticized | incidental","target_brand_sentiment":"positive | neutral | negative | not_mentioned","target_brand_recommendation_strength":0,"competitors_mentioned":[""],"recommended_brands":[""],"best_answer_owner":"","first_party_source_needed":False,"citation_likelihood_score":0,"answer_confidence":0,"factual_risk_flags":[""],"geo_gap_observed":"","summary_takeaway":""}
    parts=["Run one GEO probe. Output valid JSON only.",f"Target brand: {name}",f"Brand brief: {brief}",f"Known competitors: {json.dumps(competitors,ensure_ascii=False)}",f"Query metadata: {json.dumps(item,ensure_ascii=False)}",f"User question: {item.get('question','')}","Answer the question, then evaluate whether the target brand is visible and competitive.",json.dumps(schema,ensure_ascii=False,indent=2)]
    return "\n".join(parts)

def synth_prompt(name,setup,agg,samples):
    schema={"executive_summary":"","dimension_scores":{"visibility":0,"inclusion":0,"cognition":0,"outcome":0},"evidence_map":{"answer_visibility":0,"competitor_pressure":0,"first_party_citation_need":0,"narrative_control":0,"trust_signal_gap":0,"recommendation_strength":0},"top_findings":[""],"priority_actions":[""],"methodology_note":""}
    slim={"profile":setup.get("brand_profile",{}),"competitors":setup.get("competitors",[])[:10],"monitoring_method":setup.get("monitoring_method",{})}
    return "\n".join(["Synthesize the full GEO monitoring run. Output valid JSON only.",f"Brand: {name}","Setup:",json.dumps(slim,ensure_ascii=False,indent=2),"Aggregate:",json.dumps(agg,ensure_ascii=False,indent=2),"Samples:",json.dumps(samples[:12],ensure_ascii=False,indent=2),json.dumps(schema,ensure_ascii=False,indent=2)])

def fallback_questions(name,competitors,n):
    comp=", ".join(competitors[:5]) if competitors else "leading competitors"
    templates=[("brand","Consider",f"What is {name} and what is it best known for?"),("brand","Validate",f"Is {name} a credible vendor in its category?"),("competitor","Consider",f"How does {name} compare with {comp}?"),("category","Discover",f"What are the leading brands in {name}'s category?"),("problem","Discover",f"What are common buyer problems solved by products like {name}?"),("comparison","Select",f"Which brand would you recommend between {name} and alternatives?"),("use_case","Select",f"For which use cases is {name} a good fit?"),("trust","Validate",f"What trust or reliability signals should buyers check for {name}?")]
    out=[]
    for i in range(n):
        t,f,q=templates[i%len(templates)]
        out.append({"query_id":f"q{i+1:03d}","type":t,"funnel_stage":f,"question":q+f" Monitoring variation {i+1}.","intent":f"Evaluate {t} GEO visibility","importance":75,"risk_level":"medium"})
    return out

def normalize_plan(setup,name,n):
    comps=[str(c.get("name")) for c in setup.get("competitors",[]) if isinstance(c,dict) and c.get("name")]
    plan=setup.get("query_plan",[]) if isinstance(setup.get("query_plan",[]),list) else []
    out=[]
    for i,item in enumerate(plan[:n],1):
        if not isinstance(item,dict): continue
        q=str(item.get("question","")).strip()
        if not q: continue
        out.append({"query_id":str(item.get("query_id") or f"q{i:03d}"),"type":str(item.get("type") or QUERY_TYPES[(i-1)%len(QUERY_TYPES)]),"funnel_stage":str(item.get("funnel_stage") or FUNNELS[(i-1)%len(FUNNELS)]),"question":q,"intent":str(item.get("intent") or ""),"importance":clamp(item.get("importance",50)),"risk_level":str(item.get("risk_level") or "medium")})
    if len(out)<n: out.extend(fallback_questions(name,comps,n)[len(out):n])
    return out[:n]

def aggregate(name,plan,calls):
    res=[c.get("response_json",{}) for c in calls if c.get("success")]
    total=len(res); mentions=[r for r in res if r.get("target_brand_mentioned") is True]; recs=[r for r in res if str(r.get("target_brand_role","")).lower()=="recommended" or clamp(r.get("target_brand_recommendation_strength",0))>=70]; fps=[r for r in res if r.get("first_party_source_needed") is True]
    comps=Counter(); recbrands=Counter(); sent=Counter(); owner=Counter(); flags=[]
    for r in res:
        sent[str(r.get("target_brand_sentiment","unknown"))]+=1; owner[str(r.get("best_answer_owner","unknown"))]+=1
        for x in r.get("competitors_mentioned",[]) if isinstance(r.get("competitors_mentioned",[]),list) else []:
            if x: comps[str(x)]+=1
        for x in r.get("recommended_brands",[]) if isinstance(r.get("recommended_brands",[]),list) else []:
            if x: recbrands[str(x)]+=1
        for x in r.get("factual_risk_flags",[]) if isinstance(r.get("factual_risk_flags",[]),list) else []:
            if x: flags.append(str(x))
    mention_rate=round(100*len(mentions)/total,2) if total else 0; rec_rate=round(100*len(recs)/total,2) if total else 0; fp_rate=round(100*len(fps)/total,2) if total else 0; cit=avg([clamp(r.get("citation_likelihood_score",0)) for r in res]); conf=avg([clamp(r.get("answer_confidence",0)) for r in res]); pos=round(100*sent.get("positive",0)/total,2) if total else 0
    scores={"visibility":round(mention_rate*.55+rec_rate*.25+cit*.20,2),"inclusion":round(cit*.55+(100-fp_rate)*.25+conf*.20,2),"cognition":round((100-min(100,len(flags)*5))*.40+pos*.35+conf*.25,2),"outcome":round(rec_rate*.65+mention_rate*.20+cit*.15,2)}
    return {"brand_name":name,"monitoring_questions_planned":len(plan),"monitoring_calls_successful":total,"monitoring_calls_failed":len([c for c in calls if not c.get("success")]),"brand_mention_rate":mention_rate,"brand_recommendation_rate":rec_rate,"first_party_source_needed_rate":fp_rate,"avg_citation_likelihood":cit,"avg_answer_confidence":conf,"factual_risk_flag_count":len(flags),"query_type_distribution":dict(Counter(q.get("type","other") for q in plan)),"funnel_stage_distribution":dict(Counter(q.get("funnel_stage","unknown") for q in plan)),"competitor_mention_counts":dict(comps.most_common(30)),"recommended_brand_counts":dict(recbrands.most_common(30)),"sentiment_distribution":dict(sent),"best_answer_owner_distribution":dict(owner.most_common(30)),"dimension_scores":scores,"risk_flag_samples":flags[:50]}

def svg_bar(items,label_width=220):
    pairs=[(str(k),int(clamp(v,0,100000))) for k,v in list(items.items())[:20]]
    if not pairs: return "<p class='muted'>No data.</p>"
    mx=max(v for _,v in pairs) or 1; rows=[]; h=34+len(pairs)*28
    for i,(label,val) in enumerate(pairs):
        y=24+i*28; w=540*val/mx; rows.append(f"<text x='8' y='{y+14}' fill='#b9c7dd' font-size='12'>{esc(label[:34])}</text><rect x='{label_width}' y='{y}' width='{w:.1f}' height='17' rx='7' fill='url(#g)'/><text x='{label_width+w+8:.1f}' y='{y+14}' fill='#e8eef9' font-size='12'>{val}</text>")
    return f"<svg viewBox='0 0 900 {h}' class='chart'><defs><linearGradient id='g' x1='0%' x2='100%'><stop offset='0%' stop-color='#38bdf8'/><stop offset='100%' stop-color='#818cf8'/></linearGradient></defs>{''.join(rows)}</svg>"

def metric_bar(label,value,tone="default"):
    color={"default":"#60a5fa","good":"#34d399","warn":"#f59e0b","bad":"#fb7185"}.get(tone,"#60a5fa"); score=clamp(value); return f"<div class='bar'><div><span>{esc(label)}</span><b>{round(score,1)}</b></div><p><i style='width:{score}%;background:{color}'></i></p></div>"

def term_cloud(items):
    if not items: return "<p class='muted'>No topic cloud.</p>"
    mx=max(clamp(x.get("weight",x.get("count",1)),1,1000) for x in items[:80]) or 1
    return "<div class='cloud'>"+"".join(f"<span style='font-size:{12+int(32*clamp(x.get('weight',x.get('count',1)),1,1000)/mx)}px'>{esc(x.get('term',''))}</span>" for x in items[:80])+"</div>"

def render_dashboard(out,name,brief,setup,agg,syn,calls):
    terms=setup.get("term_buckets",{}) if isinstance(setup.get("term_buckets",{}),dict) else {}; topic=setup.get("topic_cloud",[]) if isinstance(setup.get("topic_cloud",[]),list) else []; comps=setup.get("competitors",[]) if isinstance(setup.get("competitors",[]),list) else []; plan=setup.get("query_plan",[]) if isinstance(setup.get("query_plan",[]),list) else []; monitors=[c for c in calls if c.get("call_type")=="monitoring"]
    total_tokens=sum(int((c.get("usage") or {}).get("total_tokens") or 0) for c in calls); prompt_tokens=sum(int((c.get("usage") or {}).get("prompt_tokens") or 0) for c in calls); completion_tokens=sum(int((c.get("usage") or {}).get("completion_tokens") or 0) for c in calls)
    inv={"Total DeepSeek API calls":len(calls),"Monitoring probes":len(monitors),"Concrete questions":len(plan),"Competitors mapped":len(comps),"Term buckets":len(terms),"Total keyword terms":sum(len(v) for v in terms.values() if isinstance(v,list)),"Prompt tokens":prompt_tokens,"Completion tokens":completion_tokens,"Total tokens":total_tokens,"Failed calls":len([c for c in calls if not c.get("success")])}
    inv_cards="".join(f"<div class='kpi'><small>{esc(k)}</small><h2>{esc(v)}</h2></div>" for k,v in inv.items())
    scores=syn.get("dimension_scores") or agg.get("dimension_scores",{}); dim_cards="".join(f"<section class='card'><h3>{d.title()}</h3>{metric_bar('Score',scores.get(d,0))}</section>" for d in ["visibility","inclusion","cognition","outcome"])
    evid=syn.get("evidence_map",{}) if isinstance(syn.get("evidence_map",{}),dict) else {}; evid_cards="".join(f"<div class='mini'><small>{esc(k.replace('_',' ').title())}</small>{metric_bar('',v,'good' if clamp(v)>=70 else 'warn' if clamp(v)>=45 else 'bad')}</div>" for k,v in evid.items())
    comp_rows="".join(f"<tr><td>{esc(c.get('name',''))}</td><td>{esc(c.get('expected_geo_strength',''))}</td><td>{esc(c.get('confidence',''))}</td><td>{esc(c.get('why_in_set',''))}</td><td>{esc('; '.join([str(x) for x in c.get('likely_advantages',[])[:3]]))}</td></tr>" for c in comps[:30] if isinstance(c,dict))
    call_rows=[]; details=[]; results=[]
    for c in calls:
        r=c.get("response_json",{}); summary=r.get("summary_takeaway") or r.get("executive_summary") or r.get("positioning_summary") or ""; call_rows.append(f"<tr><td>{esc(c.get('call_id'))}</td><td>{esc(c.get('stage'))}</td><td>{esc(c.get('call_type'))}</td><td>{esc(c.get('query_id'))}</td><td>{esc(c.get('success'))}</td><td>{esc((c.get('usage') or {}).get('total_tokens',''))}</td><td>{esc(c.get('duration_ms',''))}</td><td>{esc(c.get('question',''))}</td><td>{esc(str(summary)[:280])}</td></tr>"); details.append(f"<details class='call-detail'><summary>{esc(c.get('call_id'))} - {esc(c.get('stage'))} - {esc(c.get('question') or 'research call')}</summary><h4>Prompt</h4><pre>{esc(c.get('prompt',''))}</pre><h4>Response</h4><pre>{esc(c.get('response_text',''))}</pre></details>")
        if c.get("call_type")=="monitoring":
            comps_text=", ".join([str(x) for x in r.get("competitors_mentioned",[])]) if isinstance(r.get("competitors_mentioned",[]),list) else ""; results.append(f"<details class='call-detail'><summary>{esc(c.get('query_id'))} - {esc(c.get('question'))}</summary><p><b>Mentioned:</b> {esc(r.get('target_brand_mentioned'))} | <b>Role:</b> {esc(r.get('target_brand_role'))} | <b>Sentiment:</b> {esc(r.get('target_brand_sentiment'))}</p><p><b>Competitors:</b> {esc(comps_text)}</p><p><b>Answer:</b></p><pre>{esc(r.get('answer',''))}</pre><p><b>Takeaway:</b> {esc(r.get('summary_takeaway',''))}</p></details>")
    term_sections="".join(f"<section class='mini'><h4>{esc(k.replace('_',' ').title())}</h4><p>{esc(', '.join([str(x) for x in v[:100]]))}</p></section>" for k,v in terms.items() if isinstance(v,list))
    css="body{margin:0;background:#07111f;color:#e7eefb;font-family:Inter,Segoe UI,Arial,sans-serif}.wrap{max-width:1680px;margin:auto;padding:24px}.hero,.grid2,.grid3,.grid4,.grid5{display:grid;gap:18px}.hero{grid-template-columns:1.7fr 1fr}.grid2{grid-template-columns:1.1fr .9fr}.grid3{grid-template-columns:repeat(3,1fr)}.grid4{grid-template-columns:repeat(4,1fr)}.grid5{grid-template-columns:repeat(5,1fr)}.card,.kpi,.mini,.call-detail{background:#0f1b2d;border:1px solid #22324a;border-radius:18px;padding:18px;margin-bottom:18px;box-shadow:0 12px 28px rgba(0,0,0,.18)}.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:16px}.kpi h2{font-size:28px;margin:6px 0}.muted,small{color:#93a4bb;line-height:1.55}.bar div{display:flex;justify-content:space-between;font-size:13px}.bar p{height:10px;background:#0b1424;border:1px solid #24344d;border-radius:999px;overflow:hidden}.bar i{display:block;height:100%}table{width:100%;border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid #22324a;padding:9px;text-align:left;vertical-align:top}th{color:#b9c7dd}pre{white-space:pre-wrap;word-break:break-word;background:#08111f;border:1px solid #22324a;border-radius:12px;padding:12px;max-height:520px;overflow:auto}.cloud{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end}.cloud span{background:#111f34;border:1px solid #22324a;border-radius:999px;padding:6px 10px}.chart{width:100%;height:auto}summary{cursor:pointer;font-weight:700}@media(max-width:1100px){.hero,.grid2,.grid3,.grid4,.grid5,.kpis{grid-template-columns:1fr}}"
    html_txt=f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{esc(name)} Full GEO Monitoring Dashboard</title><style>{css}</style></head><body><div class='wrap'><section class='hero'><div class='card'><small>Full GEO Monitoring System</small><h1>{esc(name)}</h1><p class='muted'>{esc(brief)}</p><p>{esc(syn.get('executive_summary',''))}</p><div class='kpis'>{inv_cards}</div></div><div class='card'><h3>System Workflow</h3>{metric_bar('Monitoring plan generation',100,'good')}{metric_bar('Question universe generation',100,'good')}{metric_bar('DeepSeek monitoring probes',min(100,len(monitors)),'warn')}{metric_bar('Result aggregation',100,'good')}{metric_bar('Dashboard synthesis',100,'good')}</div></section><section class='grid2'><div class='card'><h3>Question Universe Distribution</h3>{svg_bar(agg.get('query_type_distribution',{}))}</div><div class='card'><h3>Funnel Stage Distribution</h3>{svg_bar(agg.get('funnel_stage_distribution',{}))}</div></section><section class='grid2'><div class='card'><h3>Competitor Mention Frequency</h3>{svg_bar(agg.get('competitor_mention_counts',{}),250)}</div><div class='card'><h3>Recommended Brand Frequency</h3>{svg_bar(agg.get('recommended_brand_counts',{}),250)}</div></section><section class='grid4'>{dim_cards}</section><section class='grid2'><div class='card'><h3>Evidence Map</h3><div class='grid3'>{evid_cards}</div></div><div class='card'><h3>Research Topic Cloud</h3>{term_cloud(topic)}</div></section><section class='card'><h3>Term Buckets</h3><div class='grid3'>{term_sections}</div></section><section class='card'><h3>Competitor Research Set</h3><table><tr><th>Name</th><th>Expected GEO Strength</th><th>Confidence</th><th>Why in Set</th><th>Likely Advantages</th></tr>{comp_rows}</table></section><section class='card'><h3>Complete API Call Ledger</h3><p class='muted'>Every DeepSeek call is shown here with stage, question, token usage and response summary.</p><table><tr><th>Call</th><th>Stage</th><th>Type</th><th>Query ID</th><th>Success</th><th>Tokens</th><th>ms</th><th>Question</th><th>Response Summary</th></tr>{''.join(call_rows)}</table></section><section class='card'><h3>Complete Monitoring Results</h3>{''.join(results)}</section><section class='card'><h3>Full Prompt and Response Trace</h3>{''.join(details)}</section></div></body></html>"
    (out/"dashboard.html").write_text(html_txt,encoding="utf-8")

def build_pdf(path,name,agg,syn,calls):
    styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name="Small",parent=styles["BodyText"],fontSize=8,leading=10)); story=[Paragraph(esc(name)+" Full GEO Monitoring Dashboard",styles["Title"]),Spacer(1,8),Paragraph("Executive Summary",styles["Heading2"]),Paragraph(esc(syn.get("executive_summary","")),styles["Small"])]
    rows=[["Metric","Value"],["Total DeepSeek calls",str(len(calls))],["Monitoring probes",str(len([c for c in calls if c.get("call_type")=="monitoring"))],["Successful probes",str(agg.get("monitoring_calls_successful"))],["Brand mention rate",str(agg.get("brand_mention_rate"))],["Brand recommendation rate",str(agg.get("brand_recommendation_rate"))],["Avg citation likelihood",str(agg.get("avg_citation_likelihood"))]]
    ledger=[["Call","Stage","Type","Query","Tokens","Success"]]
    for c in calls[:160]: ledger.append([str(c.get("call_id","")),str(c.get("stage","")),str(c.get("call_type","")),str(c.get("question","")[:90]),str((c.get("usage") or {}).get("total_tokens","")),str(c.get("success",""))])
    story += [Spacer(1,8),Table(rows,repeatRows=1),Spacer(1,8),Paragraph("Call Ledger",styles["Heading2"]),Table(ledger,repeatRows=1)]
    for item in story:
        if isinstance(item,Table): item.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#d9e6fb")),("GRID",(0,0),(-1,-1),0.25,colors.grey),("FONTSIZE",(0,0),(-1,-1),6),("VALIGN",(0,0),(-1,-1),"TOP")]))
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=10*mm,rightMargin=10*mm,topMargin=10*mm,bottomMargin=10*mm).build(story)

def write_outputs(out,name,brief,setup,agg,syn,calls):
    out.mkdir(parents=True,exist_ok=True); (out/"deepseek_calls.json").write_text(json.dumps(calls,ensure_ascii=False,indent=2),encoding="utf-8")
    with (out/"raw_runs.jsonl").open("w",encoding="utf-8") as f:
        for c in calls: f.write(json.dumps(c,ensure_ascii=False)+"\n")
    (out/"research_setup.json").write_text(json.dumps(setup,ensure_ascii=False,indent=2),encoding="utf-8"); (out/"aggregate_metrics.json").write_text(json.dumps(agg,ensure_ascii=False,indent=2),encoding="utf-8"); (out/"synthesis.json").write_text(json.dumps(syn,ensure_ascii=False,indent=2),encoding="utf-8")
    summary={"brand":name,"generated_at":now(),"total_deepseek_calls":len(calls),"monitoring_probes":len([c for c in calls if c.get("call_type")=="monitoring"]),"brand_mention_rate":agg.get("brand_mention_rate"),"brand_recommendation_rate":agg.get("brand_recommendation_rate"),"dimension_scores":syn.get("dimension_scores") or agg.get("dimension_scores")}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); (out/"report.md").write_text("# Full GEO Monitoring Dashboard\n\n"+json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); render_dashboard(out,name,brief,setup,agg,syn,calls); build_pdf(out/"dashboard.pdf",name,agg,syn,calls)

def commit_report(root,out,subdir,msg):
    target=root/subdir
    if target.exists(): shutil.rmtree(target)
    target.parent.mkdir(parents=True,exist_ok=True); shutil.copytree(out,target); subprocess.run(["git","add",str(target)],cwd=root,check=True); st=subprocess.run(["git","status","--porcelain"],cwd=root,check=True,capture_output=True,text=True)
    if not st.stdout.strip(): return {"committed":False,"target":str(target)}
    subprocess.run(["git","commit","-m",msg],cwd=root,check=True); return {"committed":True,"target":str(target)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--brand-name",required=True); ap.add_argument("--brand-brief",required=True); ap.add_argument("--website",default=""); ap.add_argument("--monitor-runs",type=int,default=120); ap.add_argument("--model",default="deepseek-chat"); ap.add_argument("--output",default="dist/report"); ap.add_argument("--repo-root",default="."); ap.add_argument("--report-subdir",default="reports/latest"); ap.add_argument("--commit-message",default="chore: update full GEO monitoring dashboard"); ap.add_argument("--commit-report",action="store_true"); args=ap.parse_args()
    n=int(clamp(args.monitor_runs,5,300)); out=Path(args.output).resolve(); root=Path(args.repo_root).resolve(); out.mkdir(parents=True,exist_ok=True); ds=DS(args.model); setup=ds.ask("01_monitoring_plan_generation",setup_prompt(args.brand_name,args.brand_brief,args.website,n),ctype="setup"); plan=normalize_plan(setup,args.brand_name,n); comps=[str(c.get("name")) for c in setup.get("competitors",[]) if isinstance(c,dict) and c.get("name")]
    start=len(ds.calls)
    for item in plan: ds.ask("02_deepseek_geo_probe",probe_prompt(args.brand_name,args.brand_brief,item,comps),question=item["question"],qid=item["query_id"],ctype="monitoring")
    probes=ds.calls[start:]; agg=aggregate(args.brand_name,plan,probes); samples=[c.get("response_json",{}) for c in probes if c.get("success")][:20]; syn=ds.ask("03_aggregate_synthesis",synth_prompt(args.brand_name,setup,agg,samples),ctype="synthesis"); write_outputs(out,args.brand_name,args.brand_brief,setup,agg,syn,ds.calls); info=commit_report(root,out,args.report_subdir,args.commit_message) if args.commit_report else {"committed":False}; print(json.dumps({"brand":args.brand_name,"total_deepseek_calls":len(ds.calls),"monitoring_probes":len(probes),"commit":info},ensure_ascii=False))
if __name__=="__main__": main()
