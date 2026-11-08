"""Current-source supporting figures. Render accepted aggregate estimates only."""
from pathlib import Path
import argparse,csv,datetime,fcntl,hashlib,importlib.util,json,math,os,resource,sys,time
R=Path(__file__).absolute().parents[2]
O=R/"outputs"; W=Path(__file__).absolute().parent
DATA=O/"supporting_analysis_delivery_20261029_v1"
OUT=O/"final_analysis_delivery_20261025_v1/figures/supporting_current_v1"
SOURCE="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3"
CTX=["geography_preop","geography_paired","temporal_preop","temporal_paired"]
COLORS=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#555555","#882255"]
GUARD_SHA="d9a1b50e2fe81b86610d5b3bbd2ed9cb3be2a80f95c26118c188aae71b7ee383"
RULES=["assessment_capacity10","assessment_capacity20","training_capacity10","training_capacity20","training_sensitivity80"]
RULE_LABEL={"assessment_capacity10":"Cohort-ranked top 10%","assessment_capacity20":"Cohort-ranked top 20%","training_capacity10":"Source top-10% cut-off","training_capacity20":"Source top-20% cut-off","training_sensitivity80":"Source 80%-sensitivity cut-off"}
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def save(p,d):
 p=Path(p); t=p.with_suffix(p.suffix+".tmp")
 t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n");t.replace(p)
def num(x):
 if x is None or x=="":return None
 n=float(x);return n if math.isfinite(n) else None
def rows(p):
 with Path(p).open(newline="") as f:return list(csv.DictReader(f))
def context(c):
 return c.replace("geography","Geographical").replace("temporal","Temporal").replace("_preop"," / all preoperative").replace("_paired"," / paired").replace("_all"," / all POD1-eligible").replace("_matched"," / matched POD1-eligible")
def model(m):
 return m.replace("_champion_"," / ").replace("_reference_"," reference / ").replace("__"," / ").replace("_carryforward"," carry-forward").replace("_cal"," calibrated").replace("_raw"," raw").replace("_"," ").replace("preop","Preop").replace("endop","Endop").replace("pod1","POD1")
def write_csv(path,rr):
 keys=list(dict.fromkeys(k for r in rr for k in r))
 with Path(path).open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rr)
def checked_inputs():
 lock=json.loads((W/"SUPPORTING_RENDER_INPUT_LOCK.json").read_text())
 assert sha(__file__)==lock["code_sha256"]
 for p,h in lock["direct_inputs_sha256"].items():assert sha(p)==h,p
 prep=json.loads((DATA/"PLOT_DATA_PREPARATION.json").read_text())
 for p,h in prep["inputs_sha256"].items():assert sha(p)==h,p
 for p,h in prep["outputs_sha256"].items():assert sha(p)==h,p
 d=json.loads((DATA/"PLOT_DATA.json").read_text());assert d["source_sha256"]==SOURCE
 add=json.loads((DATA/"PLOT_DISPLAY_ADDENDA.json").read_text())
 for name in ("adopted_csv","legacy_display_retained","contract","acceptance","report"):
  hkey={"legacy_display_retained":"legacy_display_sha256"}.get(name,name+"_sha256")
  assert sha(add[name])==add[hkey],name
 new=rows(add["adopted_csv"]);assert len(new)==318 and len(d["subgroups"])==44
 assert sorted(set(float(r["percentile"]) for r in new))==[10,50,90]
 assert sorted(set(round(float(r["serum_albumin_g_L"]),2) for r in new))==[33.4,40.,45.12]
 assert sha(O/"v4_development_inputs_20261018/SOURCE_LOCK.json")==SOURCE
 return d,new,lock,prep,add
def interval(ir,metric,identity):
 found=[r for r in ir if r["metric"]==metric and all(str(r.get(k))==str(v) for k,v in identity.items())]
 assert len(found)<=1,(metric,identity,len(found))
 if not found:return None,None,None,0
 r=found[0]
 return num(r["estimate"]),num(r["lower95"]),num(r["upper95"]),int(float(r.get("valid_replicates",0)))
def forest_panel(title,point,ir,metric,keys=("model",),label=None,limits=None,ref=None,point_only=False):
 rr=[]
 for r in point:
  identity={k:r[k] for k in keys}; val=num(r.get(metric))
  est,lo,hi,nrep=interval(ir,metric,identity) if not point_only else (val,None,None,0)
  if not point_only and val is not None:
   assert est is not None and abs(val-est)<1e-10,(title,identity,metric,val,est)
  if lo is not None and hi is not None:assert lo<=hi
  rr.append({"identity":identity,"label":label(r) if label else model(r.get("model","")),
    "value":val,"low":lo,"high":hi,"valid_replicates":nrep})
 return {"kind":"forest","title":title,"metric":metric,"rows":rr,"limits":limits,"reference":ref,"point_only":point_only}
def curve_panel(title,rr,xkey,ykey,labelkey,lowkey=None,highkey=None,log=False,reference=None):
 groups={}
 for r in rr:
  label=str(r[labelkey]);groups.setdefault(label,[]).append({"x":num(r[xkey]),"y":num(r[ykey]),
   "low":num(r.get(lowkey)) if lowkey else None,"high":num(r.get(highkey)) if highkey else None})
 for g in groups.values():g.sort(key=lambda r:r["x"])
 return {"kind":"curves","title":title,"series":groups,"x_label":xkey,"y_label":ykey,
  "log_y":log,"reference":reference}
def build(d,new):
 figs=[]
 def add(name,title,panels,notes,columns=2):
  figs.append(dict(name=name,title=title,panels=panels,notes=notes,columns=columns))
 # Ten locked primary routes; binned calibration is a descriptive display.
 pp=[]
 for c in CTX:
  z=d["main"][c]
  pp.append(curve_panel(context(c),z["calibration"],"mean_predicted","observed_rate","model"))
 add("S01_PRIMARY_CALIBRATION","Calibration of the ten locked primary routes",pp,
  ["Equal-count bins are descriptive; no model updating is performed.",
   "Geographical and temporal evaluations overlap and are not independent replications."])
 z=d["common6"]
 pp=[forest_panel(label,z["point"],z["intervals"],metric,limits=lim,ref=ref) for label,metric,lim,ref in [
  ("AUROC","AUROC",[0,1],.5),("Average precision","AP",[0,1],None),("Brier score","Brier",None,None),("Calibration slope","calibration_slope",None,1)]]
 add("S02_COMMON6_METRICS","Public transport evaluation: two locked six-input RF models",pp,
  ["INSPIRE: 6,043 records / 253 AKI events. Raw and source-development-calibrated predictions are both retained.",
   "Intervals: 2,000 resamples under the accepted public-evaluation protocol. Six-input evidence does not validate the 27/38-input models."])
 p=curve_panel("INSPIRE calibration",z["calibration"],"mean_predicted","observed","model")
 add("S03_COMMON6_CALIBRATION","Six-input model calibration in INSPIRE",[p],
  ["Binned observations are descriptive; no target-outcome recalibration is applied."],1)
 # Retain every locked POD1 model, not only the best observed performer.
 for c,z in d["pod1"].items():
  pp=[forest_panel(label,z["point"],z["intervals"],metric,limits=lim,ref=ref) for label,metric,lim,ref in [
   ("AUROC","AUROC",[0,1],.5),("Average precision","AP",[0,1],None),("AKI captured at 20% capacity","capacity20_sensitivity",[0,1],None)]]
  n=int(float(z["point"][0]["n"]));e=int(float(z["point"][0]["events"]))
  add("S04_POD1_"+c,"POD1 conditional endpoint: "+context(c),pp,
   [f"This evaluation partition: {n:,} records / {e} events. All {len(z['point'])} model routes are shown.",
    "POD1 eligibility is conditional; this is not the original unconditional 7-day endpoint or an exact 24-hour prediction.",
    "Intervals use 2,000 hospital-stratified patient-group resamples; sparse events limit precision."],3)
 for c in dict.fromkeys(r["context"] for r in d["published"]["point"]):
  pr=[r for r in d["published"]["point"] if r["context"]==c];ir=[r for r in d["published"]["intervals"] if r["context"]==c]
  pp=[forest_panel(label,pr,ir,metric,keys=("context","model"),limits=[0,1],ref=.5 if metric=="AUROC" else None) for label,metric in [("AUROC","AUROC"),("Average precision","AP")]]
  add("S05_PUBLISHED_"+c,"Published-model benchmark: "+c.replace("_"," / "),pp,
   [f"Complete-predictor comparison: {int(float(pr[0]['n'])):,} records / {int(float(pr[0]['events']))} events; identical patients within this panel.",
    "Original equations and source-development recalibration are distinct conditions.",
    "Chen lacks the published intercept: its ordinal score supports ranking only, not probabilities or calibrated-risk comparison."])
 # Any-infusion and positive-dose model: do not connect zero to positive exposure.
 q={r["modifier"]:float(r["BH_q_three_dose_interactions"]) for r in d["albumin"]["interactions"]}
 pp=[]
 for modifier,title in [("age65","Age"),("eGFR60","Baseline eGFR"),("baseline_alb","Baseline serum albumin: empirical P10 / P50 / P90")]:
  rr=new if modifier=="baseline_alb" else [r for r in d["albumin"]["curves"] if r["modifier"]==modifier]
  p=curve_panel(f"{title}\nInteraction BH q = {q[modifier]:.3f}",rr,"dose_g","adjusted_OR_vs_reference","display_label","bootstrap_low","bootstrap_high",True,80.)
  p.update(x_label="Non-operative-day cumulative albumin (g)",y_label="Adjusted OR relative to 80 g",zero_separate=True)
  pp.append(p)
 add("S06_ALBUMIN_MODIFIERS","Albumin dose associations across the prespecified modifiers",pp,
  ["Serum displays: 33.4 / 40.0 / 45.12 g/L, chosen from the same 3,669-record covariate distribution after the user's query.",
   "These are continuous-model display values, not three patient groups. Bands reuse the original 2,000 coefficient resamples.",
   "80 g is the dose reference, not a dose threshold. Non-significant interaction does not establish equivalence or a treatment recommendation."],3)
 legacy=[r for r in d["albumin"]["curves"] if r["modifier"]=="baseline_alb"]
 p=curve_panel("Legacy settings retained for traceability",legacy,"dose_g","adjusted_OR_vs_reference","display_label","bootstrap_low","bootstrap_high",True,80.)
 p.update(x_label="Non-operative-day cumulative albumin (g)",y_label="Adjusted OR relative to 80 g",zero_separate=True)
 add("S07_ALBUMIN_LEGACY_DISPLAY","Legacy 25 / 35 / 45 g/L display: supplementary traceability only",[p],
  ["These were numerical display settings (35 plus/minus 10), without a documented basis for the full three-value selection.",
   "The original model and intervals are retained. Use the empirical-quantile display as the main serum-albumin panel."],1)
 # All 253 events, including patients who were never selected, remain in each window denominator.
 z=d["recorded_window"]; rr=[]
 for r in z["point"]:
  for hours in (12,24,48):
   metric=f"all_AKI_detected_atleast{hours}h_fraction"
   est,lo,hi,nrep=interval(z["intervals"],metric,{"rule":r["rule"]})
   assert abs(float(r[metric])-est)<1e-12
   rr.append({"rule":r["rule"],"hours":hours,"fraction":est,"lower":lo,"upper":hi})
 p=curve_panel("Captured at least this long before first recorded SCr criterion",rr,"hours","fraction","rule","lower","upper")
 p.update(x_label="Minimum recorded interval (hours)",y_label="Fraction of all 253 AKI events",interval_points=True)
 pp=[p,forest_panel("Review workload per 1,000",z["point"],z["intervals"],"reviewed_per1000",keys=("rule",),label=lambda r:r["rule"].replace("__","\n").replace("_"," "),limits=[0,1000]),
  forest_panel("Positive predictive value",z["point"],z["intervals"],"PPV",keys=("rule",),label=lambda r:r["rule"].replace("__","\n").replace("_"," "),limits=[0,1])]
 add("S08_RECORDED_WINDOWS","Fixed surgical-time prediction and the recorded SCr interval",pp,
  ["All 6 rules are retained. Prediction time is surgical start; endpoint time is the first recorded qualifying SCr measurement.",
   "This is neither biological onset nor result-release/actual-alert time, and is not a dynamic multihorizon model.",
   "Source-locked thresholds and retrospective cohort-ranked capacities are separately labelled."],3)
 # Stage capture uses any-AKI risk and the whole-population list, never stage-only AUROC.
 for cap in (10,20):
  pp=[]
  for c in CTX:
   pr=[];ir=[]
   for stage in ("stage1","stage2","stage3","stage2_3"):
    z=d["subgroups"][c+"/"+stage]
    pr.extend(dict(r,stage=stage) for r in z["point"])
    ir.extend(dict(r,stage=stage) for r in z["intervals"])
   metric=f"capacity{cap}_target_stage_sensitivity"
   pp.append(forest_panel(context(c),pr,ir,metric,keys=("stage","model","level"),
    label=lambda r:f"{r['stage'].replace('stage','Stage ').replace('_','-')} ({int(float(r['target_stage_events']))} events)\n{model(r['model'])}",limits=[0,1]))
  add(f"S09_STAGE_CAPTURE_CAP{cap}",f"AKI-stage capture within the whole-population top {cap}%",pp,
   ["Stage 1, 2, 3 and combined 2-3 events are captured by the original any-AKI risk; a stage-specific model is not fitted.",
    "Stage 2-3 overlaps stages 2 and 3. Zero-event cells are not estimable; intervals do not establish interaction or equivalence."])
 # Every baseline stratum plus ALT observation/missingness, all frozen full-route winners.
 order=("hospital","age","sex","diabetes","tumor_site","egfr","ALT_observed_missing")
 for c in CTX:
  pr=[];ir=[]
  for factor in order:
   z=d["subgroups"][c+"/"+factor]
   pr.extend(dict(r,factor=factor) for r in z["point"])
   ir.extend(dict(r,factor=factor) for r in z["intervals"])
  lab=lambda r:f"{r['factor']}: {r['level']} (n={int(float(r['n']))}, AKI={int(float(r['events']))})\n{model(r['model'])}"
  pp=[forest_panel(label,pr,ir,metric,keys=("factor","model","level"),label=lab,limits=[0,1],ref=.5 if metric=="AUROC" else None) for label,metric in [
   ("Within-stratum AUROC","AUROC"),("AKI captured by the whole-population top 20%","capacity20_sensitivity")]]
  add("S10_STRATA_"+c,"Stratified performance: "+context(c),pp,
   ["All seven stratification factors are retained; no subgroup-specific 20% list is recomputed.",
    "ALT categories denote observed versus missing values, not normal/elevated ALT. One low eGFR does not establish CKD.",
    "Undefined estimates remain labelled. These descriptive subgroup intervals are not tests of effect modification; source category codes are not relabelled speculatively."])
 # Exact burden summaries; not invented intervals.
 for c in CTX:
  mm={r["model"] for r in d["routes"] if r["context"]==c}
  pr=[r for r in d["workflow"]["burden"] if r["context"]==c and r["model"] in mm]
  pr.sort(key=lambda r:(r["model"],RULES.index(r["rule"])))
  lab=lambda r:model(r["model"])+"\n"+RULE_LABEL[r["rule"]]
  pp=[forest_panel(label,pr,[],metric,keys=("context","model","rule"),label=lab,point_only=True) for label,metric in [
   ("Reviewed per 1,000","reviewed_per1000"),("AKI captured per 1,000","captured_per1000"),("False alerts per 1,000","false_alerts_per1000")]]
  add("S11_WORKLOAD_"+c,"Operating burden: "+context(c),pp,
   ["Exact observed point counts are shown. All five prespecified operating rules and every locked primary route are retained.",
    "Retrospective top 10%/20% capacity lists differ from fixed source-derived cut-offs.",
    "The complete review-time / extra-test assumption grid and its conditional intervals are provided in the accompanying tables."],3)
 full={c:next(r["model"] for r in d["routes"] if r["context"]==c and r["stage"]=="preop" and r["branch"]=="full") for c in CTX}
 sel=[r for r in d["workflow"]["assumed_effect"] if r["model"]==full[r["context"]]]
 vmax=max(abs(float(r["hypothetical_events_averted_per1000"])) for r in sel)
 for c in CTX:
  pp=[]
  for rule in RULES:
   rr=[r for r in sel if r["context"]==c and r["rule"]==rule]
   xs=sorted(set(float(r["assumed_RRR"]) for r in rr));ys=sorted(set(float(r["assumed_uptake"]) for r in rr))
   assert len(rr)==len(xs)*len(ys)==15
   pp.append({"kind":"heatmap","title":RULE_LABEL[rule],"x":xs,"y":ys,"rows":rr,"vmax":vmax,
    "x_label":"Assumed relative risk reduction","y_label":"Assumed uptake"})
  add("S12_HYPOTHETICAL_"+c,"Assumption grid: "+context(c),pp,
   ["Display model selected solely by locked full preoperative route; all models and all grid values remain in the companion tables.",
    "Negative, zero and positive assumptions are retained. These are hypothetical events averted per 1,000, not observed treatment effects.",
    "Conditional sampling intervals do not include uncertainty in the assumed intervention effect or uptake."],3)
 assert len(figs)==28,len(figs)
 assert len({x["name"] for x in figs})==len(figs)
 return figs
def resource_admission():
 path=R/"work/albumin_reference_display_20261029_v1/render_current_rcs.py"
 assert sha(path)==GUARD_SHA
 spec=importlib.util.spec_from_file_location("accepted_joint_resource_guard",path)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 return mod.resource_check()
def render(figspec,dest):
 import matplotlib.pyplot as plt
 from matplotlib.colors import TwoSlopeNorm
 import matplotlib.ticker as ticker
 import numpy as np
 panels=figspec["panels"];cols=min(figspec["columns"],len(panels));nrows=math.ceil(len(panels)/cols)
 maxn=max([len(p.get("rows",[])) for p in panels if p["kind"]=="forest"]+[0])
 rowheight=max(4.2,1.6+maxn*.33)
 fig,axes=plt.subplots(nrows,cols,figsize=(max(8.0,5.9*cols),nrows*rowheight+1.9),squeeze=False)
 for idx,p in enumerate(panels):
  ax=axes.flat[idx];ax.set_title(p["title"],loc="left",fontsize=10.5,pad=13)
  kind=p["kind"]
  if kind=="forest":
   rr=p["rows"]
   for i,r in enumerate(rr):
    value=r["value"];lo=r["low"];hi=r["high"]
    if value is None:
     ax.text(.02,i,"Not estimable",transform=ax.get_yaxis_transform(),fontsize=8,color="#777777")
    else:
     color=COLORS[i%len(COLORS)]
     if lo is not None and hi is not None:ax.hlines(i,lo,hi,color=color,lw=1.5)
     ax.plot([value],[i],"o",color=color,ms=4)
   ax.set_yticks(range(len(rr)),[r["label"] for r in rr],fontsize=7.6)
   ax.set_ylim(len(rr)-.5,-.7)
   if p["limits"]:ax.set_xlim(p["limits"])
   if p["reference"] is not None:ax.axvline(p["reference"],color="#888888",lw=.7,ls=":")
   ax.grid(axis="x",alpha=.16);ax.set_xlabel(p["title"].split("\n")[0],fontsize=9)
  elif kind=="curves":
   top=0.
   for i,(label,rr) in enumerate(p["series"].items()):
    color=COLORS[i%len(COLORS)];lab=model(label) if not "Baseline" in label else label
    positive=[r for r in rr if not p.get("zero_separate") or r["x"]>0]
    xx=[r["x"] for r in positive];yy=[r["y"] for r in positive]
    top=max(top,max(xx or [0]),max(yy or [0]))
    if p.get("interval_points"):
     for r in positive:
      if r["low"] is not None:ax.vlines(r["x"],r["low"],r["high"],color=color,lw=1)
     ax.plot(xx,yy,marker="o",ms=4,color=color,lw=1.2,ls=("-","--",":")[i%3],label=lab)
    else:
     ax.plot(xx,yy,color=color,lw=1.6,marker="o" if p["reference"] is None else None,ms=3,label=lab)
     if all(r["low"] is not None for r in positive):
      ax.fill_between(xx,[r["low"] for r in positive],[r["high"] for r in positive],color=color,alpha=.13,lw=0)
    if p.get("zero_separate"):
     for r in rr:
      if r["x"]==0:
       ax.vlines(0,r["low"],r["high"],color=color,lw=1.5);ax.plot(0,r["y"],"s",color=color,ms=4)
   if p["reference"] is None and not p.get("interval_points"):
    top=min(1,max(.1,top*1.08));ax.plot([0,top],[0,top],color="#888888",lw=.7,ls=":")
    ax.set_xlim(0,top);ax.set_ylim(0,top);ax.set_xlabel("Mean predicted risk");ax.set_ylabel("Observed event fraction")
   else:
    ax.set_xlabel(p["x_label"],fontsize=9);ax.set_ylabel(p["y_label"],fontsize=9)
   if p.get("interval_points"):
    ax.set_xticks([12,24,48]);ax.set_ylim(0,1)
   if p["log_y"]:
    ax.set_yscale("log");ax.yaxis.set_major_formatter(ticker.ScalarFormatter());ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.axhline(1,color="#888888",lw=.7,ls=":");ax.axvline(p["reference"],color="#888888",lw=.7,ls="--")
    ax.set_xlim(-6,215)
   ax.grid(axis="y",alpha=.15);ax.legend(fontsize=7.1,frameon=False,loc="best")
  elif kind=="heatmap":
   xx=p["x"];yy=p["y"];ar=np.full((len(yy),len(xx)),np.nan)
   for r in p["rows"]:
    if str(r["probability_assumption_compatible"]).lower()=="true":
     ar[yy.index(float(r["assumed_uptake"])),xx.index(float(r["assumed_RRR"]))]=float(r["hypothetical_events_averted_per1000"])
   im=ax.imshow(ar,origin="lower",aspect="auto",cmap="PuOr",norm=TwoSlopeNorm(vmin=-p["vmax"],vcenter=0,vmax=p["vmax"]))
   ax.set_xticks(range(len(xx)),[f"{v:.0%}" for v in xx]);ax.set_yticks(range(len(yy)),[f"{v:.0%}" for v in yy])
   for iy in range(len(yy)):
    for ix in range(len(xx)):
     value=ar[iy,ix];ax.text(ix,iy,f"{value:.1f}" if math.isfinite(value) else "N/A",ha="center",va="center",fontsize=9,color="white" if abs(value)>p["vmax"]*.5 else "black")
   ax.set_xlabel(p["x_label"]);ax.set_ylabel(p["y_label"]);fig.colorbar(im,ax=ax,fraction=.045,pad=.03)
 for ax in list(axes.flat)[len(panels):]:ax.set_visible(False)
 fig.suptitle(figspec["title"],x=.03,y=.985,ha="left",fontsize=14,fontweight="bold")
 fig.tight_layout(rect=(.015,.14,.99,.945),w_pad=2.2,h_pad=2.7)
 for i,line in enumerate(figspec["notes"]):fig.text(.03,.105-i*.026,line,fontsize=8.5,ha="left",va="top")
 files={}
 for ext in ("png","pdf","svg"):
  p=dest/(figspec["name"]+"."+ext);fig.savefig(p,dpi=220,facecolor="white");files[str(p)]=sha(p)
 plt.close(fig)
 return files
def write_tables(d,new):
 tables=OUT/"tables";tables.mkdir(exist_ok=True)
 write_csv(tables/"ALL_WORKFLOW_BURDEN.csv",d["workflow"]["burden"])
 write_csv(tables/"ALL_WORKFLOW_TIME_TEST_GRID.csv",d["workflow"]["workload"])
 write_csv(tables/"ALL_HYPOTHETICAL_EFFECT_GRID.csv",d["workflow"]["assumed_effect"])
 write_csv(tables/"ALBUMIN_QUANTILE_DISPLAY.csv",new)
 write_csv(tables/"ALBUMIN_ORIGINAL_ALL_MODIFIERS.csv",d["albumin"]["curves"])
 write_csv(tables/"ALL_44_SUBGROUP_FULL_ROUTE_POINTS.csv",[dict(r,task=task) for task,z in d["subgroups"].items() for r in z["point"]])
 write_csv(tables/"ALL_44_SUBGROUP_FULL_ROUTE_INTERVALS.csv",[dict(r,task=task) for task,z in d["subgroups"].items() for r in z["intervals"]])
 save(tables/"FULL_NATIVE_SUBGROUP_TABLE_INDEX.json",{k:{a:z[a] for a in ("all_model_source_point","all_model_source_intervals","native_model_count")} for k,z in d["subgroups"].items()})
 return {str(p):sha(p) for p in tables.iterdir() if p.is_file()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--run",action="store_true");args=ap.parse_args()
 started=time.monotonic();d,new,lock,prep,add=checked_inputs();figs=build(d,new)
 summary={"time":now(),"source_sha256":SOURCE,"code_sha256":sha(__file__),"figure_count":len(figs),
  "native_source_files_verified":len(prep["inputs_sha256"]),"all_44_subgroup_tasks_covered":len(d["subgroups"]),
  "all_22_POD1_models_covered":sum(len(z["point"]) for z in d["pod1"].values()),"all_20_published_comparison_conditions_covered":len(d["published"]["point"]),
  "all_6_recorded_window_rules_covered":len(d["recorded_window"]["point"]),"planned_figures":[{"name":f["name"],"panels":len(f["panels"])} for f in figs],
  "new_fits":0,"new_bootstrap":0,"final_visual_review":False,"final_scientific_review":False}
 if not args.run:
  summary.update(status="PREPARED_FROM_REAL_CURRENT_RESULTS_NOT_RENDERED",seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  save(DATA/"SUPPORTING_RENDER_PREPARATION.json",summary)
  print(json.dumps({k:v for k,v in summary.items() if k!="planned_figures"}));return
 admission=resource_admission()
 save(DATA/"SUPPORTING_RENDER_RESOURCE_STATUS.json",admission)
 if not admission["accepted"]:
  print(json.dumps({"status":"WAIT_JOINT_RESERVED_HEADROOM","residual_bytes":admission["residual"]}));return
 resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2))
 resource.setrlimit(resource.RLIMIT_CPU,(600,600))
 OUT.mkdir(parents=True,exist_ok=True,mode=0o700)
 for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[k]="1"
 os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["MPLCONFIGDIR"]=str(OUT/"mpl_cache");os.environ["MPLBACKEND"]="Agg"
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
 with (OUT/"RENDER.lock").open("a") as lk:
  fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
  inputs_sha=sha(W/"SUPPORTING_RENDER_INPUT_LOCK.json");receipts=[]
  for figspec in figs:
   rp=OUT/(figspec["name"]+".json")
   if rp.exists():
    v=json.loads(rp.read_text());assert v["input_lock_sha256"]==inputs_sha and v["code_sha256"]==sha(__file__)
    for p,h in v["files_sha256"].items():assert sha(p)==h
   else:
    files=render(figspec,OUT);v={"name":figspec["name"],"status":"RENDERED_PENDING_PIXEL_REVIEW","time":now(),
     "input_lock_sha256":inputs_sha,"code_sha256":sha(__file__),"files_sha256":files,"notes":figspec["notes"],"actual_pixel_review":False}
    save(rp,v)
   receipts.append({"path":str(rp),"sha256":sha(rp)})
  tables=write_tables(d,new)
  for p,h in lock["direct_inputs_sha256"].items():assert sha(p)==h,p
  summary.update(status="PASS_CURRENT_SUPPORTING_RENDER_PENDING_ACTUAL_PIXEL_REVIEW",receipt_files=receipts,tables_sha256=tables,
   seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,rendered=True)
  save(DATA/"SUPPORTING_RENDER_ACCEPTANCE.json",summary)
  print(json.dumps({k:v for k,v in summary.items() if k not in ("planned_figures","receipt_files","tables_sha256")}))
if __name__=="__main__":main()
