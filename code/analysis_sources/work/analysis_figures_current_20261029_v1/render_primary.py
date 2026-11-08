"""Current-source analysis figures; render accepted tables only, never fit models."""
from pathlib import Path
import json,hashlib,math
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
O=R/"outputs/final_analysis_delivery_20261025_v1"; F=O/"figures";F.mkdir(mode=0o700,exist_ok=True)
SOURCE_SHA="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3"
sources={};figure_files={};plotdata={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p); sources[str(p)]=sha(p); return json.loads(p.read_text())
def csv(p):
 p=Path(p); sources[str(p)]=sha(p);return pd.read_csv(p,float_precision="round_trip")
B=R/"outputs/v4_development_inputs_20261018";assert sha(B/"SOURCE_LOCK.json")==SOURCE_SHA
routes=read(R/"outputs/locked_risk_curves_20261024_v1/ROUTES.json")
evals={c:read(R/"outputs/formal_evaluation_20261023_v1"/c/"EVALUATION.json") for c in ["geography_preop","temporal_preop","geography_paired","temporal_paired"]}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,"axes.labelsize":9,"legend.fontsize":8,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.15,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
COL={"geography":"#0072B2","temporal":"#D55E00"}; rows=[]
for key,ref in sorted(routes.items()):
 assert sha(ref["path"])==ref["sha256"]; m=read(ref["path"]);t=m["identity"];ctx=t["line"]+"_"+t["cohort"];name="_".join(t[k] for k in ["stage","branch","role","family"])
 rows.append({"context":ctx,"route":key,"family":m["family"],"inputs":m["actual_input_count"],"model":name,**evals[ctx]["point"][name]})
table=pd.DataFrame(rows);table.to_csv(O/"TABLE_1_ALL_PRIMARY_MODELS.csv",index=False)
def finish(fig,name,foot=None):
 if foot:fig.text(.5,.008,foot,ha="center",va="bottom",fontsize=7)
 for ext in ["png","pdf","svg"]:
  p=F/(name+"."+ext);fig.savefig(p,dpi=300,bbox_inches="tight");figure_files[str(p)]=sha(p)
 plt.close(fig)
def ci(ctx,model,metric,difference=False):
 z=[z for z in evals[ctx]["paired_differences" if difference else "intervals"] if z["model"]==model and z["metric"]==metric];assert len(z)==1;return z[0]
def point(ax,x,y,lo,hi,color,marker="o",label=None):
 ax.plot([lo,hi],[y,y],color=color,lw=1.5);ax.plot([lo,lo],[y-.05,y+.05],color=color,lw=1);ax.plot([hi,hi],[y-.05,y+.05],color=color,lw=1);ax.scatter([x],[y],s=28,color=color,marker=marker,label=label,zorder=3)
# Audited current denominators. No onset-time or intervention inference.
meta={x:csv(B/(x+"_metadata.csv")) for x in ["preop","paired"]}
assert (len(meta["preop"]),int(meta["preop"].y.sum()))==(3710,205)
assert (len(meta["paired"]),int(meta["paired"].y.sum()))==(2428,138)
fig,ax=plt.subplots(figsize=(11.5,5.9));ax.axis("off")
def box(x,y,w,h,txt,color="#edf5fb"):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.015",facecolor=color,edgecolor="#aac0cb",transform=ax.transAxes))
 ax.text(x+w/2,y+h/2,txt,ha="center",va="center",transform=ax.transAxes,linespacing=1.6)
a=meta["preop"];b=meta["paired"]
box(.02,.65,.28,.27,f"Eligible main cohort\n{len(a):,} operations / {int(a.y.sum())} AKI\n{a.group_hash.nunique():,} patient groups")
box(.365,.65,.28,.27,f"Paired information cohort\n{len(b):,} operations / {int(b.y.sum())} AKI\n{b.group_hash.nunique():,} patient groups")
box(.72,.65,.25,.27,"Separate strict POD1 landmark\n2,947 operations / 39 AKI\nConditional postoperative estimand","#faf3e8")
box(.02,.2,.28,.27,"Preoperative assessment\n9 compact / 27 full inputs\nSelected: geography27, temporal24")
box(.365,.2,.28,.27,"Paired preoperative → end of operation\n27 → 38 inputs\nSame-patient comparison")
box(.72,.2,.25,.27,"Separate public evaluation\nINSPIRE6,043 / 253 AKI\nNew common6 source models","#edf7f1")
for x1,y1,x2,y2 in [(.31,.78,.35,.78),(.16,.63,.16,.49),(.50,.63,.50,.49)]:
 ax.annotate("",xy=(x2,y2),xytext=(x1,y1),xycoords="axes fraction",arrowprops={"arrowstyle":"->","color":"#617984"})
fig.suptitle("Current cohorts and information available at prediction",fontsize=13)
finish(fig,"Figure_1_COHORT_AND_INFORMATION","Outcome: recorded postoperative7-day AKI. Geographic and temporal evaluation sets overlap; public models use a separate six-variable representation.")
# Locked primary captured AKI with source-specified capacity and bootstrap intervals.
fig,axs=plt.subplots(1,3,figsize=(12.2,4.7),gridspec_kw={"width_ratios":[1,1,1.15]})
for j,line in enumerate(["geography","temporal"]):
 z=table[table.context.eq(line+"_preop")].set_index("route")
 for i,branch in enumerate(["compact","full","selected"]):
  rr=z.loc[line+"_preop_preop_"+branch];qq=ci(line+"_preop",rr.model,"capacity20_sensitivity")
  axs[j].bar(i,rr.capacity20_sensitivity*100,color=COL[line],width=.6,alpha=.85)
  axs[j].errorbar(i,qq["estimate"]*100,yerr=[[100*(qq["estimate"]-qq["lower95"])],[100*(qq["upper95"]-qq["estimate"])]],fmt="none",color="#333333",capsize=3)
  axs[j].text(i,qq["upper95"]*100+2,f"{int(rr.capacity20_captured_n)} captured",ha="center",fontsize=7.5)
 axs[j].set_xticks(range(3),[f"{b.title()}\n{int(z.loc[line+'_preop_preop_'+b,'inputs'])} inputs" for b in ["compact","full","selected"]])
 axs[j].set_ylim(0,100);axs[j].set_ylabel("AKI captured at20% review (%)");axs[j].set_title(chr(65+j)+"  "+line.title()+" preoperative")
for j,line in enumerate(["geography","temporal"]):
 z=table[table.context.eq(line+"_paired")].set_index("route");end=z.loc[line+"_paired_endop_full","model"];pre=z.loc[line+"_paired_preop_full","model"]
 qq=ci(line+"_paired",end+" MINUS "+pre,"capacity20_captured_per1000",True)
 point(axs[2],qq["estimate"],j,qq["lower95"],qq["upper95"],COL[line])
 axs[2].text(qq["estimate"],j+.14,f"{qq['estimate']:.2f} [{qq['lower95']:.2f}, {qq['upper95']:.2f}]",ha="center",fontsize=7.5)
axs[2].axvline(0,color="#777777",ls="--",lw=1);axs[2].set_yticks([0,1],["Geographic","Temporal"]);axs[2].set_ylim(-.4,1.45);axs[2].set_title("C  Paired end-of-operation increment");axs[2].set_xlabel("Additional AKI captured /1000\n20% review;95% bootstrap interval")
fig.tight_layout(rect=(0,.06,1,1));finish(fig,"Figure_2_PRIMARY_INFORMATION_RESULTS","Frozen models and policies;2000 hospital-stratified patient-group bootstraps. Capacity is an evaluation ranking rule, distinct from a fixed source risk threshold.")
# Exploratory ALT/no-overlap supports retain both10% and20% review.
S=R/"outputs/process_resolution_20261025_v1"
a=csv(S/"ALT_SUPPORT.csv");n=csv(S/"NONOVERLAP_SUPPORT.csv");mc=csv(S/"SELECTIVE_VS_RANDOM_MC.csv");cal=csv(S/"CALIBRATION_UPDATE_SUPPORT.csv")
diag=read(S/"POLICY_CALIBRATION_DIRECTION_DIAGNOSTIC.json")
fig,axs=plt.subplots(1,2,figsize=(11,5.0))
for j,cap in enumerate([10,20]):
 for k,line in enumerate(["geography","temporal"]):
  rr=a[a.context.eq(line+"_paired")&a.comparison.eq("change_in_gain_omit_ALT_minus_control")&a.metric.eq(f"capacity{cap}_captured_per1000")];assert len(rr)==1;rr=rr.iloc[0];i=2*j+k
  point(axs[0],rr.estimate,i,rr.lower,rr.upper,COL[line]);axs[0].text(rr.upper+.3,i,f"{rr.estimate:.2f}",va="center",fontsize=8)
axs[0].set_yticks(range(4),["Geographic10%","Temporal10%","Geographic20%","Temporal20%"]);axs[0].invert_yaxis();axs[0].set_title("A  ALT removal: change in paired increment");axs[0].set_xlabel("No ALT minus matched procedure /1000\nExploratory95% bootstrap interval")
ctx="temporal_paired";models=table[table.context.eq(ctx)].set_index("route");contrast=models.loc["temporal_paired_endop_full","model"]+" MINUS "+models.loc["temporal_paired_preop_full","model"]
for j,cap in enumerate([10,20]):
 rr=n[n.context.eq(ctx)&n.comparison.eq(contrast)&n.metric.eq(f"capacity{cap}_captured_per1000")];assert len(rr)==1;rr=rr.iloc[0]
 point(axs[1],rr.estimate,j,rr.lower,rr.upper,COL["temporal"]);axs[1].text(rr.upper+.3,j,f"{rr.estimate:.2f}",va="center",fontsize=8)
nn=n[n.context.eq(ctx)&n.comparison.eq(models.loc["temporal_paired_endop_full","model"])]
nnn=int(nn[nn.metric.eq("n")].estimate.iloc[0]);ne=int(nn[nn.metric.eq("events")].estimate.iloc[0])
axs[1].set_yticks([0,1],["10% review","20% review"]);axs[1].set_ylim(-.4,1.4);axs[1].set_title(f"B  Temporal nonoverlap: {nnn} operations / {ne} AKI");axs[1].set_xlabel("Paired end-of-operation increment /1000\nExploratory95% bootstrap interval")
for ax in axs:ax.axvline(0,color="#777777",ls="--",lw=1)
fig.tight_layout(rect=(0,.07,1,1));finish(fig,"Figure_3_ALT_AND_NONOVERLAP_SUPPORT","Posthoc support calculations preserve current source and frozen base models. Overlapping patient groups removed before evaluation; no multiplicity adjustment.")
# All16 planned Monte Carlo allocation contexts; not patient confidence intervals.
fig,axs=plt.subplots(2,2,figsize=(11.5,8.4),squeeze=False)
for ax,ctx in zip(axs.flat,["geography_preop","temporal_preop","geography_paired","temporal_paired"]):
 z=mc[mc.context.eq(ctx)].sort_values(["upgrade_fraction","review_fraction"]).reset_index(drop=True)
 for i,rr in enumerate(z.itertuples()):
  scale=1000/rr.n
  point(ax,rr.random_mean_captured_n*scale,i,rr.random_allocation_q025*scale,rr.random_allocation_q975*scale,"#777777",label="Random: mean and allocation range" if i==0 else None)
  ax.scatter(rr.selective_captured_n*scale,i,color=COL[ctx.split("_")[0]],marker="D",s=30,label="Selective" if i==0 else None,zorder=4)
 ax.set_yticks(range(len(z)),[f"{int(x.upgrade_fraction*100)}% upgrade / {int(x.review_fraction*100)}% review" for x in z.itertuples()])
 ax.invert_yaxis();ax.set_xlabel("AKI captured /1000 operations");ax.set_title(ctx.replace("_"," ").title());ax.legend(loc="best",fontsize=7)
fig.suptitle("All planned conditional random-allocation comparisons",fontsize=12)
fig.tight_layout(rect=(0,.075,1,.95))
finish(fig,"Supplement_RANDOM_ALLOCATION_ALL16","Grey2.5–97.5% allocation range is conditional Monte Carlo variability, not a patient CI or P value.\nGeography-paired50% comparator has a negative frozen development calibration slope; contrast includes that calibration behaviour.")
# Both permitted target-label updating procedures, all12 routes, no deployed replacement.
z=cal[cal.metric.eq("Brier")&cal.comparison.isin(["offset_MAP_MINUS_original_V1","logistic_MAP_MINUS_original_V1"])].copy()
keys=sorted(set(zip(z.context,z.route)));assert len(keys)==12
fig,ax=plt.subplots(figsize=(11,7.6))
for i,key in enumerate(keys):
 for typ,delta,color,marker in [("offset_MAP_MINUS_original_V1",-.12,"#0072B2","o"),("logistic_MAP_MINUS_original_V1",.12,"#D55E00","s")]:
  rr=z[z.context.eq(key[0])&z.route.eq(key[1])&z.comparison.eq(typ)];assert len(rr)==1;rr=rr.iloc[0]
  point(ax,rr.estimate,i+delta,rr.lower,rr.upper,color,marker,label=typ.split("_")[0].title() if i==0 else None)
ax.axvline(0,color="#777777",ls="--",lw=1)
ax.set_yticks(range(len(keys)),[a.replace("geography","Geo.").replace("temporal","Time").replace("_"," ")+" | "+b.replace("_champion","").replace("_"," ") for a,b in keys]);ax.invert_yaxis()
ax.set_xlabel("Brier difference: cross-fitted target update minus frozen source model\nNegative values indicate lower Brier score;2000-bootstrap interval")
ax.set_title("Posthoc target-local updating: offset and logistic MAP");ax.legend()
fig.tight_layout(rect=(0,.07,1,1));finish(fig,"Figure_4_POSTHOC_CALIBRATION","Original3 patient-group folds; each update uses other target-fold labels. Adaptation-procedure evaluation, not fresh independent validation; exploratory intervals.")
# Complete primary table as compact discrimination/probability-error forests.
fig,axs=plt.subplots(1,3,figsize=(12,7.2),sharey=True)
for i,rr in table.iterrows():
 for ax,metric in zip(axs,["AUROC","AP","Brier"]):
  qq=ci(rr.context,rr.model,metric);point(ax,qq["estimate"],i,qq["lower95"],qq["upper95"],COL[rr.context.split("_")[0]])
for ax,metric in zip(axs,["AUROC","AP","Brier"]):ax.set_xlabel(metric+" (95% bootstrap interval)");ax.grid(axis="y",alpha=.1)
axs[0].set_yticks(range(len(table)),[r.route.replace("geography","Geo.").replace("temporal","Time").replace("_preop_preop"," preop").replace("_paired_"," paired ").replace("_"," ")+f" | {int(r.inputs)} inputs, {r.family}" for r in table.itertuples()]);axs[0].invert_yaxis()
fig.suptitle("All10 locked primary model routes",fontsize=12);fig.tight_layout(rect=(0,.04,1,.96))
finish(fig,"Figure_5_ALL_PRIMARY_MODEL_METRICS","Different cohorts and development lines retained; source-specific model choices were locked before this corrected evaluation. No model is selected by this figure.")
notes="""# Current-source analysis figure notes
These are analysis-review graphics and source tables, not manuscript prose or scientific acceptance.
Figure1: verified eligible cohorts; separate POD1 and public estimands. No inferred onset or intervention effect.
Figure2: locked route-level counts and hospital-stratified patient bootstrap intervals; selected feature counts come from each MODEL_LOCK.
Figure3: exploratory ALT/no-overlap supports at both planned capacities. Confidence intervals are not adjusted for multiple comparisons.
Random-allocation supplement: all16 planned contexts. The conditional allocation spread is distinct from patient sampling uncertainty. Geography-paired50% random comparator's negative frozen calibrator is retained and disclosed; the contrast does not isolate an intrinsic selective-gating effect.
Figure4: all12 target-local routes and both originally allowed MAP updating procedures. Target labels from other folds are used; base models remain fixed. No new independently validated deployment model is asserted.
Figure5: all10 locked primary routes, with AUROC/AP/Brier intervals, including adverse or imprecise comparisons.
SHAP: development-reference whole-pipeline magnitudes; Monte Carlo error is separate, not a confidence interval for mean absolute contribution; no causal interpretation.
Risk curves: development-reference partial dependence; shaded10–90% between-case prediction spread is not a confidence band; correlated-feature changes can be uncommon combinations.
LLM comparative figures and other final integrated-delivery evidence remain pending. Actual image inspection is required after rendering.
"""
(O/"FIGURE_NOTES_ZH.md").write_text(notes)
payload={"status":"RENDERED_current_source_primary_analysis_figures_visual_review_pending","current_input_sha256":SOURCE_SHA,"figures_sha256":figure_files,"sources_sha256":sources,"code_sha256":sha(__file__),"models_refitted":False,"evaluation_results_used_to_reselect":False,"LLM_figures_complete":False,"scientific_review_complete":False,"whole_project_complete":False}
(O/"PRIMARY_FIGURE_RENDER_ACCEPTANCE.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
print("RENDERED_PRIMARY",len(figure_files),flush=True)
