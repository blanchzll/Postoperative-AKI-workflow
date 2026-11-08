"""Python-only plots of saved new numerical results; no analysis rerun."""
from pathlib import Path
import os,json,hashlib
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
A=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot/outputs/novelty_strengthening_20261101_v1");F=A/"figures"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],"font.size":8,"axes.titlesize":9,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none"})
g=pd.read_csv(A/"resource_comparison/RESOURCE_BUDGET_GRID.csv");b=pd.read_csv(A/"resource_comparison/BREAK_EVEN_BOUNDARIES.csv");renders=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(fig,name,rows):
 fig.savefig(F/(name+".pdf"),format="pdf")
 fig.savefig(F/(name+".svg"),format="svg")
 fig.savefig(F/(name+".png"),format="png",dpi=600)
 fig.savefig(F/(name+".tiff"),format="tiff",dpi=600,pil_kwargs={"compression":"tiff_lzw"})
 plt.close(fig);im=Image.open(F/(name+".png")).convert("RGB");im.thumbnail((1700,1600));im.save(F/(name+"_preview.jpg"),quality=92);im.convert("L").save(F/(name+"_grayscale_preview.jpg"),quality=92)
 pd.DataFrame(rows).to_csv(F/(name+"_SOURCE_DATA.csv"),index=False)
 renders.append({"figure":name,"files":{p.name:sha(p) for p in F.glob(name+"*")}})
ids=["geography_9v27_champ","temporal_9v27_champ","geography_27v38_champ","temporal_27v38_champ"]
titles=["Geographic: 9 to 27 inputs","Temporal: 9 to 27 inputs","Geographic: preoperative to end-of-surgery","Temporal: preoperative to end-of-surgery"]
fig,axes=plt.subplots(2,2,figsize=(7.2,6.6));fig.subplots_adjust(left=.115,right=.98,bottom=.15,top=.89,hspace=.48,wspace=.34);rows=[]
for i,(ax,n,t) in enumerate(zip(axes.ravel(),ids,titles)):
 d=g[(g.comparison_id==n)&(g["mode"]=="calibrated")&(g.budget_percent==20)&(g.feasible_at_point==True)].sort_values("extra_cost_per_person_review_equivalents")
 x=d.extra_cost_per_person_review_equivalents.to_numpy()*1000;col="#0072B2" if i%2==0 else "#D55E00"
 ax.plot(x,d.delta_captured_per1000.to_numpy(),color=col,marker="o",markersize=2.5,lw=1.3)
 ax.fill_between(x,d.delta_captured_per1000_lower95.to_numpy(),d.delta_captured_per1000_upper95.to_numpy(),color=col,alpha=.16)
 ax.axhline(0,c=".3",lw=.7,ls="--");ax.set_xlim(0,200);ax.set_xticks([0,50,100,150,200]);ax.set_title(t,loc="left",pad=14);ax.text(-.16,1.12,chr(65+i),transform=ax.transAxes,weight="bold",fontsize=10)
 ax.text(.02,.04,f"n = {int(d.n.iloc[0]):,}; events = {int(d.events.iloc[0])}",transform=ax.transAxes,fontsize=7)
 if i%2==0:ax.set_ylabel("Difference in captured events\nper 1000 eligible operations")
 if i>=2:ax.set_xlabel("Additional acquisition effort\n(review-equivalents per 1000)")
 rows.extend(d.to_dict("records"))
fig.suptitle("Equal total budget: adding information versus reviewing more patients",fontsize=10,y=.985)
fig.text(.115,.025,"Total budget = ceil(0.20 × n) reviews; acquisition effort is hypothetical and paid for every eligible patient.\nCalibrated frozen workflows; pointwise 95% patient-group bootstrap intervals (2000 replicates).",fontsize=7)
save(fig,"N1_equal_budget_capture",rows)
keys=["9v27_champ","27v38_champ","9v27_LR","27v38_LR","27v38_Voting","6v9_LR","INSPIRE6v9"];labels=["9 to 27 | selected systems","27 to 38 | selected systems","9 to 27 | LR reference","27 to 38 | LR reference","27 to 38 | Voting reference","6 to 9 | LR bridge","6 to 9 | exposed INSPIRE"]
fig,axes=plt.subplots(1,2,figsize=(7.2,4.7),sharey=True);fig.subplots_adjust(left=.28,right=.98,bottom=.24,top=.81,wspace=.16);rows=[]
for ax,line,col in zip(axes,["geography","temporal"],["#0072B2","#D55E00"]):
 for i,k in enumerate(keys):
  n=line+"_"+k;d=b[(b.comparison_id==n)&(b.budget_percent==20)&(b["mode"]=="calibrated")].iloc[0];raw=b[(b.comparison_id==n)&(b.budget_percent==20)&(b["mode"]=="raw")].iloc[0]
  ax.plot([d.lower95,d.upper95],[i,i],c=col,lw=1.4);ax.plot(d.signed_allowable_extra_cost_per1000,i,"o",color=col,ms=5);ax.plot(raw.signed_allowable_extra_cost_per1000,i+.16,"|",color=".2",ms=7);rows.extend([d.to_dict(),raw.to_dict()])
 ax.set_title(line.capitalize(),loc="left",pad=12);ax.axvline(0,c=".4",ls="--",lw=.8);ax.set_xlim(-350,200);ax.set_xticks([-300,-150,0,150]);ax.set_xlabel("Signed review-equivalent headroom\nper 1000 eligible operations");ax.grid(axis="x",color=".9",lw=.5)
axes[0].set_yticks(range(7));axes[0].set_yticklabels(labels);axes[0].invert_yaxis()
fig.suptitle("Effort available while matching baseline event capture",fontsize=10,y=.97);fig.text(.28,.89,"Baseline review capacity: 20% (integer ceiling)",fontsize=8)
fig.legend([Line2D([],[],color="#0072B2",marker="o"),Line2D([],[],color=".2",marker="|",ls="")],["Calibrated: point estimate and 95% interval","Raw: point estimate"],loc="lower center",bbox_to_anchor=(.57,.045),frameon=False,ncol=1)
fig.text(.04,.015,"Negative values require more reviews even at zero extra cost. Retrospective equal-capture boundary.\nAliases and raw/calibrated views are not independent replications.",fontsize=7)
save(fig,"N2_signed_equal_capture_boundary",rows)
D=A/"mimic_calendar_v2";ints=pd.read_csv(D/"INTERVALS.csv");cuts=pd.read_csv(D/"SOURCE_THRESHOLD_BURDEN.csv")
fig,axes=plt.subplots(2,2,figsize=(7.2,6.4));fig.subplots_adjust(left=.20,right=.98,bottom=.15,top=.9,hspace=.43,wspace=.52)
names=["geography__raw","geography__calibrated","temporal__raw","temporal__calibrated"];labs=["Geographic | raw","Geographic | calibrated","Temporal | raw","Temporal | calibrated"];rows=[]
for i,(ax,metric,title) in enumerate(zip(axes.ravel()[:3],["AUROC","Brier","calibration_slope"],["Discrimination","Prediction error","Calibration slope"])):
 for j,n in enumerate(names):
  d=ints[(ints.model==n)&(ints.metric==metric)].iloc[0];col="#0072B2" if n.startswith("geography") else "#D55E00";marker="o" if n.endswith("calibrated") else "s"
  ax.plot([d.lower95,d.upper95],[j,j],color=col,lw=1.3);ax.plot(d.estimate,j,marker,color=col,ms=4);rows.append(d.to_dict())
 ax.set_yticks(range(4));ax.set_yticklabels(labs,fontsize=7);ax.set_ylim(3.5,-.5);ax.set_title(title,loc="left",pad=12);ax.set_xlabel(metric.replace("_"," "));ax.text(-.22,1.12,chr(65+i),transform=ax.transAxes,weight="bold",fontsize=10)
 if metric=="AUROC":ax.axvline(.5,color=".5",ls="--",lw=.6);ax.set_xlim(.45,.78)
 if metric=="calibration_slope":ax.axvline(1,color=".5",ls="--",lw=.6);ax.set_xlim(-.16,1.1)
ax=axes[1,1];j=0;clabs=[]
for line in ["geography","temporal"]:
 for rule,short in [("training_capacity10","10% source"),("training_capacity20","20% source"),("training_sensitivity80","80% capture source")]:
  d=cuts[(cuts.source_line==line)&(cuts.rule==rule)&(cuts.metric=="reviewed_per1000")].iloc[0];col="#0072B2" if line=="geography" else "#D55E00"
  ax.plot([d.lower95,d.upper95],[j,j],color=col);ax.plot(d.estimate,j,"o",color=col,ms=4);clabs.append(("Geo" if line=="geography" else "Time")+" | "+short);j+=1;rows.append(d.to_dict())
ax.set_yticks(range(6));ax.set_yticklabels(clabs,fontsize=6.5);ax.set_ylim(5.5,-.5);ax.set_xlim(-30,1030);ax.set_xticks([0,200,400,600,800,1000]);ax.set_title("Frozen source-threshold workload",loc="left",pad=12);ax.set_xlabel("Patients flagged per 1000");ax.text(-.22,1.12,"D",transform=ax.transAxes,weight="bold",fontsize=10)
fig.suptitle("Independent-source calendar sensitivity: MIMIC-IV",fontsize=10,y=.98)
fig.text(.05,.022,"759 patients; 59 recorded creatinine-ratio events; baseline albumin missing in 465 (61.3%).\nAnchor: billed procedure date; calendar days +1 to +7; day 0 excluded. Six-input frozen RFs only.\nPointwise 95% patient bootstrap intervals (2000); exact surgical-clock and full-input transport not evaluated.",fontsize=7)
save(fig,"N3_MIMIC_calendar_transport",rows)
(F/"RENDER_RECEIPT.json").write_text(json.dumps({"status":"RENDERED_REQUIRES_ACTUAL_VISUAL_REVIEW","backend":"Python/matplotlib","code_sha256":sha(__file__),"figures":renders,"input_hashes":{str(p):sha(p) for p in [A/"resource_comparison/RESOURCE_BUDGET_GRID.csv",A/"resource_comparison/BREAK_EVEN_BOUNDARIES.csv",D/"INTERVALS.csv",D/"SOURCE_THRESHOLD_BURDEN.csv"]}},indent=2)+"\n")
print("Rendered three figures and source data",flush=True)
