"""Descriptive fixed-score curves; predefined10/20% capacities, no threshold selection."""
from pathlib import Path
import json,hashlib,math
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve,average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
O=R/"outputs/published_benchmark_comparison_20261024_v1"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert json.loads((O/"REVIEW_ACCEPTANCE.json").read_text())["status"].startswith("PASS")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.labelsize":11,
 "axes.titlesize":11,"axes.spines.top":False,"axes.spines.right":False,
 "legend.fontsize":9,"pdf.fonttype":42,"ps.fonttype":42})
fig,axs=plt.subplots(2,2,figsize=(11,8.2))
provenance={}
for col,(line,label,our_family) in enumerate([("geography","Geographic evaluation","LR"),("temporal","Temporal evaluation","Voting")]):
 f=O/f"private_SIM_{line}_predictions.csv";d=pd.read_csv(f)
 provenance[f.name]=sha(f);y=d.y.to_numpy(int)
 for model,name,color in [("our_endop_cal",f"Our locked end-of-operation {our_family}","#0072B2"),
                          ("SIM_original_2009","SIM-AKI published formula*","#D55E00")]:
  s=d[model].to_numpy();prec,rec,_=precision_recall_curve(y,s)
  ap=average_precision_score(y,s)
  axs[0,col].step(100*rec,100*prec,where="post",color=color,lw=1.7,label=f"{name}\nAP = {ap:.3f}")
  order=np.array(sorted(range(len(d)),key=lambda i:(-s[i],str(d.group_hash.iloc[i]),str(d.excel_row.iloc[i]))))
  captured=np.r_[0,np.cumsum(y[order])]
  x=100*np.arange(len(d)+1)/len(d);r=100*captured/y.sum()
  axs[1,col].step(x,r,where="post",color=color,lw=1.7,label=name)
  for fraction in [.1,.2]:
   k=math.ceil(fraction*len(d));axs[1,col].scatter([100*k/len(d)],[r[k]],color=color,s=28,zorder=3)
 axs[0,col].axhline(100*y.mean(),color=".5",ls=":",lw=1,label="Cohort event rate")
 axs[0,col].set(xlim=(0,100),ylim=(0,102),xlabel="Sensitivity / recall (%)",ylabel="Positive predictive value (%)",
                title=f"{label}: n = {len(d):,}; AKI = {int(y.sum())}")
 axs[0,col].legend(loc="upper right",frameon=False)
 axs[1,col].plot([0,100],[0,100],color=".5",ls=":",lw=1)
 for capacity in [10,20]:axs[1,col].axvline(capacity,color=".8",ls="--",lw=.8)
 axs[1,col].set(xlim=(0,100),ylim=(0,102),xlabel="Patients selected for review (%)",ylabel="AKI cases captured (%)",
                title="AKI capture across review capacity (10% and 20% marked)")
 axs[1,col].legend(loc="lower right",frameon=False)
 for ax in axs[:,col]:ax.grid(alpha=.12)
fig.suptitle("Published clinical benchmark: precision-recall and review burden",fontsize=14,y=.995)
fig.text(.04,.014,"* Declared primary mappings, including CKD-EPI 2009. Same patients within each comparison.\n"
 "Point curves describe fixed models; paired intervals are reported separately. No clinical intervention effect is implied.",fontsize=9)
fig.tight_layout(rect=[0,.065,1,.97])
files=[]
for ext in ["pdf","png"]:
 p=O/f"SIM_AKI_COMPARISON_CURVES.{ext}";fig.savefig(p,dpi=220,bbox_inches="tight");files.append(p.name)
plt.close(fig)
receipt={"status":"RENDERED_visual_review_pending","source_hashes":provenance,"code_sha256":sha(__file__),
 "figures_sha256":{f:sha(O/f) for f in files},"all_curve_points_included":True,
 "clinical_capacities_remain_prespecified":[.1,.2],"models_or_thresholds_changed":False}
(O/"COMPARISON_FIGURE_RENDER.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
