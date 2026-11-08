"""Final-size typography repair using accepted aggregate data; no statistical recomputation."""
from pathlib import Path
import argparse,ast,hashlib,json,os
os.environ.update(MPLBACKEND="Agg",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
import numpy as np,pandas as pd,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser()
ap.add_argument("--source-script",type=Path,required=True)
ap.add_argument("--results",type=Path,required=True)
ap.add_argument("--output",type=Path,required=True)
args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);sub=out/"subpanels";sub.mkdir(exist_ok=True)
tree=ast.parse(args.source_script.read_text())
keep=[]
for n in tree.body:
 if isinstance(n,ast.FunctionDef) and n.name!="export":keep.append(n)
 elif isinstance(n,ast.Assign) and all(isinstance(t,ast.Name) for t in n.targets) and {t.id for t in n.targets}<={"colors","styles","labels","contexts","titles"}:keep.append(n)
ns={"np":np,"pd":pd,"plt":plt}
exec(compile(ast.Module(body=keep,type_ignores=[]),str(args.source_script),"exec"),ns)
for name,file in {"D":"POLICY_PAIRED_DIFFERENCES.csv","L":"LOW_RISK_EVENTS.csv","M":"MIMIC_DELETION_SENSITIVITY.csv","W":"DAILY_SUMMARY.csv"}.items():ns[name]=pd.read_csv(args.results/file)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7.5,"axes.titlesize":8,"axes.labelsize":7.5,"xtick.labelsize":7.5,"ytick.labelsize":7.5,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.7,"savefig.facecolor":"white"})
def fingerprint(ax):
 def arr(a):return np.asarray(a,dtype=float).round(12).tolist()
 return {"lines":[{"x":arr(x.get_xdata()),"y":arr(x.get_ydata())} for x in ax.lines],"collections":[{"offsets":arr(x.get_offsets()),"paths":[arr(p.vertices) for p in x.get_paths()]} for x in ax.collections]}
def repair(ax,key):
 # Statistical definitions are in the complete figure legends.
 for text in list(ax.texts):
  if text.get_transform()==ax.transAxes:text.remove()
 for t in [ax.title,ax._left_title,ax._right_title]:t.set_fontsize(8)
 ax.tick_params(labelsize=7.5)
 ax.xaxis.label.set_size(7.5);ax.yaxis.label.set_size(7.5)
 if key.startswith("policy"):
  i=int(key[-1]);ax.set_yticks(range(5),["Selective gate","Highest base risk","Nearest cutoff","Random acquisition","All additional fields"],fontsize=7.5)
  ax.set_title(chr(65+i)+"  "+("Geographic" if i in [0,2] else "Temporal")+(": 9 → 27 inputs" if i<2 else ": 27 → 38 inputs"),loc="left",fontsize=8,pad=8)
  ax.set_xlabel("Δ AKI events captured\nper 1,000 records")
  ax.set_xticks([-10,0,10])
 elif key=="low":
  labels=[t.get_text().replace("Geo preop","Geo preop").replace("Time","Temporal").replace(" / "," · ").replace(" inputs  "," inputs ") for t in ax.get_yticklabels()]
  ax.set_yticks(range(len(labels)),labels,fontsize=7.5)
  ax.set_title("A  Recorded AKI in low-risk groups",loc="left",fontsize=8,pad=8)
 elif key=="deletion":
  ax.set_title("B  MIMIC: measurement-deletion sensitivity",loc="left",fontsize=8,pad=8)
  ax.set_xlabel("Postoperative SCr measurements deleted, %")
  ax.legend(loc="upper left",frameon=False,fontsize=7.5)
  for a in ax.texts:a.set_fontsize(7.5)
 else:
  letter="A" if key=="geography" else "B"
  ax.set_title(letter+"  "+("Geographic" if key=="geography" else "Temporal")+": potential review demand",loc="left",fontsize=8,pad=8)
  ax.set_yticklabels([t.get_text().replace(" / centre "," · center ").replace(" active days"," days") for t in ax.get_yticklabels()],fontsize=7.5)
  ax.legend(loc="lower right",frameon=False,fontsize=7.5)
def call(ax,key):
 if key.startswith("policy"):ns["policy"](ax,int(key[-1]))
 elif key in ["low","deletion"]:ns[key](ax)
 else:ns["demand"](ax,key,"A" if key=="geography" else "B")
 before=fingerprint(ax);repair(ax,key)
 x0,x1=ax.get_xlim();ax.set_xticks([x for x in ax.get_xticks() if x0<=x<=x1])
 assert fingerprint(ax)==before,("Numerical artists changed",key)
 return before
records=[];audit=[]
def save(fig,stem):
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 # Only check that text fits the canvas; clipping/overlap reviewed visually too.
 issues=[]
 for ax in fig.axes:
  for t in [*ax.texts,*ax.get_xticklabels(),*ax.get_yticklabels(),ax.xaxis.label,ax.yaxis.label,ax._left_title]:
   if not t.get_visible() or not t.get_text():continue
   b=t.get_window_extent(renderer)
   if b.x0<-.1 or b.y0<-.1 or b.x1>fig.bbox.width+.1 or b.y1>fig.bbox.height+.1:issues.append(t.get_text())
 assert not issues,("Text outside final-size canvas",stem,issues)
 for ext in ["pdf","svg","png"]:fig.savefig(out/(stem+"."+ext),dpi=300)
 audit.append({"stem":stem,"width_mm":fig.get_figwidth()*25.4,"height_mm":fig.get_figheight()*25.4,"min_font_pt":7.5,"outside_canvas":[]})
 plt.close(fig)
width=160/25.4
groups=[("eFigure_16",["policy0","policy1","policy2","policy3"],(2,2),160/25.4),
        ("eFigure_17",["low","deletion"],(2,1),170/25.4),
        ("eFigure_18",["geography","temporal"],(2,1),170/25.4)]
for stem,keys,shape,height in groups:
 fig,axes=plt.subplots(*shape,figsize=(width,height),squeeze=False)
 if stem=="eFigure_16":fig.subplots_adjust(left=.20,right=.985,bottom=.075,top=.95,hspace=.43,wspace=1.25)
 else:fig.subplots_adjust(left=.39 if stem=="eFigure_17" else .40,right=.97,bottom=.07,top=.95,hspace=.50)
 fps=[]
 for ax,key in zip(axes.ravel(),keys):fps.append(call(ax,key))
 save(fig,stem)
 for i,key in enumerate(keys):
  fig,ax=plt.subplots(figsize=(width,105/25.4))
  fig.subplots_adjust(left=.38 if stem!="eFigure_16" else .29,right=.97,bottom=.16,top=.90)
  fp=call(ax,key);assert fp==fps[i]
  # Keep a distinct output root without rewriting composite output paths.
  old=out;out=sub;save(fig,stem+"_"+chr(65+i));out=old
  records.append({"figure":stem,"panel":chr(65+i),"pdf":"subpanels/"+stem+"_"+chr(65+i)+".pdf","numeric_artist_sha256":hashlib.sha256(json.dumps(fp,sort_keys=True).encode()).hexdigest()})
manifest={"status":"NUMERICAL_ARTISTS_UNCHANGED_FINAL_SIZE_TEXT_FITS_CANVAS_PENDING_VISUAL_REVIEW","figures":audit,"panels":records,"source_script_sha256":hashlib.sha256(args.source_script.read_bytes()).hexdigest(),"source_csvs":{f:hashlib.sha256((args.results/f).read_bytes()).hexdigest() for f in ["POLICY_PAIRED_DIFFERENCES.csv","LOW_RISK_EVENTS.csv","MIMIC_DELETION_SENSITIVITY.csv","DAILY_SUMMARY.csv"]},"statistics_recomputed":False,"native_width_mm":160,"planned_supplement_width_mm":160}
(args.output/"FINAL_SIZE_FIGURE_RECEIPT.json").write_text(json.dumps(manifest,indent=2))
print(json.dumps({"composites":3,"standalone_panels":8,"native_width_mm":160,"numerical_artists_unchanged":True}))
