#!/usr/bin/env python3
"""Reflow accepted native Matplotlib panels at Arial10pt/170mm. No statistical recomputation."""
from pathlib import Path
import argparse,ast,copy,datetime,hashlib,json,os,re,string,sys,textwrap,time,resource
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--project-root',type=Path,default=Path('/home/lei/codex-workspaces/aki-prediction-manuscript'));ap.add_argument('--font-dir',type=Path);ap.add_argument('--only',nargs='*');args=ap.parse_args();R=args.root
P=args.project_root
os.environ.update(MPLBACKEND='Agg',MPLCONFIGDIR=str(R/'runtime/mpl'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
FONTDIR=args.font_dir or P/'writing_runs/final_source_audit_20261108_v1/fonts'
if not (FONTDIR/'Arial.ttf').is_file():raise FileNotFoundError('Actual licensed Arial.ttf is required via --font-dir; fallback is not accepted.')
for path in FONTDIR.glob('Arial*.ttf'):font_manager.fontManager.addfont(str(path))
FONT=font_manager.FontProperties(fname=str(FONTDIR/'Arial.ttf'))
assert FONT.get_name()=='Arial'
plt.rcParams.update({'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold'})
B=P/'writing_runs/probabilistic_integration_20261105_v1/package'
assets={pdf.stem:folder.name for folder in (B/'05_Figure_assets').iterdir() if folder.is_dir() for pdf in folder.glob('*.pdf')}
renumber=json.loads((P/'writing_runs/manuscript_final_20261031_v1/provenance/JAMA_FIGURE_RENUMBERING.json').read_text())
BASEPAN=P/'writing_runs/workbuddy_final_revision_20261108_v1/figures/native_panel_exports/PANEL_MANIFEST.json'
baseline={(x['figure'],x['panel']):x['data_artist_sha256'] for x in json.loads(BASEPAN.read_text())['panels']}
all_panels=[];figures=[];source_hashes={};used_csv={};started=time.time()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def old_fingerprint(ax):
 h=hashlib.sha256()
 for item in ax.lines:h.update(np.asarray(item.get_xydata(),dtype=float).tobytes())
 for item in ax.collections:
  h.update(np.ma.asarray(item.get_offsets()).filled(np.nan).tobytes())
  for path in item.get_paths():h.update(np.asarray(path.vertices,dtype=float).tobytes())
 for item in ax.images:h.update(np.asarray(item.get_array()).tobytes())
 for item in ax.patches:h.update(np.asarray(item.get_path().vertices,dtype=float).tobytes())
 return h.hexdigest()
def rich_fingerprint(ax):
 obj={'original_numeric_artist_sha256':old_fingerprint(ax),'rectangles':[[float(getattr(x,n)()) for n in ['get_x','get_y','get_width','get_height']] for x in ax.patches if all(hasattr(x,n) for n in ['get_x','get_y','get_width','get_height'])]}
 return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def text_fingerprint(ax):
 vals=[]
 for t in ax.texts:
  s=' '.join(t.get_text().split())
  if len(s)==1 and s in string.ascii_letters:continue
  vals.append(s)
 return hashlib.sha256(json.dumps(vals,ensure_ascii=False).encode()).hexdigest()
def set_font(t,size=10):
 t.set_fontfamily('Arial');t.set_fontsize(size)
def rendered_texts(fig,ax):
 values=list(ax.texts)+[ax.xaxis.label,ax.yaxis.label,ax.xaxis.offsetText,ax.yaxis.offsetText]
 for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
  for tick in axis.get_major_ticks()+axis.get_minor_ticks():
   if min(limits)-1e-8<=tick.get_loc()<=max(limits)+1e-8:values.extend([tick.label1,tick.label2])
 legend=ax.get_legend()
 if legend:values+=legend.get_texts()+[legend.get_title()]
 return [t for t in values if t.get_visible() and t.get_text().strip()]
def numeric_notes(ax):return [{'text':' '.join(t.get_text().split()),'position':list(map(float,t.get_position()))} for t in ax.texts if re.search(r'\d',t.get_text())]
def export_figure(fig,label,source_name):
 if not label.startswith('eFigure_') or (args.only and label not in args.only):plt.close(fig);return
 fig.canvas.draw();panels=[a for a in fig.axes if a.get_visible() and (a.has_data() or a.texts or a.patches)]
 if not panels:raise RuntimeError('No panels '+label)
 original_positions={id(a):a.get_position().frozen() for a in fig.axes}
 labels={id(a):(a.get_xlabel(),a.get_ylabel()) for a in panels}
 native_hashes=[old_fingerprint(a) for a in panels];original_fig_size=fig.get_size_inches().copy()
 note_text=[t.get_text() for t in fig.texts if len(t.get_text().strip())>1]
 fig_legends=list(fig.legends);fig_text_visible=[(t,t.get_visible()) for t in fig.texts]
 for t,_ in fig_text_visible:t.set_visible(False)
 for leg in fig_legends:leg.set_visible(False)
 if fig.get_layout_engine() is not None:fig.set_layout_engine(None)
 figure_record={'figure':label,'source_name':source_name,'panel_letters':[],'source_figure_notes':note_text,'panels':[]}
 for i,ax in enumerate(panels):
  letter=string.ascii_uppercase[i];key=(label,letter);before=old_fingerprint(ax);rich_before=rich_fingerprint(ax);texts_before=text_fingerprint(ax)
  if key in baseline:assert before==baseline[key],('Baseline numeric artist mismatch',key)
  others=[(a,a.get_visible()) for a in fig.axes if a is not ax]
  for a,_ in others:a.set_visible(False)
  ax.set_visible(True)
  source_title=ax.get_title(loc='left') or ax.get_title() or ax.get_title(loc='right')
  for loc in ['left','center','right']:ax.set_title('',loc=loc)
  source_title=re.sub(r'^[A-D]\s{2,}','',source_title).strip()
  extra_legend=None;legend=ax.get_legend()
  if not legend and fig_legends and (label!='eFigure_13' or i==3):
   leg=fig_legends[0];handles=[copy.copy(h) for h in leg.legend_handles]
   extra_legend=ax.legend(handles,[t.get_text() for t in leg.get_texts()],loc='upper center',bbox_to_anchor=(.5,-.30),frameon=False,ncol=(1 if sum(len(t.get_text()) for t in leg.get_texts())>65 else min(3,len(handles))),fontsize=10)
  elif legend:
   # Place a copy of the existing legend outside the data region.
   handles=[copy.copy(h) for h in legend.legend_handles];texts=[t.get_text() for t in legend.get_texts()]
   legend.remove();extra_legend=ax.legend(handles,texts,loc='upper center',bbox_to_anchor=(.5,-.30),frameon=False,ncol=(1 if sum(len(t) for t in texts)>65 else min(3,len(handles))),fontsize=10)
  if ax.axison:
   if not ax.get_xlabel():
    aligned=[a for a in panels if abs(original_positions[id(a)].x0-original_positions[id(ax)].x0)<.01 and labels[id(a)][0]]
    if aligned:ax.set_xlabel(labels[id(aligned[-1])][0])
   if not ax.get_ylabel():
    aligned=[a for a in panels if abs(original_positions[id(a)].y0-original_positions[id(ax)].y0)<.01 and labels[id(a)][1]]
    if aligned:ax.set_ylabel(labels[id(aligned[0])][1])
   ax.tick_params(labelleft=True,labelbottom=True,labelsize=10)
  for t in fig.findobj(Text):set_font(t)
  for t in ax.texts:
   if len(t.get_text().strip())==1 and t.get_text().strip().lower() in string.ascii_lowercase[:len(panels)] and t.get_transform()==ax.transAxes:
    x,y=t.get_position()
    if x<0 or y>=1:t.set_visible(False)
  for t in [ax.xaxis.label,ax.yaxis.label]:
   if len(t.get_text())>55 and '\n' not in t.get_text():t.set_text(textwrap.fill(t.get_text(),55))
  # Explicit source ticks prevent font-driven auto tick changes.
  if ax.axison:
   xlim=ax.get_xlim();ylim=ax.get_ylim()
   xticks=[x for x in ax.get_xticks() if min(xlim)-1e-8<=x<=max(xlim)+1e-8]
   yticks=[x for x in ax.get_yticks() if min(ylim)-1e-8<=x<=max(ylim)+1e-8]
   ax.set_xticks(xticks);ax.set_yticks(yticks);ax.set_xlim(xlim);ax.set_ylim(ylim)
  footer_texts=[t for t in ax.texts if t.get_transform()==ax.transAxes and t.get_position()[1]<0]
  for t in footer_texts:
   if any(len(line)>92 for line in t.get_text().splitlines()):t.set_text('\n'.join(textwrap.fill(line,92) for line in t.get_text().splitlines()))
   t.set_transform(fig.transFigure);t.set_va('bottom');t.set_ha('left')
  footer_h=(sum(max(1,len(t.get_text().splitlines())) for t in footer_texts)*4.3+4) if footer_texts else 0
  legend_h=0
  if extra_legend:
   legend_h=(len(extra_legend.get_texts()) if extra_legend._ncols==1 else 1)*5+4
  if label=='eFigure_17' and letter=='B':
   offsets={'First per day':(5,6),'Odd days':(6,10),'Even days':(-50,33),'Day 1 only':(6,10),'Days 1–2 only':(-80,0)}
   for t in ax.texts:
    if t.get_text() in offsets and hasattr(t,'xy'):t.set_position(offsets[t.get_text()])
  rows=len([t for t in ax.get_yticklabels() if t.get_text().strip()]) if ax.axison else 0
  height=max(90.,min(170.,36+rows*5.7))
  if not ax.axison:height=90.
  if extra_legend and footer_texts:height=max(height,110.)
  # Fit actual rendered text in physical units; never reduce font size.
  title='  '.join([letter,source_title]) if source_title else letter
  title_lines=textwrap.fill(title,width=82)
  title_artist=fig.text(3/170,1-2/height,title_lines,ha='left',va='top',fontfamily='Arial',fontsize=10,fontweight='bold')
  issues=[];layout_history=[]
  for attempt in range(3):
   fig.set_size_inches(170/25.4,height/25.4,forward=True)
   title_artist.set_position((3/170,1-2/height))
   y_footer=2.
   for t in reversed(footer_texts):
    t.set_position((3/170,y_footer/height));y_footer+=max(1,len(t.get_text().splitlines()))*4.3+1
   if extra_legend:
    extra_legend.set_bbox_to_anchor((.5,(footer_h+3)/height),transform=fig.transFigure);extra_legend._loc=8

   # Full-width panels: source geometry/values preserved, dimensions grow.
   left=0.29 if ax.axison else .035;right=.975;bottom=(max(.19,(footer_h+legend_h+14)/height) if ax.axison else .04);top=.84 if '\n' in title_lines else .89
   if label=='eFigure_2' and letter=='B':bottom+=.07
   if attempt:
    left=last[0];right=last[1];bottom=last[2];top=last[3]
   for _ in range(5):
    if right-left<.25 or top-bottom<.22:break
    ax.set_position([left,bottom,right-left,top-bottom]);fig.canvas.draw();renderer=fig.canvas.get_renderer()
    texts=rendered_texts(fig,ax);boxes=[t.get_window_extent(renderer) for t in texts]
    if not boxes:break
    bb=Bbox.union(boxes);fw,fh=fig.bbox.width,fig.bbox.height;pad=2/25.4*fig.dpi
    dl=max(0,(pad-bb.x0)/fw);dr=max(0,(bb.x1-(fw-pad))/fw);db=max(0,(pad-bb.y0)/fh);dtp=max(0,(bb.y1-(fh-12/25.4*fig.dpi))/fh)
    if max(dl,dr,db,dtp)<.0005:break
    left+=dl;right-=dr;bottom+=db;top-=dtp
   last=(left,right,bottom,top);fig.canvas.draw();renderer=fig.canvas.get_renderer();fw,fh=fig.bbox.width,fig.bbox.height
   issues=[]
   for t in rendered_texts(fig,ax)+[title_artist]:
    bb=t.get_window_extent(renderer)
    if bb.x0<-.3 or bb.y0<-.3 or bb.x1>fw+.3 or bb.y1>fh+.3:issues.append(t.get_text())
   layout_history.append({'attempt':attempt+1,'height_mm':height,'axes_position':list(ax.get_position().bounds),'outside_text':issues})
   if not issues and right-left>=.25 and top-bottom>=.22:break
   height=min(180.,height+25);last=(.34,.97,.26,.87)
  assert not issues,('Outside figure text',key,issues,layout_history)
  assert before==old_fingerprint(ax) and rich_before==rich_fingerprint(ax),('Numerical artists changed',key)
  assert texts_before==text_fingerprint(ax),('Numerical/annotation text changed',key)
  folder=R/'panels'/label;folder.mkdir(exist_ok=True);stem=folder/(label+'_'+letter)
  plt.rcParams.update({'font.family':'Arial','font.size':10,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'pdf.use14corefonts':False,'savefig.facecolor':'white'})
  for ext in ['pdf','svg','png']:fig.savefig(stem.with_suffix('.'+ext),dpi=300,facecolor='white')
  fonts=[t.get_fontsize() for t in rendered_texts(fig,ax)+[title_artist]]
  assert min(fonts)>=10-1e-8
  rec={'figure':label,'panel':letter,'pdf':str(stem.with_suffix('.pdf').relative_to(R)),'svg':str(stem.with_suffix('.svg').relative_to(R)),'png':str(stem.with_suffix('.png').relative_to(R)),'width_mm':170.,'height_mm':height,'font_family':'Arial','minimum_text_pt':min(fonts),'numeric_artist_sha256':before,'rich_numeric_artist_sha256':rich_before,'annotation_text_sha256':texts_before,'baseline_native_numeric_match':key in baseline,'source_title':source_title,'layout_history':layout_history,'numeric_texts':numeric_notes(ax),'pdf_sha256':sha(stem.with_suffix('.pdf')),'svg_sha256':sha(stem.with_suffix('.svg'))}
  all_panels.append(rec);figure_record['panel_letters'].append(letter);figure_record['panels'].append(rec)
  (R/'audit/PANEL_CHECKPOINT.json').write_text(json.dumps({'panels':all_panels,'figures':figures+[figure_record],'source_hashes':source_hashes},indent=2)+'\n')
  title_artist.remove()
  if extra_legend:extra_legend.remove()
  for a,visible in others:a.set_visible(visible)
  ax.set_position(original_positions[id(ax)])
  print('PANEL',label,letter,'height',height,'artist',before[:12],flush=True)
 assert native_hashes==[old_fingerprint(a) for a in panels]
 figures.append(figure_record);plt.close(fig)
def record(fig,name,*a,**kw):
 label=assets.get(name) or assets.get(renumber.get(name,''))
 if label:export_figure(fig,label,name)
 else:plt.close(fig)
def save(fig,name,*a,**kw):record(fig,name)
sources=[B/'02_Code_and_aggregate_data/figure_code/build_figures.py',B/'02_Code_and_aggregate_data/figure_code/gen_fig_novelty.py']
for source in sources:
 source_hashes[str(source)]=sha(source);tree=ast.parse(source.read_text());callname='record' if source.name=='build_figures.py' else 'save'
 last=max(i for i,n in enumerate(tree.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id==callname)
 body=[n for n in tree.body[:last+1] if not(isinstance(n,ast.FunctionDef) and n.name==callname)]
 canonical=P/'writing_runs/manuscript_final_20261031_v1/code/build_figures.py' if callname=='record' else source
 ns={'__file__':str(canonical),'__name__':'jno_format_native_replay',callname:globals()[callname]}
 argv=sys.argv;sys.argv=[str(canonical)]
 exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),ns);sys.argv=argv
 # Retain all source CSV hashes; they are aggregate figure inputs.
 sdir=ns.get('S')
 if sdir and Path(sdir).is_dir():
  for p in Path(sdir).glob('*.csv'):used_csv[str(p)]=sha(p)
# Three new accepted figures. Import pure definitions and explicitly load saved aggregates.
source=P/'correction_runs/clinical_supplement_20261108_v1/code/plot_supplement.py';source_hashes[str(source)]=sha(source);tree=ast.parse(source.read_text());body=[]
for n in tree.body:
 if isinstance(n,ast.FunctionDef) and n.name!='export':body.append(n)
 elif isinstance(n,ast.Assign) and all(isinstance(t,ast.Name) for t in n.targets) and {t.id for t in n.targets}<={'colors','styles','labels','contexts','titles'}:body.append(n)
ns={'np':np,'pd':pd,'plt':plt};exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),ns)
for key,file in {'D':'POLICY_PAIRED_DIFFERENCES.csv','L':'LOW_RISK_EVENTS.csv','M':'MIMIC_DELETION_SENSITIVITY.csv','W':'DAILY_SUMMARY.csv'}.items():
 path=P/'correction_runs/clinical_supplement_20261108_v1/results'/file;ns[key]=pd.read_csv(path);used_csv[str(path)]=sha(path)
for label,keys in [('eFigure_16',['policy0','policy1','policy2','policy3']),('eFigure_17',['low','deletion']),('eFigure_18',['geography','temporal'])]:
 if args.only and label not in args.only:continue
 fig,axes=plt.subplots(len(keys),1,figsize=(170/25.4,90*len(keys)/25.4));axes=np.atleast_1d(axes)
 for ax,key in zip(axes,keys):
  if key.startswith('policy'):ns['policy'](ax,int(key[-1]))
  elif key in ['low','deletion']:ns[key](ax)
  else:ns['demand'](ax,key,'A' if key=='geography' else 'B')
 export_figure(fig,label,source.name)
assert len({x['figure'] for x in all_panels})==(len(args.only) if args.only else 18)
result={'status':'PANEL_EXPORTS_COMPLETE_PENDING_PAGE_COMPOSITION_AND_VISUAL_REVIEW','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'font_family':'Arial','body_font_pt':10,'width_mm':170,'panels':all_panels,'figures':figures,'source_scripts':source_hashes,'source_csvs':used_csv,'original_native_panel_manifest_sha256':sha(BASEPAN),'source_hashes_unchanged':all(sha(p)==h for p,h in {**source_hashes,**used_csv}.items()),'new_statistics':0,'seconds':time.time()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'script_sha256':sha(__file__)}
(R/'PANEL_MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'figures':len(figures),'panels':len(all_panels),'seconds':result['seconds'],'peak_RSS_MiB':result['peak_RSS_MiB']}),flush=True)
