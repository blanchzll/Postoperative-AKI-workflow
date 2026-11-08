from pathlib import Path
import ast,json,hashlib,string,sys,copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
R=Path('/home/lei/codex-workspaces/aki-prediction-manuscript')
W=R/'writing_runs/workbuddy_final_revision_20261108_v1'
B=R/'writing_runs/probabilistic_integration_20261105_v1/package'
O=W/'figures/native_panel_exports';O.mkdir(exist_ok=True)
assets={}
for folder in (B/'05_Figure_assets').iterdir():
 if folder.is_dir():
  for pdf in folder.glob('*.pdf'):assets[pdf.stem]=folder.name
records=[];unmapped=[]
def fingerprint(ax):
 h=hashlib.sha256()
 for item in ax.lines:
  data=np.asarray(item.get_xydata(),dtype=float);h.update(data.tobytes())
 for item in ax.collections:
  h.update(np.ma.asarray(item.get_offsets()).filled(np.nan).tobytes())
  for path in item.get_paths():h.update(np.asarray(path.vertices,dtype=float).tobytes())
 for item in ax.images:h.update(np.asarray(item.get_array()).tobytes())
 for item in ax.patches:h.update(np.asarray(item.get_path().vertices,dtype=float).tobytes())
 return h.hexdigest()
def export(fig,name):
 label=assets.get(name)
 if label is None:
  renumber=R/'writing_runs/manuscript_final_20261031_v1/provenance/JAMA_FIGURE_RENUMBERING.json'
  mapped=json.loads(renumber.read_text()).get(name)
  label=assets.get(mapped)
 if label is None:
  unmapped.append(name);plt.close(fig);return
 fig.canvas.draw()
 panels=[a for a in fig.axes if a.get_visible() and (a.has_data() or a.texts or a.patches)]
 if not panels:raise RuntimeError('No data panels '+name)
 dest=O/label;dest.mkdir(exist_ok=True)
 # Keep composite panel labels consistent with the standalone PDFs and captions.
 original_hashes=[fingerprint(a) for a in panels]
 for text_artist in list(fig.texts)+[t for a in panels for t in a.texts]:
  text=text_artist.get_text().strip()
  if len(text)==1 and text.lower() in string.ascii_lowercase[:len(panels)]:
   text_artist.set_text(text.upper())
 composite=W/'figures/revised_composites'/label
 composite.parent.mkdir(exist_ok=True)
 fig.savefig(composite.with_suffix('.pdf'),format='pdf')
 fig.savefig(composite.with_suffix('.svg'),format='svg')
 fig.savefig(composite.with_suffix('.png'),format='png',dpi=600)
 fig.savefig(composite.with_suffix('.tiff'),format='tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
 assert original_hashes==[fingerprint(a) for a in panels],label
 renderer=fig.canvas.get_renderer()
 # Original frozen figure styles and artists are retained. Each PDF is rendered
 # from a single native axes object, with its own labels and uncertainty artists.
 for i,ax in enumerate(panels):
  before=fingerprint(ax)
  others=[(a,a.get_visible()) for a in fig.axes if a is not ax]
  ft=[(t,t.get_visible()) for t in fig.texts]
  legends=[(l,l.get_visible()) for l in fig.legends]
  for a,_ in others:a.set_visible(False)
  for t,_ in ft:t.set_visible(False)
  for l,_ in legends:l.set_visible(False)
  letter=string.ascii_uppercase[i]
  existing_letters=[t for t in ax.texts if t.get_visible() and t.get_text().strip().upper()==letter]
  tag=None if existing_letters else ax.text(-.02,1.12,letter,transform=ax.transAxes,ha='right',va='bottom',weight='bold',fontsize=8)
  extra_legend=None
  if ax.axison and not ax.get_xlabel():
   aligned=[a for a in panels if abs(a.get_position().x0-ax.get_position().x0)<.01 and a.get_xlabel()]
   if aligned:ax.set_xlabel(aligned[-1].get_xlabel())
  if ax.axison and not ax.get_ylabel():
   aligned=[a for a in panels if abs(a.get_position().y0-ax.get_position().y0)<.01 and a.get_ylabel()]
   if aligned:ax.set_ylabel(aligned[0].get_ylabel())
  if legends and ax.get_legend() is None and (label!='eFigure_13' or i==3):
   source_legend=legends[0][0]
   handles=[copy.copy(h) for h in source_legend.legend_handles]
   if label=='Figure_3' and ax.lines:
    color=ax.lines[0].get_color()
    handles[0].set_color(color)
    handles[0].set_markerfacecolor(color)
    handles[0].set_markeredgecolor(color)
   extra_legend=ax.legend(handles,[t.get_text() for t in source_legend.get_texts()],loc='upper left',bbox_to_anchor=(0,-.35),frameon=False,fontsize=6.5)
  # Restore inherited shared-axis tick labels for standalone interpretation.
  if ax.axison:ax.tick_params(labelleft=True,labelbottom=True)
  fig.canvas.draw()
  box=ax.get_tightbbox(fig.canvas.get_renderer())
  for t in ax.texts:
   if t.get_visible():box=Bbox.union([box,t.get_window_extent(fig.canvas.get_renderer())])
  bounds=box.transformed(fig.dpi_scale_trans.inverted()).padded(.08)
  file=dest/(label+'_'+letter+'.pdf')
  fig.savefig(file,format='pdf',bbox_inches=bounds,pad_inches=0)
  fig.savefig(dest/(label+'_'+letter+'_preview.png'),format='png',bbox_inches=bounds,pad_inches=0,dpi=160)
  after=fingerprint(ax)
  assert before==after,(name,letter)
  records.append({'data_artist_sha256':after,'data_unchanged_by_export':True,'figure':label,'source_name':name,'panel':letter,'file':str(file.relative_to(W)),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'native_axes':True,'axes_lines':len(ax.lines),'axes_collections':len(ax.collections),'axes_patches':len(ax.patches),'axes_images':len(ax.images),'own_title':ax.get_title(),'shared_figure_legend_embedded':extra_legend is not None,'visual_review':'PENDING'})
  if tag is not None:tag.remove()
  if extra_legend is not None:extra_legend.remove()
  for a,v in others:a.set_visible(v)
  for t,v in ft:t.set_visible(v)
  for l,v in legends:l.set_visible(v)
 plt.close(fig)
def record(fig,name,*args,**kwargs):export(fig,name)
def save(fig,name,*args,**kwargs):export(fig,name)
sources=[B/'02_Code_and_aggregate_data/figure_code/build_figures.py',B/'02_Code_and_aggregate_data/figure_code/gen_fig_novelty.py']
for source in sources:
 tree=ast.parse(source.read_text());callname='record' if source.name=='build_figures.py' else 'save'
 last=max(i for i,n in enumerate(tree.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id==callname)
 body=[n for n in tree.body[:last+1] if not (isinstance(n,ast.FunctionDef) and n.name==callname)]
 # The source-root identity guard receives its original canonical __file__.
 canonical=R/'writing_runs/manuscript_final_20261031_v1/code/build_figures.py' if callname=='record' else source
 ns={'__file__':str(canonical),'__name__':'native_panel_replay',callname:globals()[callname]}
 sys.argv=[str(canonical)]
 exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),ns)
manifest={'status':'NATIVE_PANEL_EXPORTS_GENERATED_VISUAL_REVIEW_PENDING','panels':records,'figures_covered':sorted({r['figure'] for r in records}),'unmapped_historical_figures':unmapped,'source_scripts':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],'analysis_refit':False,'baseline_exports_modified':False,'shared_legend_policy':'Shared global legends are embedded into standalone panels where needed; original figure captions also accompany each panel.'}
(O/'PANEL_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'panels':len(records),'figures':len(manifest['figures_covered']),'unmapped':unmapped,'review_pending':True}))
