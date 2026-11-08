#!/usr/bin/env python3
"""Rebuild publication figures exclusively from frozen aggregate estimates."""
from pathlib import Path
import argparse, json, hashlib, warnings, sys, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps

P=Path('/home/lei/codex-workspaces/aki-prediction-manuscript')
W=P/'writing_runs/manuscript_final_20261031_v1';S=W/'source_data'
assert W.resolve()==Path(__file__).resolve().parent.parent, 'Active writing-root mismatch'
fig_width_mm=183
PREVIEW_RESOLUTION=200
PRINT_PREVIEW_RESOLUTION=96
parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');parser.add_argument('--only',nargs='*');args=parser.parse_args()
JAMA_FIGURE_NAMES=json.loads((W/'provenance/JAMA_FIGURE_RENUMBERING.json').read_text())
old_manifest={x.get('source_figure',x['figure']):x for x in json.loads((W/'FIGURE_MANIFEST.json').read_text())}
old_layout={x.get('source_figure',x['figure']):x for x in json.loads((W/'qa/LAYOUT_AUDIT.json').read_text())}
mpl.rcParams.update({
 'font.family':'sans-serif','font.sans-serif':['Liberation Sans','DejaVu Sans'],
 'font.size':7,'axes.labelsize':7,'axes.titlesize':7,'xtick.labelsize':7,'ytick.labelsize':7,
 'legend.fontsize':6.5,'axes.linewidth':0.6,'lines.linewidth':0.9,
 'xtick.major.width':0.6,'ytick.major.width':0.6,'xtick.major.size':2.5,'ytick.major.size':2.5,
 'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
 'svg.fonttype':'none','pdf.fonttype':42,'pdf.use14corefonts':False,
 'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
 'text.color':'#222222','axes.labelcolor':'#222222','axes.edgecolor':'#555555',
 'xtick.color':'#333333','ytick.color':'#333333',
 'hatch.linewidth':0.6,'text.usetex':False,
})
C={'geography':'#0072B2','temporal':'#D55E00','reference':'#444444','selected':'#8063A6','grey':'#D9DDE1'}
manifest=[];layouts=[];textsizes=[]
sys.path.insert(0,str(W/'provenance/skill_snapshots/scipilot-figure-skill/scripts'))
from visual_qa import audit_layout
def load(name):return pd.read_csv(S/(name+'.csv'))
def match(df,**kw):
    d=df
    for k,v in kw.items():d=d[d[k]==v]
    assert len(d)==1,(kw,len(d))
    return d.iloc[0]
def col(context):return C['geography'] if 'geography' in context else C['temporal']
def clean(ax):
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.grid(False);ax.tick_params(direction='out')
def figure(rows,cols,height,*,left=.20,right=.97,bottom=.11,top=.91,hspace=.75,wspace=.70,ratios=None):
    fig,axs=plt.subplots(rows,cols,figsize=(fig_width_mm/25.4,height/25.4),squeeze=False,
      gridspec_kw={'height_ratios':ratios} if ratios else None)
    fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top,wspace=wspace,hspace=hspace)
    for a in axs.flat:clean(a)
    return fig,axs
def panel(ax,letter,title,column=0):
    letter=letter.lower()
    fig=ax.figure;p=ax.get_position()
    fig.text(.02 if column==0 else .515,p.y1+.025,letter,fontsize=7,fontweight='bold',va='bottom')
    ax.set_title(title,loc='left',pad=7,fontweight='normal')
def draw_ci(ax,y,v,lo,hi,color,marker='o',filled=True):
    assert np.isfinite([v,lo,hi]).all() and lo<=v+1e-9 and hi>=v-1e-9,(v,lo,hi)
    ax.errorbar(v,y,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt=marker,
      markersize=3.2,markerfacecolor=color if filled else 'white',
      markeredgecolor=color,color=color,elinewidth=.7,capsize=2,capthick=.6,zorder=3)
def forest(ax,rows,metric,*,null=None,scale=1,xlim=None,labelkey='label',numeric=False):
    ys=np.arange(len(rows))[::-1]
    for y,r in zip(ys,rows):
        color=r.get('color',col(str(r.get('context',r.get('line','geography')))))
        draw_ci(ax,y,r['estimate']*scale,r['lower95']*scale,r['upper95']*scale,color,r.get('marker','o'),r.get('filled',True))
    ax.set_yticks(ys,[r[labelkey] for r in rows]);ax.set_ylim(-.65,len(rows)-.35)
    if null is not None:ax.axvline(null,color='#909090',lw=.6,ls=(0,(3,3)),zorder=1)
    if xlim:ax.set_xlim(*xlim)
    ax.set_xlabel(metric);clean(ax)
    ax._forest_intervals=[dict(label=r[labelkey],estimate=float(r['estimate']*scale),lower95=float(r['lower95']*scale),upper95=float(r['upper95']*scale)) for r in rows]
def footer(fig,label):fig.text(.02,.018,label,fontsize=6.5,va='bottom')
def record(fig,name,height,sources,title):
    if args.only and name not in args.only:
        assert name in old_manifest and all((W/f).exists() for f in old_manifest[name]['exports'])
        manifest.append(old_manifest[name]);layouts.append(old_layout[name]);plt.close(fig);return
    source_name=name;name=JAMA_FIGURE_NAMES[name]
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();bounds=fig.bbox
    clipped=[]
    not_drawn=set()
    for ax in fig.axes:
        for axis,lim in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if tick.get_loc()<min(lim)-1e-10 or tick.get_loc()>max(lim)+1e-10:
                    not_drawn.update([tick.label1,tick.label2])
    for t in fig.findobj(mpl.text.Text):
        if t in not_drawn or not t.get_visible() or not t.get_text().strip():continue
        sz=float(t.get_fontsize());textsizes.append({'figure':name,'size_pt':sz})
        box=t.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > bounds.x1+1 or box.y1 > bounds.y1+1:
            clipped.append({'text':t.get_text(),'bbox':list(map(float,box.bounds))})
    audit=audit_layout(fig)
    forests=[];interval_clipping=[]
    for ax in fig.axes:
        if not hasattr(ax,'_forest_intervals'):continue
        lo,hi=map(float,ax.get_xlim())
        forests.append(dict(xlabel=ax.get_xlabel(),limits=[lo,hi],rows=ax._forest_intervals))
        for r in ax._forest_intervals:
            if r['lower95']<lo-1e-9 or r['upper95']>hi+1e-9:interval_clipping.append(dict(xlabel=ax.get_xlabel(),limits=[lo,hi],**r))
    assert not interval_clipping,(name,interval_clipping)
    layouts.append({'figure':name,'source_figure':source_name,'canvas_clipped_text':clipped,'intervals_clipped_by_axes':interval_clipping,'forest_intervals':forests,'scipilot_audit':audit})
    category='main' if name.startswith('Figure_') else 'supplement'
    out=W/'figures'/category/name
    fig.savefig(W/'previews'/(name+'.png'),dpi=PREVIEW_RESOLUTION)
    fig.savefig(W/'previews'/(name+'_print96.png'),dpi=PRINT_PREVIEW_RESOLUTION)
    with Image.open(W/'previews'/(name+'.png')) as im:
        ImageOps.grayscale(im).save(W/'previews'/(name+'_grayscale.png'))
    exports=[]
    if args.final:
        for ext in ['.pdf','.svg']:
            file=out.with_suffix(ext);fig.savefig(file);exports.append(str(file.relative_to(W)))
        file=out.with_suffix('.png');fig.savefig(file,dpi=600);exports.append(str(file.relative_to(W)))
        file=out.with_suffix('.tiff');fig.savefig(file,dpi=600,pil_kwargs={'compression':'tiff_lzw'});exports.append(str(file.relative_to(W)))
    manifest.append({'figure':name,'source_figure':source_name,'title':title,'width_mm':183,'height_mm':height,
      'source_data':[f'source_data/{s}.csv' for s in sources],'exports':exports,'preview':f'previews/{name}.png',
      'statistical_unit':'operative record; patient-group partition/resampling unless caption defines conditional variability'})
    plt.close(fig)

pts=load('Primary_model_points');ci=load('Primary_model_intervals')
def est(context,model,metric):
    r=match(ci,context=context,model=model,metric=metric).to_dict()
    r.update(context=context);return r
labels={
 'geography_preop_preop_compact_champion_LR':'Geo: 9 inputs (LR)',
 'geography_preop_preop_full_champion_LR':'Geo: 27 inputs (LR)',
 'geography_preop_preop_selected_champion_LR':'Geo: selected (27)',
 'temporal_preop_preop_compact_champion_Voting':'Time: 9 inputs (Voting)',
 'temporal_preop_preop_full_champion_Voting':'Time: 27 inputs (Voting)',
 'temporal_preop_preop_selected_champion_SGBT':'Time: selected (24)',
 'geography_paired_preop_full_champion_Voting':'Geo: preop (Voting)',
 'geography_paired_endop_full_champion_LR':'Geo: endop (LR)',
 'temporal_paired_preop_full_champion_SGBT':'Time: preop (SGBT)',
 'temporal_paired_endop_full_champion_Voting':'Time: endop (Voting)'}
order=[('geography_preop','preop_compact_champion_LR'),('geography_preop','preop_full_champion_LR'),
 ('temporal_preop','preop_compact_champion_Voting'),('temporal_preop','preop_full_champion_Voting')]
paired=[('geography_paired','preop_full_champion_Voting'),('geography_paired','endop_full_champion_LR'),
 ('temporal_paired','preop_full_champion_SGBT'),('temporal_paired','endop_full_champion_Voting')]

# Figure 1. Compact cohort architecture with explicit record units.
fig,aa=figure(2,2,140,left=.04,right=.98,bottom=.08,top=.91,wspace=.12,hspace=.46)
for ax in aa.flat:
    ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
a,b,c,d=aa.flat
panel(a,'a','Source and information-stage populations')
a.set_xlim(0,1);a.set_ylim(0,1)
flow=[('Registry source','4,014 records',.86,'#F1F3F5'),
 ('Adults','4,013 records',.63,'#F1F3F5'),
 ('Eligible preoperative cohort','3,710 records / 205 AKI',.40,'#E8F1F7'),
 ('Paired information cohort','2,428 records / 138 AKI',.17,'#E8F1F7')]
for title,val,y,fc in flow:
    a.add_patch(Rectangle((.02,y-.075),.79,.15,fc=fc,ec='#A7B2BA',lw=.6))
    a.text(.045,y+.021,title,va='center');a.text(.045,y-.029,val,va='center',fontweight='bold')
for y in [.745,.515,.285]:
    a.annotate('',xy=(.41,y-.035),xytext=(.41,y+.035),arrowprops={'arrowstyle':'->','lw':.6,'color':'#66717B'})
a.text(.84,.75,'Age <18\nn = 1',fontsize=6.5,va='center')
a.text(.84,.515,'Renal data\nunavailable\nn = 303',fontsize=6.5,va='center')
a.text(.02,.015,'Patient groups: 3,693 preop; 2,411 paired.',fontsize=6.5)
panel(b,'b','Training and evaluation records',1)
part=load('F1_partitions')
b.text(.01,.95,'Population / line',fontweight='bold')
b.text(.52,.95,'Training',fontweight='bold');b.text(.78,.95,'Evaluation',fontweight='bold')
for i,(pop,line) in enumerate([('preop','geography'),('preop','temporal'),('paired','geography'),('paired','temporal')]):
    y=.77-i*.18;ctx='Geographic' if line=='geography' else 'Temporal'
    tr=match(part,Population=pop,Line=line,Partition='training');te=match(part,Population=pop,Line=line,Partition='validation')
    b.add_patch(Rectangle((.005,y-.07),.025,.11,fc=col(line),ec='none'))
    b.text(.055,y,f'{pop.capitalize()}\n{ctx}',va='center')
    b.text(.52,y,f'{tr.Records:,}\n{tr.Events:,} AKI',va='center')
    b.text(.78,y,f'{te.Records:,}\n{te.Events:,} AKI',va='center')
b.text(.01,.015,'Evaluation overlap: 273 records / 272 patient groups.',fontsize=6.5)
panel(c,'c','Information required by the evaluated systems')
for i,(head,txt,fc) in enumerate([
 ('Preoperative','9 compact / 27 full inputs\n18 fewer fields (66.7%)', '#F1F3F5'),
 ('End of operation','27 preop + 11 intraoperative = 38 inputs\nPaired evaluation on the same records','#E8F1F7'),
 ('POD1 update','POD1 SCr + ratio to baseline\nSeparate conditional cohort: 2,947 / 39 AKI','#F5F0E8')]):
    yy=.82-i*.31
    c.add_patch(Rectangle((.02,yy-.115),.94,.25,fc=fc,ec='#B9C1C7',lw=.6))
    c.text(.04,yy+.058,head,fontweight='bold',va='center')
    c.text(.04,yy-.036,txt,va='center',fontsize=6.5)
panel(d,'d','Public six-input representation',1)
d.add_patch(Rectangle((.02,.42),.94,.50,fc='#EFF5F3',ec='#B9C1C7',lw=.6))
d.text(.055,.84,'INSPIRE: 6,043 records / 253 SCr events',fontweight='bold')
d.text(.055,.68,'Age, sex, creatinine, blood urea nitrogen,\nalbumin and haemoglobin',linespacing=1.6)
d.text(.055,.49,'Separate source random-forest models',fontsize=6.5)
d.text(.02,.28,'Endop uses the recorded postoperative composite.\nPOD1 uses a separate conditional SCr endpoint.',fontsize=6.5,linespacing=1.6,va='top')
record(fig,'Figure_1_Cohorts_and_information',140,['F1_partitions','F1_centres'],'Cohorts, information stages and evaluation architecture')

# Figure 2 source identity; four-panel JAMA main Figure 1.
fig,aa=figure(2,2,160,left=.21,right=.98,bottom=.09,top=.94,hspace=.80,wspace=.85)
for ax,metric,letter,title,xlim,column in [(aa[0,0],'AUROC','A','Discrimination',(.5,.85),0),(aa[0,1],'Brier','B','Prediction error',(0,.08),1)]:
    rows=[]
    for ctx,m in order:
        r=est(ctx,m,metric);r.update(label=labels[ctx+'_'+m],marker='o' if 'compact' in m else 's');rows.append(r)
    forest(ax,rows,metric,xlim=xlim);panel(ax,letter,title,column)
ax=aa[1,0];rows=[];dif=load('F2_preoperative_differences')
for cap in [20,10]:
    for context in ['geography_preop','temporal_preop']:
        r=match(dif,context=context,metric=f'capacity{cap}_captured_per1000').to_dict()
        r.update(label=('Geo' if 'geography' in context else 'Time')+f': {cap}% capacity',marker='o' if cap==10 else 's');rows.append(r)
forest(ax,rows,'27-minus-9 capture per 1,000',null=0,xlim=(-10,10));panel(ax,'C','Capture loss with nine inputs',0)
rows=[]
for context,m in order:
    b=load('Translation_'+context+'_INCREMENT_AND_INVERSE_INTERVALS')
    r=match(b,model=m,metric='increment_captured_per1000').to_dict()
    r.update(label=labels[context+'_'+m],marker='o' if 'compact' in m else 's',context=context);rows.append(r)
forest(aa[1,1],rows,'Additional capture per 1,000',null=0,xlim=(0,np.ceil(max(float(r['upper95']) for r in rows)/5)*5));panel(aa[1,1],'D','Capacity increase: 10% to 20%',1)
footer(fig,'a, b, d: 9 inputs (circles), 27 inputs (squares). c: capacity-specific paired contrasts. Points and 95% bootstrap intervals.')
record(fig,'Figure_2_Nine_input_performance_cost',160,['Primary_model_points','Primary_model_intervals','F2_preoperative_differences','Translation_geography_preop_INCREMENT_AND_INVERSE_INTERVALS','Translation_temporal_preop_INCREMENT_AND_INVERSE_INTERVALS'],'Preoperative information requirements and capture at fixed capacity')

# Figure 3. Stage-system gain with available same-family and overlap-restricted controls.
fig,aa=figure(2,2,160,left=.22,right=.98,bottom=.09,top=.93,hspace=.75,wspace=.84)
rows=[]
for ctx,m in paired:
    r=est(ctx,m,'AUROC');r.update(label=labels[ctx+'_'+m],marker='o' if 'preop' in m else 's');rows.append(r)
forest(aa[0,0],rows,'AUROC',xlim=(.45,.87));panel(aa[0,0],'A','Paired stage systems')
rows=[]
for ctx,m in paired:
    r=est(ctx,m,'Brier');r.update(label=labels[ctx+'_'+m],marker='o' if 'preop' in m else 's');rows.append(r)
forest(aa[0,1],rows,'Brier score',xlim=(.015,.085));panel(aa[0,1],'B','Prediction error',1)
st=load('F3_stage_differences')
lorder=[('geography_paired','Stage systems'),('geography_paired','Same family: LR'),('geography_paired','Same family: Voting'),
 ('temporal_paired','Stage systems'),('temporal_paired','Same family: Voting'),('temporal_paired','Same family: SGBT'),('temporal_paired','No overlap: stage systems')]
alllo=st.loc[st.metric.str.contains('captured_per1000'),'lower95'].min();allhi=st.loc[st.metric.str.contains('captured_per1000'),'upper95'].max()
lims=(np.floor((alllo-1)/5)*5,np.ceil((allhi+1)/5)*5)
for ax,cap,letter,column in [(aa[1,0],10,'C',0),(aa[1,1],20,'D',1)]:
    rows=[]
    for ctx,label in lorder:
        r=match(st,context=ctx,label=label,metric=f'capacity{cap}_captured_per1000').to_dict()
        lab=('Geo' if 'geography' in ctx else 'Time')+': '+label.replace('Same family: ','').replace('Stage systems','systems').replace('No overlap: stage systems','non-overlap')
        r.update(label=lab,marker='s' if label=='Stage systems' else ('x' if 'No overlap' in label else 'o'),filled=label=='Stage systems')
        rows.append(r)
    forest(ax,rows,'Additional AKI captured per 1,000',null=0,xlim=lims);panel(ax,letter,f'{cap}% review capacity',column)
footer(fig,'Endop minus preop; frozen stage-system and same-family comparisons. Non-overlap temporal evaluation: 738 records / 29 AKI.')
record(fig,'Figure_3_Stage_gain_and_consistency',160,['Primary_model_points','Primary_model_intervals','F3_stage_differences'],'Stage-system gains and their consistency across comparisons')

# Figure 4. Observed event capture over the complete retrospective review budget.
fig,aa=figure(2,2,160,left=.18,right=.98,bottom=.17,top=.90,wspace=.67,hspace=.85,ratios=[1.25,1])
incrows=[]
for ax,ctx,letter,label in [(aa[0,0],'geography_preop','a','Geographic: 1,159 records / 72 AKI'),(aa[0,1],'temporal_preop','b','Temporal: 1,683 records / 66 AKI')]:
    curves=load('Translation_'+ctx+'_CAPTURE_REVIEW_CURVE');points=load('Translation_'+ctx+'_INCREMENT_POINTS');bounds=load('Translation_'+ctx+'_INCREMENT_AND_INVERSE_INTERVALS')
    for key,ls,marker in [('compact','-','o'),('full','--','s')]:
        model=next(v for v in points.model.unique() if 'preop_'+key+'_champion_' in v)
        z=curves[curves.model==model].sort_values('reviewed_n');n=float(points[points.model==model].n.iloc[0])
        ax.step(z.reviewed_per1000/10,z.sensitivity*100,where='post',color=col(ctx),ls=ls,lw=.95,label=('9 inputs' if key=='compact' else '27 inputs'))
        r=match(points,model=model)
        for cap in [10,20]:
            k=int(r['reviews'+str(cap)]);v=match(z,reviewed_n=k)
            ax.plot(k/n*100,v.sensitivity*100,marker=marker,ms=3.2,color=col(ctx),mfc=col(ctx) if key=='compact' else 'white',mew=.8,ls='')
        increment_ci=match(bounds,model=model,metric='increment_captured_per1000')
        incrows.append(dict(context=ctx,label=('Geo' if ctx.startswith('geography') else 'Time')+(': 9 inputs' if key=='compact' else ': 27 inputs'),estimate=increment_ci.estimate,lower95=increment_ci.lower95,upper95=increment_ci.upper95,marker=marker,filled=key=='compact'))
    ax.set_xlim(0,100);ax.set_ylim(0,102);ax.set_xticks([0,20,40,60,80,100]);ax.set_yticks([0,20,40,60,80,100]);ax.set_xlabel('Records selected for review (%)');ax.set_ylabel('Recorded AKI captured (%)');ax.axvline(10,color='#B6BABE',lw=.5,ls=':');ax.axvline(20,color='#B6BABE',lw=.5,ls=':');ax.legend(loc='lower right');panel(ax,letter,label,0 if letter=='a' else 1)
gs=aa[1,0].get_subplotspec().get_gridspec();aa[1,0].remove();aa[1,1].remove();ax=fig.add_subplot(gs[1,:]);clean(ax)
forest(ax,incrows,'Extra AKI captured per 1,000 records (20% minus 10% review)',null=0,xlim=(-.5,20));panel(ax,'c','Capture gained when increasing review capacity')
footer(fig,'Retrospective ranked lists. Additional records selected: 116 geographic; 168 temporal. Bars: paired 95% bootstrap intervals.')
record(fig,'Figure_4_Capture_review_capacity',160,['Translation_geography_preop_CAPTURE_REVIEW_CURVE','Translation_temporal_preop_CAPTURE_REVIEW_CURVE','Translation_geography_preop_INCREMENT_POINTS','Translation_temporal_preop_INCREMENT_POINTS','Translation_geography_preop_INCREMENT_AND_INVERSE_INTERVALS','Translation_temporal_preop_INCREMENT_AND_INVERSE_INTERVALS'],'Event capture and review capacity')

# Figure 5. Public ranking, source thresholds and workload.
fig,aa=figure(2,2,150,left=.22,right=.98,bottom=.10,top=.92,hspace=.75,wspace=.82)
pub=load('F5_public_model_points');pci=load('F5_public_model_intervals');bur=load('F5_public_review_and_capture')
for ax,metric,letter,title,scale,xlabel,lims in [
 (aa[0,0],'AUROC','a','Public six-input source models',1,'AUROC',(.5,.85)),
 (aa[0,1],'capacity20_sensitivity','b','Capture at target 20% ranking',100,'Recorded SCr events captured (%)',(0,100))]:
    rows=[]
    for line in ['geography','temporal']:
        r=match(pci,model=line+'__calibrated',metric=metric).to_dict();r.update(line=line,label=('Geographic source' if line=='geography' else 'Temporal source'),marker='o' if line=='geography' else 's');rows.append(r)
    forest(ax,rows,xlabel,scale=scale,xlim=lims);panel(ax,letter,title,0 if letter=='a' else 1)
rules=['Source 10% capacity','Source 20% capacity','Source 80% sensitivity','Target 20% ranking']
for ax,metric,letter,title,scale,xlab in [
 (aa[1,0],'reviewed_per1000','c','Review burden',.1,'Records selected for review (%)'),
 (aa[1,1],'sensitivity','d','Event capture',100,'Recorded SCr events captured (%)')]:
    for y,rule in zip(np.arange(4)[::-1],rules):
        for line,dy in [('geography',.10),('temporal',-.10)]:
            r=match(bur,line=line,rule=rule)
            draw_ci(ax,y+dy,r[metric]*scale,r[metric+'_lower95']*scale,r[metric+'_upper95']*scale,col(line),'o' if line=='geography' else 's')
    ax.set_yticks(np.arange(4)[::-1],rules);ax.set_ylim(-.6,3.6);ax.set_xlim(-1,104);ax.set_xticks([0,20,40,60,80,100]);ax.set_xlabel(xlab);panel(ax,letter,title,0 if letter=='c' else 1)
footer(fig,'Same 6,043 records / 253 SCr events; source-specific RF models. Blue circles: geographic source; vermilion squares: temporal source.')
record(fig,'Figure_5_Public_threshold_transport',150,['F5_public_model_points','F5_public_model_intervals','F5_public_review_and_capture'],'Public ranking and threshold-defined review workload')

# S1. Calibration curves: frozen decile point estimates, no fabricated bands.
cal=load('Calibration_deciles')
fig,aa=figure(2,2,145,left=.12,right=.98,bottom=.10,top=.92,hspace=.78,wspace=.42)
contexts=['geography_preop','temporal_preop','geography_paired','temporal_paired']
lim=np.ceil(max(cal.mean_predicted.max(),cal.observed_rate.max())*10)/10
for ax,ctx,letter,column in zip(aa.flat,contexts,'abcd',[0,1,0,1]):
    z=cal[cal.context==ctx]
    for m in z.model.drop_duplicates():
        zz=z[z.model==m].sort_values('bin')
        label='Compact' if 'compact' in m else ('Selected' if 'selected' in m else ('Preop' if ctx.endswith('paired') and m.startswith('preop') else ('Endop' if m.startswith('endop') else 'Full')))
        style={'Compact':('#444444','o','-'),'Selected':(C['selected'],'^',':'),'Full':(col(ctx),'s','--'),'Preop':('#444444','o','-'),'Endop':(col(ctx),'s','--')}[label]
        ax.plot(zz.mean_predicted,zz.observed_rate,color=style[0],marker=style[1],ls=style[2],markersize=3,lw=.8,label=label)
    ax.plot([0,lim],[0,lim],color='#999999',ls='--',lw=.6);ax.set_xlim(0,lim);ax.set_ylim(0,lim)
    ax.set_xlabel('Mean predicted probability');ax.set_ylabel('Observed event proportion')
    ax.legend(loc='upper left',handlelength=1.8)
    panel(ax,letter,ctx.replace('_',' ').capitalize(),column)
footer(fig,'Decile point estimates; decile record and event counts are included in the source data.')
record(fig,'Supplementary_Figure_1_Calibration',145,['Calibration_deciles'],'Calibration of the ten primary routes')

# S2. All twelve original and target-local adapted routes, within-route comparisons.
up=load('S2_calibration_update_support');upp=load('S2_calibration_update_points')
uorder=[(ctx,m) for ctx in contexts for m in pts[pts.context==ctx].model]+[('INSPIRE_common6','common6_geography'),('INSPIRE_common6','common6_temporal')]
fig,aa=figure(1,2,155,left=.25,right=.98,bottom=.18,top=.92,wspace=.95)
for ax in aa.flat:ax.set_ylim(-.6,len(uorder)-.4)
unames=[]
for i,(ctx,m) in enumerate(uorder):
    y=len(uorder)-1-i
    name=labels.get(ctx+'_'+m,('Public: geo' if m.endswith('geography') else 'Public: time'))
    unames.append(name)
    for method,dy,mk in [('original_V1',.18,'o'),('offset_MAP',0,'s'),('logistic_MAP',-.18,'^')]:
        r=match(up,context=ctx,route=m,comparison=method,metric='Brier')
        draw_ci(aa[0,0],y+dy,r.estimate*1000,r.lower*1000,r.upper*1000,C['reference'] if method=='original_V1' else col('temporal' if 'temporal' in ctx or m.endswith('temporal') else 'geography'),mk,method!='offset_MAP')
    for method,dy,mk in [('offset_MAP',.1,'s'),('logistic_MAP',-.1,'^')]:
        r=match(up,context=ctx,route=m,comparison=method+'_MINUS_original_V1',metric='Brier')
        draw_ci(aa[0,1],y+dy,r.estimate*1000,r.lower*1000,r.upper*1000,col('temporal' if 'temporal' in ctx or m.endswith('temporal') else 'geography'),mk,method!='offset_MAP')
for a in aa.flat:a.set_yticks(np.arange(len(uorder))[::-1],unames)
aa[0,0].set_xlabel('Brier score × 1,000');aa[0,1].set_xlabel('Change in Brier score × 1,000');aa[0,1].axvline(0,color='#999999',lw=.6,ls='--')
panel(aa[0,0],'a','Original and target-local updates');panel(aa[0,1],'b','Update minus original',1)
fig.legend(handles=[Line2D([],[],marker=m,ls='',color='#555555',label=l) for m,l in [('o','Original'),('s','Offset MAP'),('^','Logistic MAP')]],ncol=3,loc='lower center',bbox_to_anchor=(.55,.075))
footer(fig,'Target-local updates use other target-fold labels; intervals reflect resampling conditional on the specified adaptation procedure.')
record(fig,'Supplementary_Figure_2_Local_calibration',155,['S2_calibration_update_points','S2_calibration_update_support'],'Target-local probability updates')

# S3. POD1 remains a distinct conditional endpoint.
pod=load('S3_POD1_LR_points');pdiff=load('S3_POD1_LR_differences')
fig,aa=figure(1,2,100,left=.24,right=.98,bottom=.20,top=.85,wspace=.92)
for ax,metric,letter,scale,xlabel in [(aa[0,0],'AUROC','a',1,'Change in AUROC'),(aa[0,1],'Brier','b',1000,'Change in Brier score × 1,000')]:
    rows=[]
    for ctx in ['geography_all','geography_matched','temporal_all','temporal_matched']:
        contrast='all_pod1__LR MINUS all_preop__LR' if ctx.endswith('all') else 'matched_pod1__LR MINUS matched_endop__LR'
        r=match(pdiff,context=ctx,model=contrast,metric=metric).to_dict()
        z=pod[(pod.context==ctx)&pod.model.str.contains('pod1')].iloc[0]
        r.update(label=('Geo' if 'geography' in ctx else 'Time')+(' all' if ctx.endswith('all') else ' matched')+f'\n{int(z.n):,} records / {int(z.events)} AKI')
        rows.append(r)
    forest(ax,rows,xlabel,null=0,scale=scale);panel(ax,letter,'POD1 update: '+('discrimination' if metric=='AUROC' else 'prediction error'),0 if letter=='a' else 1)
footer(fig,'POD1 adds SCr and the baseline ratio. Eligible cohort: 2,947 records / 39 subsequent SCr events.')
record(fig,'Supplementary_Figure_3_POD1',100,['S3_POD1_LR_points','S3_POD1_LR_differences'],'POD1 updating in the conditional SCr cohort')

# S4. All 24 calibrated LLM conditions, with no outcome-based filtering.
lp=load('S4_LLM_all24_calibrated_points');lc=load('S4_LLM_all24_calibrated_intervals')
fig,aa=figure(2,2,150,left=.24,right=.98,bottom=.11,top=.92,wspace=.95,hspace=.70)
models=[('biomedlm_2_7b','BioMedLM'),('medgemma_4b_it','MedGemma'),('llama3_8b_instruct_ngc_v1','Llama 3')]
for ax,line,stage,letter,column in [
 (aa[0,0],'geography','preop','a',0),(aa[0,1],'temporal','preop','b',1),
 (aa[1,0],'geography','endop','c',0),(aa[1,1],'temporal','endop','d',1)]:
    rows=[]
    for model,title in models:
        for state in ['before','after']:
            name=f'llm_{model}_{state}_{stage}_cal'
            r=match(lc,line=line,model=name,metric='AUROC').to_dict()
            r.update(line=line,label=title+' / '+state,marker='o' if state=='before' else 's',filled=state=='after');rows.append(r)
    forest(ax,rows,'AUROC',xlim=(.2,1))
    panel(ax,letter,('Geographic' if line=='geography' else 'Temporal')+' / '+stage,column)
footer(fig,'Before/after refers to supervised training; both conditions use source-development calibration. All 54 paired comparisons are supplied.')
record(fig,'Supplementary_Figure_4_LLM_all24',150,['S4_LLM_all24_calibrated_points','S4_LLM_all24_calibrated_intervals','LLM_all54_comparisons_all_metrics'],'All calibrated medical-language-model conditions')

# S6. Conditional refit variability, not patient-level confidence intervals.
stab=load('S5_conditional_stability').sort_values('route')
def route_label(s):
    return s.replace('geography_','Geo ').replace('temporal_','Time ').replace('paired_','paired ').replace('preop_','preop ').replace('endop_','endop ').replace('_',' ').replace('preop preop ','preop ')
fig,aa=figure(1,2,140,left=.23,right=.98,bottom=.13,top=.91,wspace=.90)
for ax,stem,letter,title,xlabel in [
 (aa[0,0],'top20_overlap_fraction','a','Top-20% list overlap','List overlap (%)'),
 (aa[0,1],'training_sensitivity80_fixed_cut_flip','b','Source-cut classification changes','Classification changed (%)')]:
    rows=[]
    for _,r in stab.iterrows():
        rows.append(dict(context=r.route,label=route_label(r.route),estimate=r[stem+'_mean'],lower95=r[stem+'_p025'],upper95=r[stem+'_p975']))
    forest(ax,rows,xlabel,scale=100,xlim=(0,100));panel(ax,letter,title,0 if letter=='a' else 1)
footer(fig,'Means and central 95% ranges across 200 conditional patient-group refits per route. Original finite-probe failures = zero.')
record(fig,'Supplementary_Figure_5_Stability',140,['S5_conditional_stability'],'Conditional prediction and review-list stability')

# S7. Recorded interval coverage uses all events, including untriggered events.
lead=load('S6_recorded_window_points');lci=load('S6_recorded_window_intervals')
fig,aa=figure(1,2,110,left=.17,right=.98,bottom=.20,top=.86,wspace=.65)
ax=aa[0,0]
for line,mk,ls in [('geography','o','-'),('temporal','s','--')]:
    xx=[12,24,48];yy=[];lo=[];hi=[]
    for h in xx:
        r=match(lci,rule=line+'__retrospective_capacity20',metric=f'all_AKI_detected_atleast{h}h_fraction')
        yy.append(r.estimate*100);lo.append(r.lower95*100);hi.append(r.upper95*100)
    ax.errorbar(xx,yy,yerr=[np.array(yy)-lo,np.array(hi)-yy],color=col(line),marker=mk,ls=ls,markersize=3,capsize=2,label=('Geographic' if line=='geography' else 'Temporal')+' source')
ax.set_xticks([12,24,48]);ax.set_ylim(0,100);ax.set_xlabel('Minimum recorded interval (hours)');ax.set_ylabel('All recorded SCr events covered (%)');ax.legend(loc='upper right')
panel(ax,'a','Target 20% ranking')
ax=aa[0,1]
rules=[('source_sensitivity80','Source 80% sensitivity'),('retrospective_capacity10','Target 10% ranking'),('retrospective_capacity20','Target 20% ranking')]
for y,(rule,lab) in zip(np.arange(3)[::-1],rules):
    for line,dy,mk in [('geography',.09,'o'),('temporal',-.09,'s')]:
        r=match(lci,rule=line+'__'+rule,metric='all_AKI_detected_atleast24h_fraction')
        draw_ci(ax,y+dy,r.estimate*100,r.lower95*100,r.upper95*100,col(line),mk)
ax.set_yticks(np.arange(3)[::-1],[x[1] for x in rules]);ax.set_ylim(-.6,2.6);ax.set_xlim(0,100);ax.set_xlabel('All SCr events with interval ≥24 h (%)')
panel(ax,'b','Rule-specific ≥24-hour recorded coverage',1)
footer(fig,'Denominator: all 253 SCr events. Interval: surgery start to first qualifying recorded SCr; no alert or result-release timestamp is inferred.')
record(fig,'Supplementary_Figure_6_Recorded_intervals',110,['S6_recorded_window_points','S6_recorded_window_intervals'],'Recorded surgery-to-qualifying-SCr interval coverage')

# S8. ALT difference-in-differences, preserving all four contexts.
alt=load('S7_ALT_all_metrics')
fig,aa=figure(1,2,105,left=.21,right=.98,bottom=.20,top=.86,wspace=.80)
ar=[]
for cap in [10,20]:
    rows=[]
    for ctx in contexts:
        r=match(alt,context=ctx,comparison='change_in_gain_omit_ALT_minus_control',metric=f'capacity{cap}_captured_per1000')
        rows.append(dict(context=ctx,label=ctx.replace('geography','Geo').replace('temporal','Time').replace('_',' '),estimate=r.estimate,lower95=r.lower,upper95=r.upper))
    ar.extend(rows)
    forest(aa[0,0 if cap==10 else 1],rows,'Change in capture increment per 1,000',null=0)
    panel(aa[0,0 if cap==10 else 1],'a' if cap==10 else 'b',f'{cap}% review capacity',0 if cap==10 else 1)
lim=max(abs(x[k]) for x in ar for k in ['lower95','upper95']);lim=np.ceil((lim+1)/2)*2
for a in aa.flat:a.set_xlim(-lim,lim)
footer(fig,'ALT-omission increment minus the matched-control increment. Fixed current-source models and paired bootstrap intervals.')
record(fig,'Supplementary_Figure_7_ALT',105,['S7_ALT_all_metrics'],'ALT omission and changes in information increments')

# S9. All 16 allocation scenarios, including null/adverse outcomes.
mc=load('F4_allocation_16_scenarios')
fig,aa=figure(2,2,140,left=.23,right=.98,bottom=.12,top=.92,wspace=.88,hspace=.72)
lim=max(abs(mc.conditional_difference_low_per1000.min()),abs(mc.conditional_difference_high_per1000.max()))
lim=np.ceil((lim+1)/5)*5
for ax,ctx,letter,column in zip(aa.flat,contexts,'abcd',[0,1,0,1]):
    rows=[]
    for uf in [.25,.5]:
        for rf in [.1,.2]:
            r=match(mc,context=ctx,upgrade_fraction=uf,review_fraction=rf)
            rows.append(dict(context=ctx,label=f'{int(uf*100)}% upgrade / {int(rf*100)}% review',
              estimate=r.selective_minus_random_mean_per1000,lower95=r.conditional_difference_low_per1000,upper95=r.conditional_difference_high_per1000,
              marker='o' if uf==.25 else 's'))
    forest(ax,rows,'Selective − random mean capture\n(per 1,000 records)',null=0,xlim=(-lim,lim))
    panel(ax,letter,ctx.replace('_',' ').capitalize(),column)
footer(fig,'Intervals are conditional random-allocation ranges. Policy contrasts include source-calibration behavior (including a negative random-policy slope).')
record(fig,'Supplementary_Figure_8_Allocation',140,['F4_allocation_16_scenarios'],'All constrained information-allocation scenarios')

# S10. Reference margins are discrete reporting values; the exact upper bounds remain visible.
mg=load('S9_MARGIN_SENSITIVITY');joint=load('S9_CROSS_CONTEXT_SENSITIVITY');tips=load('S9_TIPPING_POINT_SUMMARY')
fig,aa=figure(1,2,125,left=.20,right=.98,bottom=.20,top=.86,wspace=.90)
rows=[]
for cap in [20,10]:
    for ctx in ['geography_preop','temporal_preop']:
        r=match(tips,context=ctx,review_capacity_percent=cap)
        rows.append(dict(context=ctx,label=('Geo' if ctx.startswith('geography') else 'Time')+f': {cap}% review',estimate=r.loss_full27_minus_compact9,lower95=r.lower95,upper95=r.upper95,marker='s' if cap==20 else 'o',filled=cap==20))
forest(aa[0,0],rows,'Nine-input capture loss per 1,000',null=0,xlim=(-6,10))
panel(aa[0,0],'a','Original paired estimates and intervals')
ax=aa[0,1];gridrows=[]
for cap in [20,10]:
    for ctx in ['geography_preop','temporal_preop']:
        z=mg[(mg.context==ctx)&(mg.review_capacity_percent==cap)].sort_values('delta_capture_per1000')
        u=float(z.loss_upper_bound.iloc[0]);gridrows.append((('Geo' if ctx.startswith('geography') else 'Time')+f' / {cap}%',z.delta_capture_per1000.to_numpy(),z.bound_strictly_below_margin.astype(bool).to_numpy(),u,col(ctx)))
    z=joint[joint.review_capacity_percent==cap].sort_values('delta_capture_per1000')
    gridrows.append((f'Both / {cap}%',z.delta_capture_per1000.to_numpy(),z.both_context_bounds_below_margin.astype(bool).to_numpy(),float(z.both_contexts_continuous_boundary.iloc[0]),C['reference']))
for y,(label,xx,compatible,u,color) in zip(np.arange(6)[::-1],gridrows):
    for x,ok in zip(xx,compatible):ax.plot(x,y,marker='s',ms=3.1,mfc=color if ok else 'white',mec=color if ok else '#A4A9AE',mew=.6,ls='')
    ax.vlines(u,y-.22,y+.22,color='#222222',lw=.8,zorder=5)
ax.set_yticks(np.arange(6)[::-1],[x[0] for x in gridrows]);ax.set_ylim(-.6,5.6);ax.set_xlim(-.3,10.3);ax.set_xticks(list(range(11)))
ax.set_xlabel('Reference margin δ\n(events per 1,000 records)');panel(ax,'b','Complete discrete reference grid',1)
fig.legend(handles=[Line2D([],[],marker='s',ms=3,ls='',color='#555555',label='U < δ'),Line2D([],[],marker='s',ms=3,ls='',mfc='white',mec='#A4A9AE',label='U ≥ δ'),Line2D([],[],marker='|',ms=7,ls='',color='#222222',label='Exact upper bound U')],ncol=3,loc='lower center',bbox_to_anchor=(.55,.075),columnspacing=1.5)
footer(fig,'Post hoc reference grid, step 0.5. Both = logical intersection. Original 2,000 paired patient-group bootstrap samples.')
record(fig,'Supplementary_Figure_9_Loss_margin',125,['S9_MARGIN_SENSITIVITY','S9_CROSS_CONTEXT_SENSITIVITY','S9_TIPPING_POINT_SUMMARY'],'Capture-loss bounds across reference margins')

# S10. Virtual clinical use: observed capture + explicit work/effect assumptions.
fig,aa=figure(2,2,160,left=.21,right=.98,bottom=.10,top=.93,hspace=.78,wspace=.82)
oper=load('F4_representative_operating_rules');wr=load('F4_representative_workload_assumptions');ef=load('F4_representative_effect_assumptions')
regimens=[('geography_preop',10),('geography_preop',20),('temporal_preop',10),('temporal_preop',20)]
reglabels=[('Geo' if 'geography' in ctx else 'Time')+f': {cap}% review' for ctx,cap in regimens]
rows=[]
for (ctx,cap),lab in zip(regimens,reglabels):
    r=match(oper,context=ctx,rule=f'assessment_capacity{cap}')
    z=est(ctx,r.model,f'capacity{cap}_captured_per1000')
    z.update(label=lab);rows.append(z)
forest(aa[0,0],rows,'AKI captured per 1,000',xlim=(0,55))
panel(aa[0,0],'a','Nine-input review policies')
ax=aa[0,0]
for y,(ctx,cap) in zip(np.arange(4)[::-1],regimens):
    r=match(oper,context=ctx,rule=f'assessment_capacity{cap}')
    ax.text(.98,(y+.65)/4.3,f'{int(r.captured_n)} / {int(r.missed_n)}',transform=ax.transAxes,ha='right',va='center',fontsize=6.5)
ax.text(.98,1.0,'Captured / missed',transform=ax.transAxes,ha='right',va='bottom',fontsize=6.5)
ass=[('Brief review',5,0),('Intermediate review',10,.5),('Extended review',15,1)]
for ax,metric,letter,title,xlabel in [
 (aa[0,1],'review_hours_per1000','b','Assumed review effort','Review hours per 1,000'),
 (aa[1,0],'additional_tests_per1000','c','Assumed additional testing','Additional tests per 1,000')]:
    ys=np.arange(6)[::-1]
    ll=[]
    for i,(lab,minutes,fraction) in enumerate(ass):
        for j,cap in enumerate([10,20]):
            y=ys[i*2+j];ll.append(f'{minutes} min / {int(fraction*100)}% tests\n{cap}% review')
            for ctx,dy in [('geography_preop',.08),('temporal_preop',-.08)]:
                r=match(wr,context=ctx,rule=f'assessment_capacity{cap}',assumption_label=lab)
                draw_ci(ax,y+dy,r[metric],r[metric+'_lower95'],r[metric+'_upper95'],col(ctx),'o' if ctx.startswith('geography') else 's')
    ax.set_yticks(ys,ll);ax.set_ylim(-.6,5.6);ax.set_xlim(0,54 if metric.startswith('review') else 220);ax.set_xlabel(xlabel);panel(ax,letter,title,1 if letter=='b' else 0)
ax=aa[1,1]
for y,(ctx,cap) in zip(np.arange(4)[::-1],regimens):
    for rrr,dy,mk in [(-.1,-.14,'v'),(0,0,'x'),(.2,.14,'o')]:
        r=match(ef,context=ctx,rule=f'assessment_capacity{cap}',assumed_RRR=rrr)
        draw_ci(ax,y+dy,r.hypothetical_events_averted_per1000,r.conditional_sampling_lower95,r.conditional_sampling_upper95,col(ctx),mk)
ax.set_yticks(np.arange(4)[::-1],reglabels);ax.set_ylim(-.6,3.6);ax.axvline(0,color='#999999',lw=.6,ls='--')
ax.set_xlim(-5,9);ax.set_xlabel('Hypothetical events averted per 1,000');panel(ax,'d','Assumed effect at 75% uptake',1)
handles=[Line2D([],[],marker=m,color='#555555',ls='',markersize=3,label=l) for m,l in [('v','RRR −10%'),('x','RRR 0%'),('o','RRR 20%')]]
fig.legend(handles=handles,loc='center',bbox_to_anchor=(.76,.50),ncol=3,handlelength=1,columnspacing=.9)
footer(fig,'Blue circles: geographic; vermilion squares: temporal (b–c). Work and treatment-effect parameters are scenario assumptions.')
record(fig,'Supplementary_Figure_10_Virtual_clinical_use',160,['F4_representative_operating_rules','F4_representative_workload_assumptions','F4_representative_effect_assumptions','Primary_model_intervals'],'Observed capture and assumption-based clinical-use trade-offs')


# S11. Same-family source-cohort bridge; input-specific frozen workflows and paired intervals.
bri=pd.concat([load('Translation_geography_BRIDGE_INTERVALS'),load('Translation_temporal_BRIDGE_INTERVALS')],ignore_index=True)
fig,aa=figure(1,2,115,left=.20,right=.98,bottom=.20,top=.86,wspace=.90)
for ax,metric,letter,title,xlabel,xlim in [(aa[0,0],'AUROC','a','Change in AUROC','LR AUROC difference',(-.06,.08)),(aa[0,1],'capacity20_captured_per1000','b','Change in capture at 20% review','Additional AKI captured per 1,000',(-6,9))]:
    rows=[]
    for ctx in ['geography','temporal']:
        for model,label,marker,filled in [('nine','9 − 6 inputs','o',True),('27','27 − 6 inputs','s',False)]:
            r=match(bri,context=ctx,model=model+'_LR__calibrated MINUS six_LR__calibrated',metric=metric)
            rows.append(dict(context=ctx,label=('Geo: ' if ctx=='geography' else 'Time: ')+label,estimate=r.estimate,lower95=r.lower95,upper95=r.upper95,marker=marker,filled=filled))
    forest(ax,rows,xlabel,null=0,xlim=xlim);panel(ax,letter,title,0 if letter=='a' else 1)
footer(fig,'Source-calibrated LR workflows; input-specific imputation and hyperparameters retained. All raw/calibrated views: Table S15.')
record(fig,'Supplementary_Figure_11_LR_input_bridge',115,['Translation_geography_BRIDGE_INTERVALS','Translation_temporal_BRIDGE_INTERVALS','Translation_BRIDGE_POINT_ESTIMATES'],'Six-, nine- and 27-input source-cohort LR comparisons')

# S12. Frozen public coded-history workflows; threshold burden and achieved coverage together.
public9_points=load('Coded9_POINT_ESTIMATES');public9_intervals=load('Coded9_INTERVALS');public9_summary=load('Coded9_PUBLIC9_REVIEWED_SUMMARY');public9_coverage=load('Coded9_SOURCE_THRESHOLD_COVERAGE')
fig,aa=figure(2,2,160,left=.23,right=.98,bottom=.14,top=.92,wspace=.93,hspace=.92)
public9_order=[('geography__six_RF__calibrated','Geo: 6 RF'),('geography__coded9_LR__calibrated','Geo: 9 LR'),('temporal__six_RF__calibrated','Time: 6 RF'),('temporal__coded9_Voting__calibrated','Time: 9 voting')]
for ax,metric,letter,title,xlabel,scale,limits in [(aa[0,0],'AUROC','a','Public workflow discrimination','AUROC',1,(.5,.85)),(aa[0,1],'Brier','b','Public workflow prediction error','Brier score × 1,000',1000,(30,50))]:
    public9_rows=[]
    for name,label in public9_order:
        r=match(public9_intervals,model=name,metric=metric)
        public9_rows.append(dict(context=name,label=label,estimate=r.estimate,lower95=r.lower95,upper95=r.upper95,marker='s' if 'coded9' in name else 'o',filled='coded9' in name))
    forest(ax,public9_rows,xlabel,scale=scale,xlim=limits);panel(ax,letter,title,0 if letter=='a' else 1)
ax=aa[1,0];public9_rows=[]
for _,r in public9_summary.iterrows():
    public9_rows.append(dict(context=r.source_line,label=('Geo: 9 LR − 6 RF' if r.source_line=='geography' else 'Time: 9 voting − 6 RF'),estimate=r.capture_difference_per1000,lower95=r.capture_difference_lower95,upper95=r.capture_difference_upper95))
forest(ax,public9_rows,'Additional AKI captured per 1,000\n(coded 9 minus common 6)',null=0,xlim=(-2,3));panel(ax,'C','Paired capture at 20% review')
ax=aa[1,1];ax.axis('off');panel(ax,'D','Source 80%-sensitivity thresholds',1)
for x,head in [(0,'Workflow'),(.43,'Selected\nn (%)'),(.86,'Captured/253\nTarget\nsens. (%)')]:ax.text(x,.93,head,ha='left' if x==0 else 'center',va='top',fontsize=6.5,fontweight='bold',transform=ax.transAxes)
for y,(_,r) in zip([.68,.49,.30,.11],public9_coverage.iterrows()):
    label=('Geo\n' if r.source_line=='geography' else 'Time\n')+('6 RF' if r.model=='six_RF' else ('9 LR' if r.source_line=='geography' else '9 voting'))
    vals=[label,f'{int(r.reviewed_n):,}\n({r.reviewed_percent:.1f}%)',f'{int(r.captured_n)}\n({r.target_sensitivity_percent:.2f}%)']
    for x,val in zip([0,.43,.86],vals):ax.text(x,y,val,ha='left' if x==0 else 'center',va='center',fontsize=6.5,color=col(r.source_line),transform=ax.transAxes)
footer(fig,'Same 6,043 records / 253 SCr events; source-calibrated workflows. Bars: 95% paired-patient bootstrap intervals. Raw views: Table S16.')
record(fig,'Supplementary_Figure_12_Public_coded9',160,['Coded9_POINT_ESTIMATES','Coded9_INTERVALS','Coded9_PUBLIC9_REVIEWED_SUMMARY','Coded9_SOURCE_THRESHOLD_COVERAGE'],'Public coded-history nine-input workflows and achieved coverage')

manifest.sort(key=lambda x:(0 if x['figure'].startswith('Figure_') else 1,int(x['figure'].split('_')[1 if x['figure'].startswith('Figure_') else 2])))
layouts.sort(key=lambda x:(0 if x['figure'].startswith('Figure_') else 1,int(x['figure'].split('_')[1 if x['figure'].startswith('Figure_') else 2])))
(W/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
(W/'qa/LAYOUT_AUDIT.json').write_text(json.dumps(layouts,indent=2,default=str)+'\n')
(W/'qa/TEXT_SIZE_AUDIT.json').write_text(json.dumps({'minimum_pt':min(x['size_pt'] for x in textsizes),'maximum_pt':max(x['size_pt'] for x in textsizes),'count':len(textsizes)},indent=2)+'\n')
print(json.dumps({'figures':len(manifest),'final':args.final,'clipped_texts':sum(len(x['canvas_clipped_text']) for x in layouts),'minimum_font':min(x['size_pt'] for x in textsizes)}))
