from pathlib import Path
import os,json,hashlib
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg')
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];F=R/'figures';F.mkdir(exist_ok=True);SUB=F/'subpanels';SUB.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'savefig.facecolor':'white'})
D=pd.read_csv(R/'results/POLICY_PAIRED_DIFFERENCES.csv');L=pd.read_csv(R/'results/LOW_RISK_EVENTS.csv');M=pd.read_csv(R/'results/MIMIC_DELETION_SENSITIVITY.csv');W=pd.read_csv(R/'results/DAILY_SUMMARY.csv');manifest=[]
colors={'geography':'#0072B2','temporal':'#D55E00'};styles=['existing_selective25','highest_base_risk25','near_training_cutoff25','random25','all_collection'];labels=['Existing selective gate','Highest base risk','Nearest training cutoff','Seeded random acquisition','Collect all additional fields']
contexts=['geography_preop','temporal_preop','geography_paired','temporal_paired'];titles=['Geographic evaluation: 9 → 27 inputs','Temporal evaluation: 9 → 27 inputs','Geographic evaluation: 27 → 38 inputs','Temporal evaluation: 27 → 38 inputs']
def policy(ax,idx):
 ctx=contexts[idx];z=D[(D.context==ctx)&D.cost_ratio.eq(.1)].set_index('policy').loc[styles];v=z.delta_captured_per1000.to_numpy();ax.errorbar(v,np.arange(5),xerr=np.vstack([v-z.lo.to_numpy(),z.hi.to_numpy()-v]),fmt='o',ms=4,color=colors[ctx.split('_')[0]],elinewidth=1.1,capsize=2);ax.axvline(0,color='#777777',lw=.8,ls='--');ax.set_yticks(np.arange(5),labels);ax.invert_yaxis();ax.set_xlim(-13,18);ax.set_xlabel('Difference in captured AKI events per 1,000');ax.set_title(chr(65+idx)+'  '+titles[idx],loc='left',pad=10);ax.grid(axis='x',alpha=.15);ax.text(.02,-.3,'Reference: expanded base review at the same total cost\nBudget anchor: 25% acquisition + 20% review; cost ratio = 0.1\n95% paired patient-group bootstrap intervals; fixed models/rules',transform=ax.transAxes,fontsize=7.5,va='top')
def low(ax):
 z=L[L.rule=='training_sensitivity80'].copy();names=[];val=[];lo=[];hi=[];cc=[]
 for ctx in contexts:
  for role in ['base','upgrade']:
   q=z[(z.context==ctx)&(z.role==role)].iloc[0];names.append(('Geo' if ctx.startswith('geo') else 'Time')+' '+('preop' if ctx.endswith('preop') else 'paired')+f' / {int(q.inputs)} inputs  ({int(q.low_risk_events)}/{int(q.low_risk_n)})');val.append(q.observed_low_risk_event_rate*100);lo.append((q.patient_boot_lo if q.low_risk_events>0 else q.wilson_lo)*100);hi.append((q.patient_boot_hi if q.low_risk_events>0 else q.wilson_hi)*100);cc.append(colors[ctx.split('_')[0]])
 for i in range(8):
  ax.hlines(i,lo[i],hi[i],color=cc[i],lw=1.1);ax.plot(val[i],i,'o',ms=4,color=cc[i])
 ax.set_yticks(range(8),names);ax.invert_yaxis();ax.set_xlim(-.3,31);ax.set_xlabel('Observed event rate in the low-risk group, %');ax.set_title('A  Low-risk events at frozen source thresholds',loc='left',pad=10);ax.grid(axis='x',alpha=.15);ax.text(.01,-.24,'Source training sensitivity 80%; 95% patient-group bootstrap intervals\nZero events in 9 distinct patients: Wilson interval shown instead.',transform=ax.transAxes,fontsize=7.5,va='top')
def deletion(ax):
 names={'first_per_calendar_day':'First per day','odd_calendar_days':'Odd days','even_calendar_days':'Even days','calendar_day1_only':'Day 1 only','calendar_days1_2_only':'Days 1–2 only'};markers=['o','s','^','D','v']
 for pop,col,label in [('all_observed_records','#0072B2','All measured records: 759 / 59 events'),('geography_low_risk6','#D55E00','Geographic six-input low risk: 258 / 11 events')]:
  for k,(mode,name) in enumerate(names.items()):
   q=M[(M.population==pop)&(M['mode']==mode)].iloc[0];x=q.deleted_fraction*100;y=q.lost_known_fraction*100;lo=q.lost_event_Wilson_lo*100;hi=q.lost_event_Wilson_hi*100;ax.vlines(x,lo,hi,color=col,lw=.8);ax.plot(x,y,markers[k],color=col,ms=5,label=label if k==0 else None)
   if pop=='all_observed_records':ax.annotate(name,(x,y),xytext=({'even_calendar_days':(4,17),'calendar_days1_2_only':(-67,-20)}.get(mode,(4,6))),textcoords='offset points',fontsize=7.5)
 ax.set_xlim(0,90);ax.set_ylim(-3,85);ax.set_xlabel('Recorded postoperative SCr measurements deleted, %');ax.set_ylabel('Known recorded events lost, %');ax.set_title('B  MIMIC six-input sensitivity: measurement deletion',loc='left',pad=10);ax.legend(loc='upper left',frameon=False,fontsize=7.3);ax.grid(alpha=.15);ax.text(.01,-.24,'Exact calendar-ratio endpoint retained as the reference.\nEvent denominators differ by group; intervals are Wilson 95%.',transform=ax.transAxes,fontsize=7.5,va='top')
def demand(ax,line,letter):
 z=W[(W.context.str.startswith(line))&W.denominator.eq('active_operation_days')&W.metric.eq('planned_final_reviews')].copy();z=z.sort_values(['context','center']);names=[]
 for _,q in z.iterrows():names.append(('9 → 27' if q.context.endswith('preop') else '27 → 38')+f' / centre {int(q.center)}  ({int(q.days)} active days)')
 y=np.arange(len(z));v=z['mean'].to_numpy();ax.errorbar(v,y,xerr=np.vstack([v-z.mean_block7_lo.to_numpy(),z.mean_block7_hi.to_numpy()-v]),fmt='o',ms=4,color=colors[line],capsize=2,label='Mean, 95% 7-day block interval');ax.scatter(z.p95,y,marker='s',s=20,facecolors='none',edgecolors='#555555',label='95th percentile');ax.scatter(z['max'],y,marker='|',s=55,c='#111111',label='Maximum');ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(-.1,3.6);ax.set_xlabel('Potential final reviews per active surgery day');ax.set_title(letter+'  '+('Geographic' if line=='geography' else 'Temporal')+' evaluation: study-cohort demand',loc='left',pad=10);ax.grid(axis='x',alpha=.15);ax.legend(loc='lower right',frameon=False,fontsize=7.1);ax.text(.01,-.24,'Operation-date indexed; one final-policy review per record.\nActual alert arrivals, completed actions and shift clocks are unavailable.',transform=ax.transAxes,fontsize=7.5,va='top')
def export(name,functions,size,shape):
 fig,axes=plt.subplots(*shape,figsize=size,squeeze=False);fig.subplots_adjust(left=.25,right=.97,bottom=.19,top=.92,hspace=.7,wspace=.85)
 for i,fun in enumerate(functions):fun(axes.ravel()[i])
 fig.savefig(F/(name+'.pdf'),bbox_inches='tight');fig.savefig(F/(name+'.png'),dpi=140,bbox_inches='tight');plt.close(fig)
 for i,fun in enumerate(functions):
  fig,ax=plt.subplots(figsize=(7,4.2 if name!='CS3_daily_demand' else 5));fig.subplots_adjust(left=.43,right=.97,bottom=.27,top=.88);fun(ax);key=name+'_'+chr(65+i);fig.savefig(SUB/(key+'.pdf'),bbox_inches='tight');fig.savefig(SUB/(key+'.png'),dpi=120,bbox_inches='tight');plt.close(fig);manifest.append({'figure':name,'panel':chr(65+i),'pdf':str((SUB/(key+'.pdf')).relative_to(R)),'png':str((SUB/(key+'.png')).relative_to(R))})
export('CS1_same_budget_policy',[lambda ax,i=i:policy(ax,i) for i in range(4)],(15,9),(2,2))
export('CS2_monitoring_dependence',[low,deletion],(15,4.6),(1,2))
export('CS3_daily_demand',[lambda ax:demand(ax,'geography','A'),lambda ax:demand(ax,'temporal','B')],(15,6),(1,2))
inputs=[R/'results'/n for n in ['POLICY_PAIRED_DIFFERENCES.csv','LOW_RISK_EVENTS.csv','MIMIC_DELETION_SENSITIVITY.csv','DAILY_SUMMARY.csv']];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(R/'figures/FIGURE_SOURCE_MANIFEST.json').write_text(json.dumps({'new_figures':3,'independent_subpanels':8,'panels':manifest,'source_csvs':{str(p.relative_to(R)):sha(p) for p in inputs},'code':sha(Path(__file__)),'matplotlib':matplotlib.__version__,'original_20_figures_unchanged':True,'SFL_reference':'Read-only forest-family search; no template selected, materialized, or executed; Python figures written from aggregate tables','inference':'All new policy/monitoring comparisons are posthoc; no intervention effects or actual alert-arrival workflow measured'},indent=2))
print('PLOTTED_3_FIGURES_8_INDEPENDENT_SUBPANELS')
