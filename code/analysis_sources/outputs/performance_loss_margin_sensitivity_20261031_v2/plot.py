from pathlib import Path
import json,csv,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
T=Path(__file__).resolve().parent
read=lambda p:list(csv.DictReader(Path(p).open()))
data=read(T/'MARGIN_SENSITIVITY.csv')
order=[('geography_preop',20),('temporal_preop',20),('geography_preop',10),('temporal_preop',10)]
labels=['Geographic | 20% review','Temporal | 20% review','Geographic | 10% review','Temporal | 10% review']
grid=np.arange(0,10.01,.5)
matrix=np.array([[int(next(r['bound_strictly_below_margin'] for r in data if r['context']==context and int(r['review_capacity_percent'])==quota and float(r['delta_capture_per1000'])==d)) for d in grid] for context,quota in order])
tips=read(T/'TIPPING_POINT_SUMMARY.csv')
ordered=[next(r for r in tips if (r['context'],int(r['review_capacity_percent']))==o) for o in order]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','figure.dpi':150,'savefig.dpi':300})
fig,axes=plt.subplots(1,2,figsize=(12.8,4.8),gridspec_kw={'width_ratios':[1,1.5]},sharey=True)
fig.subplots_adjust(left=.19,right=.985,bottom=.29,top=.77,wspace=.13)
for y,r in enumerate(ordered):
 a,l,u=[float(r[k]) for k in ['loss_full27_minus_compact9','lower95','upper95']]
 axes[0].plot([l,u],[y,y],color='#0072B2',lw=2)
 axes[0].plot([l,l],[y-.07,y+.07],color='#0072B2',lw=1)
 axes[0].plot([u,u],[y-.07,y+.07],color='#0072B2',lw=1)
 axes[0].plot(a,y,'o',color='#1B1B1B',ms=5)
 axes[0].text(u+.25,y+.19,f'U = {u:.2f}',fontsize=8,color='#4A4A4A',va='center')
axes[0].axvline(0,color='#707070',lw=.8,ls='--')
axes[0].set_yticks(range(4),labels);axes[0].set_xlim(-6,10)
axes[0].set_xticks([-5,0,5,10]);axes[0].set_xlabel('Capture loss per 1,000 operations')
axes[0].set_title('A  Observed loss and original 95% interval',loc='left',pad=15)
axes[0].grid(axis='x',alpha=.13)
edges=np.arange(-.25,10.26,.5)
axes[1].pcolormesh(edges,np.arange(-.5,4.5,1),matrix,cmap=ListedColormap(['#E5E5E5','#0072B2']),vmin=0,vmax=1,edgecolors='white',linewidth=.8,shading='flat',rasterized=False)
for y,r in enumerate(ordered):
 u=float(r['upper95'])
 axes[1].plot([u,u],[y-.36,y+.36],color='#1A1A1A',lw=1.6)
axes[1].set_xticks(np.arange(0,11));axes[1].set_xticks(grid,minor=True)
axes[1].set_xlabel('Tolerated capture loss, delta (per 1,000)')
axes[1].set_title('B  Original upper bound compared with each margin',loc='left',pad=15)
axes[1].tick_params(axis='y',left=False,labelleft=False)
for ax in axes:
 ax.axhline(1.5,color='#666666',ls=':',lw=.8)
 ax.set_ylim(3.5,-.5)
fig.suptitle('Sensitivity to the tolerated loss: 9-input versus 27-input models',x=.19,y=.96,ha='left',fontsize=14,fontweight='bold')
fig.text(.19,.86,'20% review is primary; 10% is supportive. Positive loss means fewer AKI events captured by the 9-input model.',fontsize=9.3)
fig.legend(handles=[Patch(facecolor='#E5E5E5',label='Upper loss bound >= delta'),Patch(facecolor='#0072B2',label='Upper loss bound < delta')],loc='lower left',bbox_to_anchor=(.185,.125),ncol=2,frameon=False,fontsize=9.2)
fig.text(.19,.065,'Post hoc exploratory margins: 0 to 10, step 0.5. No margin has been independently validated as clinically acceptable.',fontsize=8.7)
fig.text(.19,.025,'U is the original 97.5th percentile of the paired hospital-stratified patient bootstrap (2,000 replicates). Black marks show U.',fontsize=8.7)
for ext in ['pdf','svg','png']:fig.savefig(T/('MARGIN_SENSITIVITY.'+ext),dpi=300,facecolor='white',bbox_inches='tight')
plt.close(fig)
receipt={'status':'RENDERED_PENDING_ACTUAL_PIXEL_REVIEW','matplotlib':matplotlib.__version__,'numpy':np.__version__,'clinical_data_sent_to_external_service':False,'original_matplotlib_figure':True,'figure_library_templates_used':False,'academic_plotting_workflow':'numeric data chart; no generative images','grid_cells':int(matrix.size)}
(T/'FIGURE_RENDER_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
