"""Plot verified locked-pipeline risk curves; scientific review figures."""
from pathlib import Path
import json,hashlib,os,math,resource
from datetime import datetime,timezone,timedelta
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
SOURCE=ROOT/'outputs/locked_risk_curves_20261024_v1';OUT=ROOT/'outputs/final_analysis_delivery_20261025_v1/figures/risk_curves';OUT.mkdir(mode=0o700,parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
stage=read(SOURCE/'STAGE_CURVE_ACCEPTANCE.json');assert len(stage['receipts'])==10
for f,h in read(SOURCE/'CODE_LOCK.json').items():assert sha(ROOT/'work/locked_risk_curves_20261024_v1'/f)==h
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
 'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,
 'lines.linewidth':1.7,'pdf.fonttype':42,'ps.fonttype':42})
labels={'Age':'Age (years)','PreopCr':'Preoperative creatinine (umol/L)','PreopBUN':'Preoperative urea nitrogen (mmol/L)',
 'PreopAlb':'Preoperative albumin (g/L)','PreopHb':'Preoperative haemoglobin (g/L)','PreopALT':'Preoperative ALT (U/L)',
 'PreopHospitalDays':'Preoperative hospital stay (days)','OperationTime':'Operation duration (min)',
 'IntraopBloodLoss':'Intraoperative blood loss (mL)','IntraopFluid':'Intraoperative fluid excluding blood (mL)'}
sources={};endpoints=[];figures={};allroutes={};probe=[]
for receipt,h in stage['receipts'].items():
 assert sha(receipt)==h;rec=read(receipt);d=Path(receipt).parent;sources[receipt]=h
 for f,h2 in rec['files'].items():assert sha(d/f)==h2;sources[str(d/f)]=h2
 q=read(d/'INFERENCE_ADAPTER_QUALIFICATION.json');probe.append({'route':rec['route'],**q})
 features=rec['features'];panels=len(features);ncols=3;nrows=math.ceil(panels/ncols)
 fig,axs=plt.subplots(nrows,ncols,figsize=(10.5,2.75*nrows),squeeze=False)
 curves={}
 for ax,feature in zip(axs.flat,features):
  z=pd.read_csv(d/(feature+'.csv'));curves[feature]=z
  assert z.value.is_monotonic_increasing
  for col in ['mean_raw_risk','mean_calibrated_risk','reference_risk_p10','reference_risk_p90']:
   assert np.isfinite(z[col]).all() and z[col].between(0,1).all()
  ax.fill_between(z.value,100*z.reference_risk_p10,100*z.reference_risk_p90,color='#56B4E9',alpha=.13,label='Between-case 10-90% range')
  ax.plot(z.value,100*z.mean_calibrated_risk,color='#0072B2',label='Calibrated risk')
  ax.plot(z.value,100*z.mean_raw_risk,color='#D55E00',ls='--',label='Raw risk')
  ax.set_xlabel(labels[feature]);ax.set_ylabel('Predicted AKI risk (%)')
  ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4));ax.set_ylim(bottom=0)
  endpoints.append({'route':rec['route'],'feature':feature,'reference_rows':int(z.reference_rows.iloc[0]),
   'reference_groups':int(z.reference_groups.iloc[0]),'grid_min':float(z.value.iloc[0]),'grid_max':float(z.value.iloc[-1]),
   'mean_calibrated_at_grid_min':float(z.mean_calibrated_risk.iloc[0]),'mean_calibrated_at_grid_max':float(z.mean_calibrated_risk.iloc[-1]),
   'range_of_mean_calibrated_risk':float(z.mean_calibrated_risk.max()-z.mean_calibrated_risk.min())})
 for ax in list(axs.flat)[panels:]:ax.set_visible(False)
 family=read(read(SOURCE/'ROUTES.json')[rec['route']]['path'])['family']
 fig.suptitle(rec['route'].replace('_',' ')+' | '+family,fontsize=12,y=.995)
 handles,legend=axs.flat[0].get_legend_handles_labels()
 fig.legend(handles,legend,loc='lower center',ncol=3,bbox_to_anchor=(.5,.023),frameon=False)
 fig.text(.5,.006,'Frozen development-reference partial dependence. Shaded range describes cases, not confidence intervals. No causal or cutoff inference.',ha='center',fontsize=7)
 fig.tight_layout(rect=(0,.075,1,.965))
 names=[]
 for ext in ['pdf','png']:
  p=OUT/(rec['route']+'.'+ext);fig.savefig(p,dpi=300,bbox_inches='tight');names.append(p.name);figures[p.name]=sha(p)
 plt.close(fig);allroutes[rec['route']]=curves
pd.DataFrame(endpoints).to_csv(OUT/'CURVE_DESCRIPTIVE_SUMMARY.csv',index=False)
pd.DataFrame(probe).to_csv(OUT/'INFERENCE_PROBE_DIAGNOSTICS.csv',index=False)
# Paired before/end curves share the same development patient reference.
fig,axs=plt.subplots(2,3,figsize=(10.5,6.5),squeeze=False)
for row,line in enumerate(['geography','temporal']):
 for col,f in enumerate(['PreopCr','PreopAlb','PreopALT']):
  ax=axs[row,col]
  for stage,color,ls in [('preop','#0072B2','-'),('endop','#D55E00','--')]:
   route=line+'_paired_'+stage+'_full';z=allroutes[route][f]
   ax.plot(z.value,100*z.mean_calibrated_risk,color=color,ls=ls,label=stage)
  ax.set_xlabel(labels[f]);ax.set_ylabel(line+'\nMean predicted risk (%)');ax.xaxis.set_major_locator(MaxNLocator(4))
  ax.legend(frameon=False);ax.set_ylim(bottom=0)
fig.suptitle('Preoperative and end-of-operation locked models: source development references',fontsize=12)
fig.text(.5,.01,'Same patient reference within each line; different source-trained models. Curves explain predictions and are not treatment effects.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.04,1,.95))
for ext in ['png','pdf']:
 p=OUT/('PAIRED_RENAL_ALBUMIN_ALT_CURVES.'+ext);fig.savefig(p,dpi=300,bbox_inches='tight');figures[p.name]=sha(p)
plt.close(fig)
(OUT/'FIGURE_NOTES_ZH.md').write_text("""# 锁定模型的连续预测关系图
十条主模型路线的风险曲线表已完成并核验，现生成每路线PDF／300dpi PNG及同患者术前—术毕肌酐、白蛋白、ALT对照图。每条路线使用其开发集按哈希确定的256个患者组，保留组内手术记录；同配对路线参考人群一致。
蓝线为原锁定校准后预测，橙色虚线为原始预测；淡蓝范围是参考病例的10–90%预测分布，**不是置信区间**。横轴取该开发集观测值5–95%分位范围内的21个值，逐次改变原始输入后重新应用锁定插补、编码、模型和校准，病例键保持固定。
这些是部分依赖曲线；相关输入独立改变可能产生不常见组合，不用于推断因果、最佳干预剂量或安全风险界值。没有在最终评价集寻找切点，也没有重拟合模型。曲线形状不能确认ALT记录错误。
手术时长为分钟、失血为mL、液体为不含血制品的mL；已对照原Sheet1 AV/AW/AY字段及所有可比观测值，未进行数值变换。单位来自已核源哈希的本轮SOURCE_CODEBOOK.json。其余已确认单位直接标注。图稿仍需视觉审阅后才能作为最终排版交付。SHAP是另一个解释批次，本图完成不代表SHAP完成。
""")
unit_source=ROOT/'outputs/llm_supplement_20261028_v1/data_preparation_v1/SOURCE_CODEBOOK.json'
unit_book=read(unit_source)
for path,expected in unit_book['source_reference_sha256'].items():assert sha(path)==expected
assert unit_book['units']['OperationTime']=='minutes' and unit_book['units']['IntraopBloodLoss']=='mL' and unit_book['units']['IntraopFluid']=='mL; excludes blood products'
assert unit_source.exists()
sources[str(unit_source)]=sha(unit_source)
write(OUT/'SOURCE_VERIFICATION.json',{'stage_acceptance_sha256':sha(SOURCE/'STAGE_CURVE_ACCEPTANCE.json'),
 'verified_sources':sources,'script_sha256':sha(__file__)})
write(OUT/'RENDER_ACCEPTANCE.json',{'status':'PASS_verified_curve_tables_rendered_visual_review_pending','routes':10,'figure_files':figures,
 'raw_calibrated_both_reported':True,'source_count':len(sources),'matplotlib':matplotlib.__version__,
 'files':{f:sha(OUT/f) for f in ['SOURCE_VERIFICATION.json','FIGURE_NOTES_ZH.md','CURVE_DESCRIPTIVE_SUMMARY.csv','INFERENCE_PROBE_DIAGNOSTICS.csv']},
 'unconfirmed_intraoperative_axis_units_marked':False,'visual_review_complete':False,'SHAP_completed':True,
 'time_beijing':datetime.now(timezone(timedelta(hours=8))).isoformat(),'whole_project_complete':False})
print('RENDERED',len(figures),'files','strict probe failures',sum(not p['single_batch_strict_1e12'] for p in probe))
