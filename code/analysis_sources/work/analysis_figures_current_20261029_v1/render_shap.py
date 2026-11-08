"""Source-verified SHAP magnitude figures; no fitting or final-data use."""
from pathlib import Path
import hashlib,json,re
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps,ImageDraw
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
SRC=ROOT/'outputs/locked_pipeline_shap_20261024_v1';OUT=ROOT/'outputs/final_analysis_delivery_20261025_v1/figures/SHAP';OUT.mkdir(mode=0o700,parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
r=read(SRC/'REVIEW_ACCEPTANCE.json');assert r['status']=='PASS_ten_frozen_pipeline_SHAP_outputs_and_arithmetic_reverified'
for f,h in r['new_report_files'].items():assert sha(SRC/f)==h
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
labels={'Age':'Age','PreopCr':'Preoperative creatinine','PreopBUN':'Preoperative urea nitrogen','PreopAlb':'Preoperative albumin','PreopHb':'Preoperative haemoglobin','PreopALT':'Preoperative ALT','PreopHospitalDays':'Preoperative hospital stay','PreopGlucose':'Preoperative glucose','OperationTime':'Operation duration','IntraopBloodLoss':'Intraoperative blood loss','IntraopFluid':'Intraoperative fluid','BMI':'BMI','ASA':'ASA class','OtherMalignancy':'Other malignancy','NeoadjuvantChemo':'Neoadjuvant chemotherapy','HeartDisease':'Heart disease','TumorGroup':'Tumour group','PreviousAbdominalSurgery':'Previous abdominal surgery'}
def label(s):return labels.get(s,re.sub(r'(?<!^)(?=[A-Z][a-z])',' ',s))
sources={};figures={};tables={}
for receipt,h in r['source_receipts'].items():
 p=Path(receipt).parent
 assert sha(receipt)==h
 rec=read(receipt);assert sha(p/'FEATURE_IMPORTANCE.csv')==rec['files']['FEATURE_IMPORTANCE.csv']
 z=pd.read_csv(p/'FEATURE_IMPORTANCE.csv',float_precision='round_trip')
 names=z[z.scale=='calibrated_probability'].sort_values(['mean_absolute_SHAP','feature'],ascending=[False,True]).head(10).feature.tolist()[::-1]
 fig,axes=plt.subplots(1,2,figsize=(11,4.9),sharey=True)
 for ax,scale,title,color in zip(axes,['raw_probability','calibrated_probability'],['Raw probability','Calibrated probability'],['#D55E00','#0072B2']):
  q=z[z.scale==scale].set_index('feature').loc[names]
  vals=q.mean_absolute_SHAP.to_numpy()*100
  ax.barh(np.arange(len(names)),vals,color=color,height=.7)
  ax.set_yticks(np.arange(len(names)),[label(f) for f in names])
  ax.set_title(title);ax.set_xlabel('Mean absolute SHAP (percentage points)')
  ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
 fig.suptitle(p.name.replace('_',' '),fontsize=12)
 fig.text(.5,.015,'Development reference: 128 patient groups. Approximate whole-pipeline explanations; not causal effects. MC error is reported separately.',ha='center',fontsize=7)
 fig.tight_layout(rect=(0,.05,1,.94))
 for ext in ['png','pdf']:
  dest=OUT/(p.name+'.'+ext);fig.savefig(dest,dpi=300,bbox_inches='tight');figures[dest.name]=sha(dest)
 plt.close(fig)
 sources[str(p/'FEATURE_IMPORTANCE.csv')]=sha(p/'FEATURE_IMPORTANCE.csv');tables[p.name]=z
fig,axes=plt.subplots(2,2,figsize=(11.5,8))
for row,line in enumerate(['geography','temporal']):
 for col,stage in enumerate(['preop','endop']):
  name=line+'_paired_'+stage+'_full';z=tables[name]
  q=z[z.scale=='calibrated_probability'].sort_values(['mean_absolute_SHAP','feature'],ascending=[False,True]).head(7).iloc[::-1]
  axes[row,col].barh(np.arange(len(q)),100*q.mean_absolute_SHAP,color=['#0072B2','#D55E00'][col])
  axes[row,col].set_yticks(np.arange(len(q)),[label(f) for f in q.feature])
  axes[row,col].set_title(line.title()+' | '+('Preoperative' if stage=='preop' else 'End of operation'))
  axes[row,col].set_xlabel('Mean absolute SHAP (percentage points)');axes[row,col].grid(axis='x',alpha=.15);axes[row,col].set_axisbelow(True)
fig.suptitle('Information used by paired locked models: calibrated predictions',fontsize=12)
fig.text(.5,.012,'Source development references are shared within each line. Magnitudes also depend on the model and calibration; compare rankings descriptively.',ha='center',fontsize=7.5)
fig.tight_layout(rect=(0,.045,1,.95))
for ext in ['png','pdf']:
 p=OUT/('PAIRED_INFORMATION_CONTRIBUTIONS.'+ext);fig.savefig(p,dpi=300,bbox_inches='tight');figures[p.name]=sha(p)
plt.close(fig)
pngs=[OUT/(k+'.png') for k in sorted(tables)]
thumbs=[]
for p in pngs:
 im=Image.open(p).convert('RGB');im.thumbnail((1400,680))
 canvas=Image.new('RGB',(1420,720),'white');canvas.paste(im,((1420-im.width)//2,20));thumbs.append(canvas);im.close()
sheet=Image.new('RGB',(2840,720*5),'#dddddd')
for i,im in enumerate(thumbs):sheet.paste(im,((i%2)*1420,(i//2)*720));im.close()
sheet.save(OUT/'ALL_TEN_ROUTE_CONTACT_SHEET.png');sheet.close()
(OUT/'FIGURE_NOTES_ZH.md').write_text("""# 十路线SHAP图稿
十张每路线原始/校准概率贡献对照及一张配对主图。条长为平均绝对贡献，乘100换算百分点；每路线两尺度采用其校准尺度前十字段以便同字段对照，完整全部字段保留于源CSV。没有用变量重要性再筛选模型。
未绘制贡献或重要性置信区间；逐记录排列MC误差见数值表，不能把平均逐例MCSE当平均绝对贡献的标准误。解释基于开发参考128组、16背景组、64条反向配对路径，包含缺失和病例键的完整冻结预测流程。不是因果效应或干预阈值。
此为审阅图，仍需实际视觉审阅；接触表和主图不提供患者身份。原始逐例解释保持私有。
""")
save={'status':'PASS_verified_SHAP_aggregate_figures_rendered_visual_review_pending','source_review_sha256':sha(SRC/'REVIEW_ACCEPTANCE.json'),
 'source_importance_files':sources,'figures':figures,'contact_sheet_sha256':sha(OUT/'ALL_TEN_ROUTE_CONTACT_SHEET.png'),
 'script_sha256':sha(__file__),'visual_review_complete':False,'models_refitted':False}
(OUT/'RENDER_ACCEPTANCE.json').write_text(json.dumps(save,indent=2)+'\n')
print('SHAP_RENDERED',len(figures))
