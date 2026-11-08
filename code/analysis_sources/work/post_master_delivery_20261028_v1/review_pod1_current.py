"""Read-only verification and interpretation of corrected fixed-choice POD1 supplement."""
from pathlib import Path
import json,sys,hashlib,fcntl,datetime,gc
import numpy as np,pandas as pd,joblib
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
FIT=ROOT/'outputs/pod1_corrected_fixed_update_20261024_v1';EV=ROOT/'outputs/pod1_corrected_evaluation_20261024_v1'
sys.path.insert(0,str(ROOT/'work/pod1_corrected_fixed_update_20261024_v1'));import engine as e
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
seen={};files=0
def verify(p,expected=None):
 global files
 p=Path(p)
 assert p.resolve().is_relative_to((ROOT/'outputs').resolve()),('Result receipt outside corrected-run outputs',str(p))
 if expected:assert sha(p)==expected,(p,'receipt hash')
 if str(p) in seen:return
 r=read(p);seen[str(p)]=sha(p)
 for name,h in r.get('files',{}).items():assert sha(p.parent/name)==h,(p,name);files+=1
 sig=r.get('signature',{})
 if 'train' in sig and 'valid' in sig:
  tr=sig['train'];va=sig['valid']
  assert set(e.DATA.loc[tr,'group_hash']).isdisjoint(e.DATA.loc[va,'group_hash'])
 for field in ['upstream_receipts','raw_fit_receipts']:
  for f,h in r.get(field,{}).items():verify(f,h)
def exited(out):
 o=read(out/'LAUNCH.json');p=Path('/proc',str(o['pid']),'stat')
 assert not (p.exists() and p.read_text().split()[21]==str(o['start_ticks']))
 with (out/'SUPERVISOR.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 return o
def main():
 source_hash=sha(ROOT/'outputs/v4_development_inputs_20261018/SOURCE_LOCK.json')
 assert source_hash=='afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3'
 e.check();e.init()
 eligible=[len(e.DATA),int(e.DATA.y.sum())];assert eligible==[2947,39]
 fitowner=exited(FIT);evalowner=exited(EV)
 for f,h in read(EV/'CODE_LOCK.json').items():assert sha(ROOT/'work/pod1_corrected_evaluation_20261024_v1'/f)==h
 for f,h in read(EV/'DEPENDENCIES.json').items():assert sha(f)==h
 fa=read(FIT/'STAGE_POD1_FIT_ACCEPTANCE.json');assert fa['receipts'] and fa['final_models']>0
 parity=[]
 for f,h in fa['receipts'].items():
  verify(f,h);r=read(f);d=Path(f).parent
  z=pd.read_csv(d/'PRIVATE_PREDICTIONS.csv',float_precision='round_trip')
  cal=joblib.load(d/'calibrator.joblib')
  p=cal.predict_proba(e.logit(z.prob_raw.to_numpy()))[:,1]
  np.testing.assert_allclose(p,z.prob_calibrated.to_numpy(),rtol=0,atol=1e-12)
  assert np.isfinite(p).all()
  if r['signature']['task']['role']=='final':
   bundle=r['diagnostic']['bundle_path']
   raw=e.predict_bundle(bundle,z.excel_row.astype(int).tolist())
   np.testing.assert_allclose(raw,z.prob_raw.to_numpy(),rtol=1e-12,atol=1e-12)
   parity.append({'model':r['signature']['task']['line']+'/'+r['signature']['task']['branch']+'/'+r['signature']['task']['family'],
    'replay_max_abs':float(np.max(abs(raw-z.prob_raw.to_numpy()))),**r['diagnostic']})
  gc.collect()
 assert len(parity)==fa['final_models']
 pd.DataFrame(parity).to_csv(EV/'MODEL_REPLAY_REVIEW.csv',index=False)
 ea=read(EV/'STAGE_POD1_EVALUATION_ACCEPTANCE.json');assert len(ea['receipts'])==4
 pts=[];diff=[]
 for f,h in ea['receipts'].items():
  verify(f,h);dest=Path(f).parent
  r=read(dest/'EVALUATION.json');assert r['receipt']['replicates']==2000 and r['receipt']['seed']==20261017
  a=pd.read_csv(dest/'POINT_ESTIMATES.csv',float_precision='round_trip');b=pd.read_csv(dest/'PAIRED_DIFFERENCES.csv',float_precision='round_trip')
  assert a.n.nunique()==1 and a.events.nunique()==1
  a.insert(0,'context',dest.name);b.insert(0,'context',dest.name);pts.append(a);diff.append(b)
 pts=pd.concat(pts,ignore_index=True);diff=pd.concat(diff,ignore_index=True)
 pts.to_csv(EV/'ALL_POINT_ESTIMATES.csv',index=False)
 primary=pts[pts.model.str.endswith('__LR')].copy()
 primary.to_csv(EV/'PRESPECIFIED_LR_POINTS.csv',index=False)
 dd=diff[diff.metric.isin(['AUROC','AP','Brier','capacity10_captured_per1000','capacity20_captured_per1000'])].copy()
 dd.to_csv(EV/'KEY_PAIRED_DIFFERENCES.csv',index=False)
 tab=primary[['context','model','n','events','AUROC','AP','Brier','observed_expected','capacity20_captured_n']]
 lines=['# 当前纠正数据POD1结果：计算核验与边界记录','',
  '本文件是分析质量核验记录，不是稿件正文或最终科学审查。所有表和指标均读取本轮保存预测与原患者组重采样结果；不导入纠正前的数值或结论。','',
  '当前源锁：'+source_hash,
  '资格人群：'+str(eligible[0])+'条／'+str(eligible[1])+'个条件性后续采样肌酐事件。',
  '本轮已完成固定选择拟合情境：'+str(len(fa['receipts']))+'；最终模型包：'+str(fa['final_models'])+'；独立重放：'+str(len(parity))+'；评价分区：'+str(len(ea['receipts']))+'。',
  '全部保存校准概率复算、递归回执／患者边界／哈希核验和最终模型实际批次重放须在本程序中通过，才创建验收。没有重新拟合模型。','',
  '## 本轮结果入口','',
  '预先指定LR的各分区结果见PRESPECIFIED_LR_POINTS.csv；关键配对差值及原区间见KEY_PAIRED_DIFFERENCES.csv；全部候选见ALL_POINT_ESTIMATES.csv。',
  '下表仅列当前点估计；区间和配对差值不能由点估计推断：','',
  tab.to_csv(index=False),
  '## 保留的研究边界','',
  'POD1资格条件与后续采样肌酐终点按当前源合同保持；日类别采血记录不能补造成精确24小时预测时点或首次临床发病时刻。它是条件性后续肌酐风险更新。',
  '不能据POD1肌酐资格推定所有人当时均未满足尿量或RRT定义的AKI；主临床AKI终点与该条件性终点不同，不跨终点直接排列AUC。',
  '分区、预先指定LR和支持家族、固定参数与原患者种子保持；本次结果已暴露，内部评价条件于既有选择，不能称新盲验证，也不覆盖完整重新选参的不确定性。',
  '排序、总体概率误差、校准及相同容量下检出收益应分别依据当前保存配对区间判断；不根据已见评价改主模型或阈值。保留全部不利或不一致结果。',
  '任何病例、患者组、事件或分期均不在本程序内修改。最终科学结论仍须在全部37项分析齐备后的研究过程严审中决定。']
 (EV/'POD1_CURRENT_SOURCE_QC_ZH.md').write_text('\n'.join(lines)+'\n')
 reportfiles=['MODEL_REPLAY_REVIEW.csv','ALL_POINT_ESTIMATES.csv','PRESPECIFIED_LR_POINTS.csv','KEY_PAIRED_DIFFERENCES.csv','POD1_CURRENT_SOURCE_QC_ZH.md']
 save(EV/'REVIEW_ACCEPTANCE.json',{'status':'PASS_current_source_POD1_development_models_and_evaluations_reverified',
  'review_source_sha256':sha(__file__),'verified_receipts':len(seen),'verified_files':files,
  'final_model_full_batch_replays':len(parity),'eligible_cohort':eligible,'current_source_lock_sha256':source_hash,'actual_fixed_fit_contexts':len(fa['receipts']),'actual_final_models':fa['final_models'],'actual_evaluation_contexts':len(ea['receipts']),'scientific_review_performed':False,'manuscript_review_performed':False,'maximum_replay_absolute_error':max(x['replay_max_abs'] for x in parity),
  'raw_single_batch_diagnostics_retained':True,'source_owners_exited':[fitowner,evalowner],
  'all_case_parameters_previously_fixed_not_full_selection_validation':True,'no_primary_model_replacement':True,
  'report_files':{f:sha(EV/f) for f in reportfiles},'whole_project_complete':False,
  'time_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()})
 save(ROOT/'outputs/tonight_delivery_20261024_v1/POD1_ACCEPTANCE.json',{'status':'COMPLETE','acceptance':str(EV/'REVIEW_ACCEPTANCE.json'),'sha256':sha(EV/'REVIEW_ACCEPTANCE.json')})
 print('POD1_REVIEW_ACCEPTED',len(seen),'receipts',files,'files',flush=True)
if __name__=='__main__':main()
