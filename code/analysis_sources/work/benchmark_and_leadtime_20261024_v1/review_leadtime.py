from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
OUT=ROOT/'outputs/benchmark_and_leadtime_20261024_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 fit=json.loads((OUT/'LEADTIME_FITTING_ACCEPTANCE.json').read_text())
 for f,h in fit['files'].items():assert sha(OUT/f)==h
 owner=json.loads((OUT/'LEADTIME_LAUNCH.json').read_text());p=Path('/proc')/str(owner['pid'])
 assert not(p.exists() and (p/'stat').read_text().split()[21]==owner['start_ticks'])
 points=pd.read_csv(OUT/'LEADTIME_POINT_ESTIMATES.csv').set_index('rule')
 intervals=pd.read_csv(OUT/'LEADTIME_INTERVALS.csv')
 boots=pd.read_csv(OUT/'LEADTIME_BOOTSTRAPS.csv')
 m=pd.read_csv(OUT/'PRIVATE_LEADTIME_JOIN.csv',dtype={'record_key':str})
 assert len(boots)==12000 and all(boots.groupby('rule').size()==2000)
 assert set(boots.replicate.unique())==set(range(2000))
 assert (boots['all_AKI_detected_atleast48h_n']<=boots['all_AKI_detected_atleast24h_n']).all()
 assert (boots['all_AKI_detected_atleast24h_n']<=boots['all_AKI_detected_atleast12h_n']).all()
 assert (boots['all_AKI_detected_atleast12h_n']<=boots.captured_n).all()
 assert (boots.captured_n+boots.missed_n==boots.events).all()
 assert (boots.captured_n+boots.false_alerts_n==boots.reviewed_n).all()
 checked=[]
 for line,threshold in fit['source_thresholds'].items():
  pred=pd.read_csv(ROOT/f'outputs/common6_public_evaluation_20261024_v1/{line}/PRIVATE_PREDICTIONS.csv',dtype={'record_key':str},float_precision='round_trip')
  z=m.merge(pred[['record_key','prob_calibrated']],on='record_key',validate='one_to_one')
  t=(z.first_observed_criterion_time-z.opstart_time)/60
  flags=z.prob_calibrated>=threshold;event=z.y_scr==1
  expected={'reviewed_n':int(flags.sum()),'captured_n':int((event&flags).sum()),
   'missed_n':int((event&~flags).sum()),'false_alerts_n':int((~event&flags).sum()),
   'detected_record_interval_hours_median':float(np.median(t[event&flags]))}
  for h in [12,24,48]:expected[f'all_AKI_detected_atleast{h}h_n']=int((event&flags&(t>=h)).sum())
  row=points.loc[line+'__source_sensitivity80']
  for k,v in expected.items():np.testing.assert_allclose(row[k],v,atol=1e-12,rtol=1e-12)
  checked.append({'line':line,'fixed_threshold':threshold,'independent_join_and_event_counts_match':True})
 # Independently reaggregate each interval from saved repeat values.
 for r in intervals.itertuples():
  q=boots.loc[boots.rule==r.rule,r.metric].dropna().to_numpy()
  assert len(q)==r.valid_replicates
  if len(q):
   qq=np.quantile(q,[.025,.975]);np.testing.assert_allclose(qq,[r.lower95,r.upper95],atol=1e-10,rtol=1e-10)
 report=['# INSPIRE固定术前时点至首次记录肌酐达标的间隔','','## 评价对象',
 '使用此前已锁定并评价的6043条INSPIRE首次合格胃/结直肠肿瘤编码手术，253例记录的肌酐终点；仅两条中国训练common6模型，未重新拟合或选择模型、未改变结局。',
 '规定术前预测时点为记录的手术开始；输入取该时点前的观测化验，普通个体缺失仍用原源训练插补状态。现有数据没有检验结果实际发布/医生看到结果的时间，因此这是固定时点到首次记录肌酐达标的回顾性间隔，是潜在行动窗口的支持性指标，不是已实测的临床系统预警提前量。',
 'INSPIRE使用共同的相对分钟时钟，记录最高5分钟聚合，实验室数值经过匿名化量化。首次记录达标时间不等于最早生物学损伤或完整临床AKI时间，且取决于采样安排。中国五中心不能用该数字代替自身提前时间。','',
 '## 冻结规则',
 '主要规则为各源模型锁定的training_sensitivity80风险阈值，实际公共队列灵敏度与预警率必须重新报告；名称不代表公共队列恰有80%灵敏度。支持规则为回顾性全批次最高10%/20%容量，不冒充逐例实时部署规则。相同中国主方案的医院内患者组2000次重采样、种子20261017用于区间，重采样权重哈希与既有公共评价完全一致。','',
 '|模型来源／规则|复核人数|检出/253|PPV|检出病例记录间隔中位数（Q1–Q3，小时）|≥12/24/48小时检出数（分母均253）|',
 '|---|---:|---:|---:|---|---|']
 for rule,z in points.iterrows():
  report.append(f"|{rule}|{int(z.reviewed_n)}|{int(z.captured_n)}|{z.PPV:.1%}|{z.detected_record_interval_hours_median:.1f}（{z.detected_record_interval_hours_Q1:.1f}–{z.detected_record_interval_hours_Q3:.1f}）|{int(z.all_AKI_detected_atleast12h_n)}/{int(z.all_AKI_detected_atleast24h_n)}/{int(z.all_AKI_detected_atleast48h_n)}|")
 report+=['','## 结果解读',
 '在回顾性20%复核容量下，两模型检出121/123例。检出病例的记录间隔中位数约41小时，但分母为全部253例时，至少提前24小时检出的比例仅79/253和82/253，至少48小时为36/253和37/253。不能把检出者中位数解释为所有AKI均可提前约41小时识别。',
 '源训练80%灵敏度阈值在公共队列分别触发2619和5024次复核，约43.3%和83.1%病例，PPV仅6.9%和4.8%。因此较高捕获不能脱离明显复核负担单独作为成功结论；这些阈值没有利用公共结局重新优化。',
 '本评价是单次术前、7日记录肌酐风险。它不同于Nature2019连续更新、48小时未来终点的研究；不能将其AUC、提前窗口或日报警率直接作同条件排名。未证明降低AKI或已可执行干预。','',
 '## 可复核输出',
 'LEADTIME_POINT_ESTIMATES.csv、LEADTIME_INTERVALS.csv保存全部点估计、2000次有效区间与分母；源阈值及确切模型身份在合同及原模型锁中。没有在评估病例拟合校准器或更改阈值。','',
 '## 原始来源',
 '- [INSPIRE v1.4.2 数据定义与时间轴](https://physionet.org/content/inspire/1.4.2/)。',
 '- [Tomašev等，Nature2019](https://www.nature.com/articles/s41586-019-1390-1)，作为提前时间与报警负担联合报告的方法背景。']
 (OUT/'LEADTIME_RESULTS_ZH.md').write_text('\n'.join(report)+'\n')
 receipt={'status':'PASS_fixed_landmark_record_time_interval_support_reviewed','events':253,'cohort':6043,'source_only_common6_models':2,
 'actual_clinical_alert_lead_time_validated':False,'Chinese_lead_time_imputed':False,'no_final_model_or_threshold_retuning':True,
 'resampling_proof':fit['public_resampling_proof_matches'],'independent_point_checks':checked,
 'intervals_reaggregated':len(intervals),'finite_owner_exited':True,'source_acceptance_sha256':sha(OUT/'LEADTIME_FITTING_ACCEPTANCE.json'),
 'review_code_sha256':sha(__file__),'report_sha256':sha(OUT/'LEADTIME_RESULTS_ZH.md')}
 save(OUT/'LEADTIME_REVIEW_ACCEPTANCE.json',receipt)
 print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
