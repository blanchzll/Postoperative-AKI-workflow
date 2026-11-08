from pathlib import Path
from decimal import Decimal,getcontext
import csv,json,hashlib,math
getcontext().prec=50
R=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
T=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(fn,rows):
 with (T/fn).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
c=json.loads((T/'ANALYSIS_CONTRACT.json').read_text())
assert c['analysis_type']=='posthoc_exploratory_CI_based_loss_margin_sensitivity'
for p,h in c['source_sha256'].items():assert sha(p)==h,('SOURCE_CHANGED',p)
for fn,h in c['code_sha256'].items():assert sha(T/fn)==h
grid=[Decimal(s) for s in c['margin_grid_capture_per1000']]
assert grid==[Decimal(i)/2 for i in range(21)]
sensitivity=[];tips=[];support=[];original=[];counts=[]
for context,full,compact in [('geography_preop','preop_full_champion_LR','preop_compact_champion_LR'),('temporal_preop','preop_full_champion_Voting','preop_compact_champion_Voting')]:
 d=R/'outputs/formal_evaluation_20261023_v1'/context
 points={r['model']:r for r in csv.DictReader((d/'POINT_ESTIMATES.csv').open())}
 pairs=[r for r in csv.DictReader((d/'PAIRED_DIFFERENCES.csv').open()) if r['model']==full+' MINUS '+compact]
 assert len(pairs)==39
 native={r['metric']:r for r in pairs}
 for r in pairs:original.append({'context':context,**r})
 f,p=points[full],points[compact];assert f['n']==p['n'] and f['events']==p['events']
 for quota in [20,10]:
  key=f'capacity{quota}';r=native[key+'_captured_per1000']
  assert r['valid_replicates']=='2000' and r['requested_replicates']=='2000'
  est,lo,hi=(Decimal(r[k]) for k in ['estimate','lower95','upper95'])
  for delta in grid:
   satisfies=hi<delta
   sensitivity.append({'context':context,'review_capacity_percent':quota,'role':'primary' if quota==20 else 'support',
    'n':f['n'],'events':f['events'],'delta_capture_per1000':str(delta),
    'loss_full27_minus_compact9':str(est),'lower95':str(lo),'upper95':str(hi),
    'one_sided_confidence_level':'0.975','loss_upper_bound':str(hi),
    'upper_minus_delta':str(hi-delta),'bound_strictly_below_margin':int(satisfies),
    'interpretation':'CI_compatible_with_this_margin' if satisfies else 'CI_does_not_rule_out_loss_at_least_this_margin',
    'clinical_margin_validated':False,'exploratory':True,'valid_replicates':r['valid_replicates'],'seed':20261017})
  passing=[g for g in grid if hi<g]
  tips.append({'context':context,'review_capacity_percent':quota,'role':'primary' if quota==20 else 'support',
   'loss_full27_minus_compact9':str(est),'lower95':str(lo),'upper95':str(hi),
   'continuous_boundary_delta':str(hi),'continuous_rule':'delta > boundary; equality does not satisfy',
   'first_grid_margin_satisfying_rule':str(min(passing)) if passing else '',
   'independently_clinically_justified_delta':'not_established'})
  counts.append({'context':context,'review_capacity_percent':quota,'n':f['n'],'events':f['events'],
   'full27_model':full,'compact9_model':compact,'reviewed_each':f[key+'_reviewed_n'],
   'full27_captured':f[key+'_captured_n'],'compact9_captured':p[key+'_captured_n'],
   'full27_missed':f[key+'_missed_n'],'compact9_missed':p[key+'_missed_n'],
   'full27_false_alerts':f[key+'_false_alerts_n'],'compact9_false_alerts':p[key+'_false_alerts_n'],
   'full27_PPV':f[key+'_PPV'],'compact9_PPV':p[key+'_PPV'],
   'full27_sensitivity':f[key+'_sensitivity'],'compact9_sensitivity':p[key+'_sensitivity'],
   'full27_capture_per1000':f[key+'_captured_per1000'],'compact9_capture_per1000':p[key+'_captured_per1000']})
 for metric in ['AUROC','AP','Brier']:
  r=native[metric];v,l,u=[Decimal(r[k]) for k in ['estimate','lower95','upper95']]
  if metric=='Brier':v,l,u=-v,-u,-l
  support.append({'context':context,'metric':metric,'full27_value':f[metric],'compact9_value':p[metric],
   'compact_loss_estimate':str(v),'compact_loss_lower95':str(l),'compact_loss_upper95':str(u),
   'loss_definition':'compact9_minus_full27' if metric=='Brier' else 'full27_minus_compact9',
   'valid_replicates':r['valid_replicates'],'requested_replicates':r['requested_replicates'],
   'margin_grid_applied':False,'reason':'Different metric scales; no cross-scale clinical margin assumed.'})
joint=[]
for quota in [20,10]:
 a=[t for t in tips if t['review_capacity_percent']==quota]
 boundary=max(Decimal(t['upper95']) for t in a)
 for g in grid:
  flags=[Decimal(t['upper95'])<g for t in a]
  joint.append({'review_capacity_percent':quota,'delta_capture_per1000':str(g),
   'geography_bound_below_margin':int(flags[0]),'temporal_bound_below_margin':int(flags[1]),
   'both_context_bounds_below_margin':int(all(flags)),
   'both_contexts_continuous_boundary':str(boundary),
   'interpretation':'Logical conjunction of two contextual CI criteria; no pooled effect or independence assumed.',
   'clinical_margin_validated':False})
dump('MARGIN_SENSITIVITY.csv',sensitivity);dump('TIPPING_POINT_SUMMARY.csv',tips)
dump('CROSS_CONTEXT_SENSITIVITY.csv',joint);dump('PERFORMANCE_SUPPORT.csv',support)
dump('CLINICAL_COUNTS_AT_CAPACITY.csv',counts);dump('ALL_ORIGINAL_PAIRED_METRICS.csv',original)
primary=[t for t in tips if t['review_capacity_percent']==20]
table='|验证线|九输入捕获损失/1000|原95%区间|完整0–10网格中首个满足判据的δ|\\n|---|---:|---|---:|\\n'
for t in primary:table+=f"|{'地理' if t['context'].startswith('geo') else '时间'}|{Decimal(t['loss_full27_minus_compact9']):.3f}|[{Decimal(t['lower95']):.3f}, {Decimal(t['upper95']):.3f}]|{t['first_grid_margin_satisfying_rule']}|\\n"
table=table.replace('\\n','\n')
text=f"""# 九输入相对27输入：性能损失界值敏感性分析 v2
已根据用户2026-10-31的澄清，实际完成本问题的独立探索性分析附件。研究对象是预测性能取舍，不需要白蛋白给药或AKI发病时间，也未引入因果森林/Double ML。v1的历史判断和原始接受材料保留。

## 分析合同与判据
主指标L=27输入模型捕获事件/1000减9输入模型捕获事件/1000，L>0代表九输入损失。主情景为20%复核容量，10%为支持情景。δ完整网格固定0至10、步长0.5，共21档；这是每千名评价手术少捕获事件的容忍量，10只是展示上限，任何一档均未被认定临床合理或安全。
合同在生成本轮表图前锁定；原点估计和区间早已暴露，因此**不是评价前预注册或确认性非劣试验**。原医院内患者组bootstrap2000次、seed20261017、原模型/校准/样本/容量取整和完整区间均保留。每条线39项原配对指标全部另存，其中38项数值、upgraded_per1000不适用且不填0。
原CI由2.5%/97.5%分位数构成。取其上端U作为相容的单侧97.5%上界；**U<δ**记为“该区间与此容忍界值相容”。U=δ不满足。本单向损失对应的非劣假设为H0:L≥δ；此处只作事后区间反演判读，不生成新P值，不将bootstrap抽样频率称为非劣概率或后验概率。

## 主结果
{table}
这给出明确的界值敏感性：δ=5时，时间线满足U<δ，地理线不满足；δ=7时，两线都满足。这并不表示5或7成为临床推荐界值，也不表示九输入已获临床非劣认证。
连续界值边界分别为地理δ>{primary[0]['upper95']}、时间δ>{primary[1]['upper95']}。两线同时满足逐线判据要求δ>{max(Decimal(t['upper95']) for t in primary)}，0.5步长展示网格的首档是7。连续边界来自区间端点，不能反过来作为选择临床δ的理由。
地理主术前评价为1159手术/72事件，复核232例时全27输入捕获34例、九输入32例；时间评价为1683/66，复核337例时分别捕获38例、35例。两模型在每一情景使用相同评价人口、相同复核人数。这是full与compact的配对预测比较，不是术前/术毕配对子人群，也不是临床治疗分配。

## 支持结果与完整披露
10%容量：地理损失上界为6.866952789699571、时间为2.9691211401425193，网格首个满足档分别7、3；同样是条件相容结论。所有84条逐线×容量×δ和42条跨线逻辑联合结果完整保存。
AUROC、AP和Brier的两线6条原差值区间完整保留。AUROC/AP按full27−compact9，Brier按compact9−full27并反转区间端点，全部统一为正值表示九输入损失。它们有不同单位，不把捕获/1000的δ套在AUC/AP/Brier上，也不据指标方向重选主赢家。
两线分别输出，共同相容只是逐线判据的逻辑交集；没有输出合并效应、独立重复验证或跨人群普遍非劣主张。原评价暴露及固定模型条件推断限制继续适用。界值网格是同一估计与区间的完整敏感性解释，不是21个独立实验，也不提供临床容忍量本身的新证据。

## 可以解决及仍需临床判断的部分
本分析已解决：给定任何展示δ时判断如何、跨验证线是否一致、判读改变的边界在哪里。临床需要决定的仍是可容忍漏捕获量及其与信息采集负担的取舍；27变9少18字段并不等于实测工时或费用，捕获事件也不等于治疗获益或避免AKI。
原结论矩阵C02所撤回的“27输入普遍确定优于9输入”保持撤回；本附件只新增可追溯的条件性结果，不把它改写为九输入普遍等效/非劣。未查看或修改论文正文及原92保护文件，正式写作整合由原独立写作对话负责。

## 复核
分析脚本、合同、源CSV和原接受文件均记录SHA256；另一个Python程序独立重建网格、阈值判断、转向区间及捕获整数算术。图根据CSV实际渲染后进行像素审读。最终接受仅针对本次探索性敏感性附件；模型训练、新bootstrap、因果估计均为0。
"""
(T/'RESULTS_INTERPRETATION_ZH.md').write_text(text)
(T/'CONCLUSION_DISPOSITION.json').write_text(json.dumps({'prior_C02_disposition':'withdrawn; unchanged','new_conditional_claims':['At delta5, temporal20 meets original upper-bound criterion and geography20 does not.','At delta7, both original contextual20 intervals meet the criterion.'],'no_independent_clinical_margin_established':True,'no_global_noninferiority_or_equivalence_claim':True,'models_or_core_review_materials_changed':False},ensure_ascii=False,indent=2))
print(json.dumps({'sensitivity_rows':len(sensitivity),'cross_context_rows':len(joint),'original_paired_rows':len(original),'support_rows':len(support),'clinical_count_rows':len(counts),'tipping_points':tips},ensure_ascii=False),flush=True)
