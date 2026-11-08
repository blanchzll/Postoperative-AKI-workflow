"""Current-source independent SHAP arithmetic QC; no manuscript or numerical refitting."""
from pathlib import Path
import datetime,hashlib,json,fcntl,importlib.util,os
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2]
O=R/"outputs/locked_pipeline_shap_20261024_v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 t=Path(str(p)+".tmp");t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n");os.replace(t,p)
def run():
 assert sha(R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json")=="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3"
 with (O/"REVIEW_OWNER.lock").open("a") as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  source=R/"work/support_results_review_20261024_v1/review.py"
  spec=importlib.util.spec_from_file_location("native_qc",source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  ac,n=m.verify_stage(O,"STAGE_SHAP_ACCEPTANCE.json",R/"work/locked_pipeline_shap_20261024_v1")
  assert ac["status"]=="PASS_ten_locked_model_permutation_SHAP_routes" and len(ac["receipts"])==10
  owner=m.exited(O);summ=[];tops=[];sources={}
  for f,h in ac["receipts"].items():
   p=Path(f).parent
   with np.load(p/"PRIVATE_SHAP_VALUES.npz",allow_pickle=False) as z:
    v=z["values"];se=z["mc_se"];pred=z["prediction"];base=z["baseline"]
    assert v.shape==se.shape and v.shape[2]==2
    assert np.isfinite(v).all() and np.isfinite(se).all() and (se>=0).all()
    err=float(np.max(np.abs(base+v.sum(axis=1)-pred)));assert err<1e-12
    lock=read(p/"EXPLANATION_LOCK.json")
    assert np.array_equal(z["excel_row"],lock["reference_rows"])
    pairfiles=sorted(p.glob("background_*_draw_*.npz"));assert len(pairfiles)==32
    draws=[]
    for q in pairfiles:
     with np.load(q,allow_pickle=False) as zpair:draws.append(zpair["values"])
    vv=np.stack(draws).reshape(16,2,*v.shape)
    assert np.max(np.abs(vv.mean(axis=(0,1))-v))<1e-12
    independent_se=np.sqrt(np.var(vv,axis=1,ddof=1).sum(axis=0)/(2*16**2))
    assert np.max(np.abs(independent_se-se))<1e-12
    imp=pd.read_csv(p/"FEATURE_IMPORTANCE.csv",float_precision="round_trip")
    for i,feature in enumerate(z["features"]):
     for j,scale in enumerate(["raw_probability","calibrated_probability"]):
      hit=imp[(imp.feature==feature)&(imp.scale==scale)];assert len(hit)==1
      row=hit.iloc[0]
      assert abs(float(row.mean_absolute_SHAP)-float(np.mean(np.abs(v[:,i,j]))))<1e-14
      assert abs(float(row.mean_signed_SHAP)-float(np.mean(v[:,i,j])))<1e-14
      assert abs(float(row.mean_MC_SE)-float(np.mean(se[:,i,j])))<1e-14
    top=imp[imp.scale=="calibrated_probability"].sort_values(["mean_absolute_SHAP","feature"],ascending=[False,True]).head(6).copy()
    top.insert(0,"route",p.name);top.insert(1,"rank",range(1,len(top)+1));tops.append(top)
    diag=read(p/"EXPLANATION_DIAGNOSTICS.json");assert diag["original_adapter_agrees"]
    summ.append({"route":p.name,"records":v.shape[0],"raw_fields":v.shape[1],"max_additivity_error":err,"mean_calibrated_risk":float(pred[:,1].mean()),"mean_MC_SE":float(se[:,:,1].mean()),"background_groups":16,"independent_draws_per_background":2,"paths":64})
   sources[str(Path(f))]=h
  summary=pd.DataFrame(summ);top=pd.concat(tops,ignore_index=True)
  summary.to_csv(O/"ROUTE_REVIEW_SUMMARY.csv",index=False);top.to_csv(O/"CALIBRATED_TOP_SIX_FEATURES.csv",index=False)
  lines=["# 当前源十路线SHAP数值复核","",
  "重新核验十条路线的代码、依赖、回执及全部输出，并独立重算加和恒等式、两次排列抽样的MC标准误和字段聚合值。本报告不拟合模型，不审改稿件，也不代表最终统计与科学严审。","",
  m.mdtable(summary),"",m.mdtable(top[["route","rank","feature","mean_absolute_SHAP","mean_MC_SE"]]),"",
  "每路线按原合同抽取128个开发患者组，保留组内记录；聚合均值的分母为这些手术记录。背景16组、每背景2个独立排列及反向路径，共64条路径。原始和校准概率分别计算贡献。上述表格完全来自本轮文件；没有沿用旧数值或预写变量排序结论。","",
  "MC标准误只量化给定背景下有限排列的近似误差，不包括背景抽样、模型训练或临床风险不确定性。相关字段独立替换可能产生少见组合；贡献不是因果效应、用药适应证或安全界值。背景与参考病例均来自开发侧，未按评价表现选择。","",
  "图稿渲染与逐图审读另行验收；主假说与信息增量以同患者评价结果为依据。"]
  (O/"RESULTS_INTERPRETATION_ZH.md").write_text("\n".join(lines)+"\n")
  receipt={"status":"PASS_ten_frozen_pipeline_SHAP_outputs_and_arithmetic_reverified","verified_output_files":n,"route_count":10,"max_additivity_error":float(summary.max_additivity_error.max()),"code_and_dependencies_checked":True,"source_owner_exited":owner,"source_receipts":sources,"current_input_sha256":sha(R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json"),"current_QC_code_sha256":sha(__file__),"native_verifier_code_sha256":sha(source),"new_report_files":{str(O/f):sha(O/f) for f in ["ROUTE_REVIEW_SUMMARY.csv","CALIBRATED_TOP_SIX_FEATURES.csv","RESULTS_INTERPRETATION_ZH.md"]},"figures_reviewed":False,"whole_project_complete":False,"final_scientific_review_performed":False,"time_UTC":datetime.datetime.now(datetime.timezone.utc).isoformat()}
  save(O/"REVIEW_ACCEPTANCE.json",receipt)
  print(json.dumps({"verified_files":n,"routes":10,"maximum_additivity_error":receipt["max_additivity_error"]}))
if __name__=="__main__":run()
