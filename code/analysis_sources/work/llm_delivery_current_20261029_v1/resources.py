"""Read sealed task resource receipts once all formal work is complete; never infer efficiency from parameter count."""
import argparse, collections, datetime, math
from pathlib import Path
import common as c
def run():
 c.check_code();a=c.formal_guard()
 inputs={str(c.F/"TRAINING_AND_PREDICTION_ACCEPTANCE.json"):c.sha(c.F/"TRAINING_AND_PREDICTION_ACCEPTANCE.json")}
 with c.owned_lock("resources"):
  if c.existing_receipt("RESOURCE_REPORT_ACCEPTANCE.json",inputs):return
  rows=[];refs={}
  for task in a["tasks"]:
   path=Path(task["receipt"]);d=path.parent;rec=c.read(path);refs[str(path)]=c.sha(path)
   admission=c.read(d/"RESOURCE_ADMISSION.json")
   fields=c.read(d/"SIGNATURE.json")["fields"]
   assert rec["n_training_records"]==len(fields["train_record_keys"])
   assert rec["n_prediction_records"]==len(fields["prediction_record_keys"])
   row={k:rec[k] for k in ("model","stage","line","fit","kind","n_training_records","n_prediction_records",
         "peak_GPU_allocated_bytes","peak_GPU_reserved_bytes","peak_host_RSS_bytes","elapsed_seconds")}
   row["start_of_recorded_worker"]=admission["time"];row["completion_of_recorded_worker"]=rec["time"]
   assert math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"]>0
   assert 0<row["peak_GPU_allocated_bytes"]<=row["peak_GPU_reserved_bytes"]<=12*1024**3
   assert 0<row["peak_host_RSS_bytes"]<=16*1024**3
   row.update(training_seconds_this_process=None,training_updates=0,training_record_exposures=0,
              kind_scope="engineering_operator_only" if rec["kind"]=="pilot" else "formal_supplement")
   if rec["kind"] in ("fit","pilot"):
    p=d/"TRAINING_ACCEPTANCE.json";tr=c.read(p);refs[str(p)]=c.sha(p)
    assert tr["signature"]==rec["signature"] and tr["training_passes"]==1
    assert not tr["engineering_adapter_loaded"] and tr["all_training_keys_are_original_development"]
    assert tr["completed_updates"]==math.ceil(row["n_training_records"]/4)
    assert tr["training_seconds_this_process"]>0 and tr["training_seconds_this_process"]<=row["elapsed_seconds"]
    row.update(training_seconds_this_process=tr["training_seconds_this_process"],
               training_updates=tr["completed_updates"],training_record_exposures=tr["n_training_records"])
   else:assert row["n_training_records"]==0
   rows.append(row)
  assert len(rows)==57
  summary=[]
  for model in c.MODELS:
   for kind,count in (("before",2),("fit",16),("pilot",1)):
    subset=[x for x in rows if x["model"]==model and x["kind"]==kind];assert len(subset)==count
    summary.append({"model":model,"kind":kind,"tasks":count,
     "summed_completed_worker_seconds":math.fsum(x["elapsed_seconds"] for x in subset),
     "summed_recorded_training_seconds":math.fsum(x["training_seconds_this_process"] or 0 for x in subset),
     "prediction_record_exposures":sum(x["n_prediction_records"] for x in subset),
     "training_record_exposures":sum(x["training_record_exposures"] for x in subset),
     "maximum_worker_host_RSS_bytes":max(x["peak_host_RSS_bytes"] for x in subset),
     "maximum_GPU_allocated_bytes":max(x["peak_GPU_allocated_bytes"] for x in subset),
     "maximum_GPU_reserved_bytes":max(x["peak_GPU_reserved_bytes"] for x in subset)})
  c.write_csv(c.O/"TASK_RESOURCE_OBSERVATIONS.csv",rows);c.write_csv(c.O/"MODEL_RESOURCE_SUMMARY.csv",summary)
  notes="""# 三模型有限任务资源记录（分析附件）
此表读取57个封存任务回执：6个微调前预测银行、48个正式拟合及3个独立工程pilot。工程pilot另列，不并入正式训练总数。每条耗时为该完成进程记录的加载、训练和预测总耗时；训练字段是training_seconds_this_process，重试、此前进程、等待和下载不据此补算。两者之差不能称纯推理时长，也不把任务耗时之和称整个研究的墙钟时间。

峰值RAM是进程ru_maxrss，包含被计入RSS的映射页；与cgroup匿名内存或物理独占容量不同。GPU allocated与reserved分别保留，报告各任务最大值而非将串行峰值相加。总训练/预测记录次数包含原OOF及最终拟合的重复暴露，不是新增患者或独立样本。

各模型保留已经实测并冻结的microbatch、累积和推理批量，P40的实际热状态与环境亦会影响耗时。因此这里只比较本机固定方案的实测计算代价，不能由此单独归因于医学预训练、参数量或架构，也不代表其它硬件的吞吐。
"""
  (c.O/"RESOURCE_SCOPE_ZH.md").write_text(notes)
  c.save(c.O/"RESOURCE_REPORT_ACCEPTANCE.json",{"status":"PASS_SEALED_FORMAL_AND_PILOT_RESOURCES_REPORTED_WITH_MEASUREMENT_SCOPE",
    "time":c.now(),"inputs_sha256":inputs,"code_lock_sha256":c.check_code(),"receipts":refs,
    "files":{str(c.O/n):c.sha(c.O/n) for n in ("TASK_RESOURCE_OBSERVATIONS.csv","MODEL_RESOURCE_SUMMARY.csv","RESOURCE_SCOPE_ZH.md")},
    "formal_fit_count":48,"before_bank_count":6,"pilot_count":3,"not_total_project_runtime":True,
    "not_architecture_or_domain_single_factor_efficiency":True,"final_scientific_review_complete":False})
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--run",action="store_true");a=p.parse_args()
 if not a.run:raise SystemExit("Explicit --run after all formal tasks complete")
 run()
