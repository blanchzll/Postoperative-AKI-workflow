"""Measured finite cold/warm performance, same API IO and actual Chrome qualification."""
from pathlib import Path
import os,sys,json,hashlib,datetime,subprocess,fcntl,multiprocessing as mp,importlib.metadata as md
R=Path(__file__).resolve().parents[2];W=R/"work/calculator_current_20261029_v1";O=R/"outputs/calculator_release_20261025_v1";A=O/"app"
sys.path.insert(0,str(W))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 t=Path(str(p)+".tmp");t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n");os.replace(t,p)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def main():
 import qualify_routes_current as route
 with (O/"IO_PERFORMANCE_OWNER.lock").open("a") as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);sig=route.expected_signature()
  accepted=read(O/"ROUTE_ACCEPTANCE.json");assert accepted["passed"]==10 and accepted["current_qualification_signature"]==sig
  if not (O/"PERFORMANCE_ACCEPTANCE.json").exists():
   import benchmark_calculator_current_explicit_root as bench
   import pandas as pd
   usable,admission=route.resources()
   peak=max(read(p)["peak_rss_bytes"] for p in accepted["route_receipts"])
   perworker=max(int(1.5*1024**3),int(3*peak))
   assert usable>perworker+1024**3
   workers=max(1,min(10,len(os.sched_getaffinity(0)),int((usable-1024**3)//perworker)))
   save(O/"PERFORMANCE_RESOURCE_BUDGET.json",{"time":now(),"workers":workers,"per_parent_plus_cold_CLI_peak_budget_bytes":perworker,"real_resource_admission":admission,"filesystem_cache_dropped":False})
   save(O/"CURRENT_IO_PERFORMANCE_STATUS.json",{"status":"RUNNING_CURRENT_COLD_WARM_SYNTHETIC_BENCHMARKS","time":now(),"workers":workers})
   with mp.get_context("fork").Pool(workers) as pool:
    rows=[v for block in pool.imap_unordered(bench.run,list(read(A/"catalog.json"))) for v in block]
   assert len(rows)==30 and all(v["cold_warm_prediction_equal"] for v in rows)
   pd.DataFrame(rows).sort_values(["route","n"]).to_csv(O/"PERFORMANCE_COLD_WARM.csv",index=False)
   save(O/"PERFORMANCE_ACCEPTANCE.json",{"status":"PASS_all_ten_routes_cold_and_warm_1_100_1000","time":now(),"measurements":30,"exact_prediction_equal":True,"inputs_synthetic_only":True,"filesystem_cache_not_dropped":True,"current_qualification_signature":sig,"table_sha256":sha(O/"PERFORMANCE_COLD_WARM.csv")})
  if not (O/"IO_ACCEPTANCE.json").exists():
   env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
   with (O/"CURRENT_IO_BROWSER.log").open("ab") as log:
    child=subprocess.Popen([sys.executable,"-B",str(W/"qualify_io.py")],env=env,stdout=log,stderr=log)
    save(O/"CURRENT_IO_PERFORMANCE_STATUS.json",{"status":"RUNNING_INPUT_HTTP_CLI_AND_ACTUAL_CHROME_INTERACTION","time":now(),"worker_PID":child.pid})
    try:rc=child.wait(timeout=1200)
    except subprocess.TimeoutExpired:child.kill();child.wait();raise TimeoutError("Owned finite calculator IO/browser qualification budget exceeded")
   assert rc==0,"Inspect actual browser/IO receipts and logs before retry"
  assert read(O/"IO_ACCEPTANCE.json")["status"]=="PASS_validation_HTTP_CLI_and_actual_browser"
  assert read(O/"BROWSER_ACCEPTANCE.json")["status"]=="PASS_actual_browser_interaction_and_screenshots"
  assert read(O/"HTTP_SERVER_EXIT.json")["owned_server_exited"] and read(O/"HTTP_SERVER_EXIT.json")["normal_shutdown"]
  route.expected_signature()
  packages=["numpy","pandas","scipy","scikit-learn","joblib","threadpoolctl","xgboost","matplotlib","Pillow","openpyxl","statsmodels"]
  env={"python":sys.version,"executable":sys.executable,"resolved_executable":str(Path(sys.executable).resolve()),"packages":{name:md.version(name) for name in packages},"existing_frozen_environment_reused":True,"no_package_installation_or_upgrade_performed":True}
  save(O/"ENVIRONMENT_SNAPSHOT.json",env);(O/"requirements-core.txt").write_text("\n".join(name+"=="+version for name,version in env["packages"].items())+"\n")
  (O/"README_ZH.md").write_text("# 当前源研究计算器\n\n仅使用当前3710/205和配对2428/138开发评价中已冻结的十条中国主路线。各路线输入、原模型/插补/源校准/训练阈值来自新catalog和MODEL_LOCK，未部署事后目标域更新、common6、POD1、选择性策略或LLM。临床风险研究工具的工程验收不等于真实部署验证或干预疗效。\n\n使用原解释器运行app/cli.py --project-root R --input synthetic_examples/某路线.json，或app/web.py --project-root R --port 8765 --duration 3600；R指本次纠正分析根目录。网页仅监听服务器127.0.0.1，限定时长后退出，必要时自行通过SSH本地转发访问。验收服务器已退出；没有常驻或公网发布。\n\n所有列显式提供、真正缺失用null，单位和分类代码遵循catalog，未知编码不猜测语义。匿名case_key不得用真实姓名/住院号；键不进入预测变量，仅参与原随机插补复现，保持同一就诊稳定。普通缺失不作异常数据错误；有符号WeightLoss允许负值代表增加。\n\n共同engine支持CLI/HTTP，固定100行数值块在源插补之后执行；这是保持源预测与单例/重排一致性的工程实现，没有重新训练或放宽1e-12标准。全部评价数据实际回放、开发100/128、单例/批次/重排/新实例、完整数据键独立及全缺失稳定性逐路线验收。真实回放数量和误差见ROUTE_ACCEPTANCE；输入/HTTP/CLI/实际Chrome见IO、BROWSER和HTTP_SERVER_EXIT，冷进程/暖态1、100、1000合成病例计时见PERFORMANCE_COLD_WARM，不保证临床现场时延。\n\n模型状态包含训练插补信息，仅在私有项目/私有复现包使用。共享审阅材料和私有包另有独立门槛；本文件不授权公开发布。92个稿件文件保持保护，本次没有写改稿件。\n")
  artifacts=["ROUTE_ACCEPTANCE.json","IO_ACCEPTANCE.json","BROWSER_ACCEPTANCE.json","HTTP_SERVER_EXIT.json","PERFORMANCE_ACCEPTANCE.json","PERFORMANCE_COLD_WARM.csv","CURRENT_QUALIFICATION_CONTRACT.json","ENVIRONMENT_SNAPSHOT.json","requirements-core.txt","README_ZH.md","VARIABLE_DICTIONARY.csv"]
  save(O/"CALCULATOR_ACCEPTANCE.json",{"status":"PASS_complete_research_calculator_qualification","time":now(),"current_input_sha256":read(A/"VERSION.json")["current_input_sha256"],"current_qualification_signature":sig,"routes":10,"heldout_records_replayed":accepted["heldout_records_replayed"],"max_absolute_difference":accepted["max_absolute_difference"],"strict_absolute_tolerance":1e-12,"artifacts":{str(O/name):sha(O/name) for name in artifacts},"current_models_and_source_thresholds":True,"main_models_refitted":False,"target_local_recalibration_deployed":False,"actual_browser_qualification":True,"private_review_packages_complete":False,"final_scientific_review_complete":False,"whole_project_complete":False,"controller_code_sha256":sha(__file__)})
  save(O/"CURRENT_IO_PERFORMANCE_STATUS.json",{"status":"COMPLETE_CURRENT_CALCULATOR_ENGINEERING_ACTUAL_BROWSER_VISUAL_REVIEW_AND_PACKAGES_SEPARATE","time":now(),"owned_HTTP_Chrome_and_worker_processes_exited":True})
if __name__=="__main__":main()
