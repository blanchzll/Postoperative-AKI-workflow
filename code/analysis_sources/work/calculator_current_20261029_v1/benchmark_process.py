from pathlib import Path
import os,sys,json,time,copy,subprocess,multiprocessing as mp
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2];O=R/"outputs/calculator_release_20261025_v1";A=O/"app";sys.path.insert(0,str(A))
from engine import Calculator
def run(route):
 base=json.loads((O/"synthetic_examples"/(route+".json")).read_text());calc=Calculator(R);rows=[]
 for n in [1,100,1000]:
  q=copy.deepcopy(base);q["records"]=[]
  for i in range(n):
   rec=dict(base["records"][0],case_key=f"DEMO_perf_{i:04d}")
   rec["Age"]=float(45+i%40)
   rec["PreopCr"]=None if i%7==0 else float(60+5*(i%20))
   rec["PreopAlb"]=None if i%11==0 else float(32+i%12)
   q["records"].append(rec)
  p=O/"synthetic_performance"/f"{route}_{n}.json";p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(q,ensure_ascii=False)+"\n")
  start=time.monotonic();cold=subprocess.run([sys.executable,str(A/"cli.py"),"--input",str(p)],capture_output=True,text=True,timeout=120);coldsec=time.monotonic()-start;assert cold.returncode==0,cold.stderr
  expected=json.loads(cold.stdout)
  calc.predict(q)
  start=time.monotonic();warm=calc.predict(q);warmsec=time.monotonic()-start
  assert warm==expected,(route,n)
  rows.append({"route":route,"n":n,"input":"constructed synthetic with controlled missingness","cold_process_seconds":coldsec,"warm_resident_seconds":warmsec,"cold_includes_interpreter_and_imports":True,"filesystem_cache_not_dropped":True,"cold_warm_prediction_equal":True})
 return rows
