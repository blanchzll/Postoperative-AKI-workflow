"""Original runtime audit of reused imputer objects; no cross-version unpickling."""
from core import *
from preprocess import StableMixedImputer
import joblib,gc
def audit():
 records=[]
 for p in (OUT/"cache").glob("*/COMPLETE.json"):
  r=json.loads(p.read_text())
  if "imported_compatible_cache" not in r:continue
  assert ready(p.parent)
  imp=joblib.load(p.parent/"imputer.joblib")
  sig=r["signature"]
  assert imp.spec==sig["spec"] and imp.method==sig["method"]
  assert imp.seed==42 and imp.iterations==5 and imp.trees==200 and imp.neighbors==5
  assert imp.cols==sig["spec"]["num"]+sig["spec"]["cat"]
  task=r["task"];meta_path=INPUT/(task["cohort"]+"_metadata.csv")
  import pandas as pd
  m=pd.read_csv(meta_path,dtype={"group_hash":str}).set_index("excel_row",drop=False).loc[sig["train_rows"]]
  expected=hashlib.sha256("\n".join(sorted(keys(m))).encode()).hexdigest()
  assert imp.fit_key_digest_==expected
  records.append({"receipt":str(p.relative_to(OUT)),"method":imp.method,"input_count":len(imp.cols),"fit_key_digest_verified":True})
  del imp;gc.collect()
 atomic(OUT/"CACHE_IMPORT_OBJECT_AUDIT.json",{"status":"PASS","objects":records,"count":len(records),"scope":"compatible original-runtime objects, actual train-key signatures and method settings","not_external_performance_evidence":True})
 print(canonical({"status":"PASS","count":len(records)}))
if __name__=="__main__":audit()
