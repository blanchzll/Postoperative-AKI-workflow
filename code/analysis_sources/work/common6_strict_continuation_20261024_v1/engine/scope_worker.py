"""One finite coarse selection or selected-base fit in an isolated scope."""
from core import *
from preprocess import impute
import argparse,gc,resource
def coarse_scope(s,threads):
 from coarse import score_task
 from coordinator_a import select
 line,cohort=s["line"],s["cohort"]
 for method in METHODS:
  for p in parts(line,cohort,"outer_0"):
   impute(pair_task(line,cohort,"preop","outer_0",p,method),threads);gc.collect()
  for family in FAMILIES:
   t={"kind":"coarse","line":line,"cohort":cohort,"context":"outer_0","method":method,"family":family,"scope_id":s["id"],"scope_sha256":sha(Path(s["input"]).parent/"SCOPE.json")}
   score_task(t,threads);gc.collect()
  atomic(OUT/"COARSE_PROGRESS.json",{"last_method":method,"updated":time.time()})
 assert select(line,cohort,"outer_0")
 atomic(OUT/"COARSE_ACCEPTANCE.json",{"status":"PASS","scope":s["id"],"source_scope_sha256":sha(Path(s["input"]).parent/"SCOPE.json"),"selected_method":json.loads((OUT/"selection"/line/cohort/"outer_0/METHOD_SELECTION.json").read_text())["selected_method"],"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
def fitted(s,t,threads):
 from raw_outer import run
 run(t,threads)
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("phase");a.add_argument("scope");a.add_argument("--task");a.add_argument("--threads",type=int,default=1);v=a.parse_args()
 s=json.loads(Path(v.scope).read_text())
 assert OUT==Path(s["output"]) and INPUT==Path(s["input"])
 if v.phase=="coarse":coarse_scope(s,v.threads)
 elif v.phase=="fit":fitted(s,json.loads(Path(v.task).read_text()),v.threads)
 else:raise ValueError(v.phase)
