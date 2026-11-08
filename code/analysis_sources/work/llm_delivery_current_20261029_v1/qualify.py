"""Finite synthetic operator qualification and active-formal refusal; no clinical performance."""
import ast,hashlib,random,math,resource,sys,time
from pathlib import Path
import common as c
import independent_metrics as m
def run():
 started=time.monotonic();checks=m.selftest()["checks"];worst=0.
 rng=random.Random(20261029)
 for trial in range(50):
  n=8+rng.randrange(12);y=[rng.randrange(2) for _ in range(n)]
  p=[rng.choice((0.,.1,.2,.5,.8,.9,1.)) for _ in range(n)]
  g=[f"g{i//2}" for i in range(n)];r=[str(i) for i in range(n)]
  w=[rng.randrange(4) for _ in range(n)]
  if not sum(w):w[0]=1
  got=m.metrics(y,p,g,r,w)
  positive=[i for i in range(n) for _ in range(w[i]) if y[i]]
  negative=[i for i in range(n) for _ in range(w[i]) if not y[i]]
  if positive and negative:
   auc=math.fsum(1. if p[i]>p[j] else .5 if p[i]==p[j] else 0. for i in positive for j in negative)/(len(positive)*len(negative))
   assert m.close(got["AUROC"],auc,1e-14);worst=max(worst,abs(got["AUROC"]-auc));checks+=1
  if positive:
   ap=math.fsum(sum(w[j]*y[j] for j in range(n) if p[j]>=p[i])/sum(w[j] for j in range(n) if p[j]>=p[i]) for i in positive)/len(positive)
   assert m.close(got["AP"],ap,1e-14);worst=max(worst,abs(got["AP"]-ap));checks+=1
  order=list(range(n));rng.shuffle(order)
  other=m.metrics([y[i] for i in order],[p[i] for i in order],[g[i] for i in order],[r[i] for i in order],[w[i] for i in order])
  for k,v in got.items():assert m.close(other[k],v,1e-12),(trial,k);checks+=1
 for weights in ([.5,1],[1,-1],[float("nan"),1]):
  try:m.metrics([0,1],[.1,.8],["g0","g1"],["1","2"],weights)
  except AssertionError:checks+=1
  else:raise AssertionError("invalid weights accepted")
 try:c.formal_guard()
 except AssertionError as e:
  assert str(e)=="FORMAL_NOT_CLOSED_NO_REAL_COMPARISON_ALLOWED"
  guard="ACTIVE_FORMAL_REFUSED_REAL_RESULT_ACCESS"
 else:guard="FORMAL_ALREADY_COMPLETE_REFUSAL_TEST_NOT_APPLICABLE"
 files={str(p):c.sha(p) for p in sorted(c.W.glob("*.py"))}
 for p in c.W.glob("*.py"):ast.parse(p.read_text())
 # Static mutation boundary: new helpers do not import frozen training or execute fitting.
 for p in (c.W/"verify_results.py",c.W/"resources.py",c.W/"render_report.py"):
  tree=ast.parse(p.read_text())
  assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ("fit","partial_fit","backward") for n in ast.walk(tree))
 lock=c.W/"DELIVERY_CODE_LOCK.json"
 payload={"status":"FROZEN_DELIVERY_IMPLEMENTATION_BEFORE_REAL_RESULT_ACCESS","files":files,
          "source_lock_sha256":c.SOURCE_SHA,"formal_training_contract_sha256":c.FORMAL_SHA,
          "statistics_code_lock_sha256":c.STATS_LOCK_SHA,
          "independent_bootstrap_replicates":[0,1,24,25,63,127,255,511,1023,1535,1999],
          "qualification_scope":"Synthetic scalar operators, parsing, mutation boundary and active-formal refusal; future real IO/plots still require actual execution."}
 if lock.exists():assert c.read(lock)==payload
 else:c.save(lock,payload)
 dest=c.L/"delivery_preparation_v1";dest.mkdir(mode=0o700,parents=True,exist_ok=True)
 c.save(dest/"DELIVERY_PREPARATION_ACCEPTANCE.json",{"status":"PASS_SYNTHETIC_INDEPENDENT_OPERATORS_AND_GUARDED_DELIVERY_CODE_PREPARATION_ONLY",
   "time":c.now(),"checks":checks,"max_abs_operator_discrepancy":worst,"active_formal_guard":guard,
   "code_lock":str(lock),"code_lock_sha256":c.sha(lock),"files":files,
   "peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
   "elapsed_seconds":time.monotonic()-started,"real_LLM_performance_computed":False,
   "actual_full_delivery_executed":False,"final_scientific_review_complete":False})
 print("QUALIFIED_PREPARATION_ONLY",checks,worst,guard)
if __name__=="__main__":run()
