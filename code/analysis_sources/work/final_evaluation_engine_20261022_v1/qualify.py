"""Finite synthetic qualification. Never reads clinical predictions or final outcomes."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sys,hashlib,json,time,os,resource,importlib.util
import numpy as np
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
OUT=ROOT/'outputs/final_evaluation_engine_20261022_v1'
sys.path.insert(0,str(ROOT/'work/ensemble_calibration_20261020_v1'))
from core import resource_snapshot
from fixed_prediction_metrics import FixedRisk,bootstrap_fixed_predictions

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,d):
 q=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
 with q.open('w') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 q.replace(p)
def main():
 snap=resource_snapshot()
 if snap['available_bytes']<int(4.5*1024**3):
  atomic(OUT/'QUALIFICATION_HOLD.json',{'state':'resource_hold_before_synthetic_batch','resource':snap})
  return
 lock=ROOT/'outputs/ensemble_calibration_20261020_v1/CODE_LOCK.json'
 frozen=json.loads(lock.read_text())
 assert all(sha(ROOT/'work/ensemble_calibration_20261020_v1'/f)==h for f,h in frozen.items())
 code_lock_before=sha(lock)
 p=ROOT/'work/selective_information_20261021_v1/policy_core.py'
 fq=json.loads((ROOT/'outputs/selective_information_20261021_v1/FUNCTIONAL_QUALIFICATION.json').read_text())
 assert sha(p)==fq['source_sha256']['work/selective_information_20261021_v1/policy_core.py']
 sp=importlib.util.spec_from_file_location('frozen_policy_core',p)
 policy=importlib.util.module_from_spec(sp);sp.loader.exec_module(policy)
 rng=np.random.default_rng(8804);checks=0
 for n in [1,2,7,30,121]:
  yy=rng.integers(0,2,n);risk=np.round(rng.random(n),1)
  groups=[str(i//2) for i in range(n)];keys=[str(i) for i in range(n)]
  model=FixedRisk(yy,risk,groups,keys)
  for fraction in [0,.1,.2,.25,.5,1]:
   np.testing.assert_array_equal(model.review_counts(fraction),policy.alert_mask(risk,groups,keys,fraction).astype(int))
   checks+=1
 n=241;groups=[f'g{i//2:03d}' for i in range(n)]
 centers=[str((i//2)%3) for i in range(n)];keys=[str(i) for i in range(n)]
 y=rng.integers(0,2,n);base=np.round(rng.random(n),2);other=np.round(rng.random(n),2)
 started=time.perf_counter()
 result=bootstrap_fixed_predictions(y,{'base':base,'same':base,'alternate':other},groups,keys,centers,'base',replicates=2000)
 elapsed=time.perf_counter()-started
 for row in result['paired_differences']:
  if row['model']=='same' and row['valid_replicates']:
   assert row['estimate']==row['lower95']==row['upper95']==0
 for row in result['point']:assert row['reviewed_n']==int(np.ceil(n*row['review_fraction']))
 assert result['receipt']['requested_replicates']==2000
 log=(OUT/'TEST.log').read_text()
 assert 'Ran 12 tests' in log and log.strip().endswith('OK')
 assert sha(lock)==code_lock_before
 assert all(sha(ROOT/'work/ensemble_calibration_20261020_v1'/f)==h for f,h in frozen.items())
 files={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('fixed_prediction_metrics.py'),
    Path(__file__).with_name('test_metrics.py'),OUT/'TEST.log',OUT/'TEST_INITIAL.log',OUT/'REVIEW_CHANGES.json']}
 skills=[
 '/home/lei/codex-workspaces/skill-management/skills/phd-skills/SKILL.md',
 '/home/lei/codex-workspaces/skill-management/skills/phd-skills/references/compatibility.md',
 '/home/lei/codex-workspaces/skill-management/skills/phd-paper-verification/SKILL.md',
 '/home/lei/codex-workspaces/skill-management/releases/phd-skills/50df7160ab357eff023d200756f1843b3c72f924/plugin/skills/paper-verification/SKILL.md']
 rec={'status':'PASS_synthetic_fixed_risk_evaluation_engine_only',
  'time_beijing':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'pid':os.getpid(),
  'unit_tests_passed':12,'weighted_metrics_match_independent_sklearn_reference':True,
  'frozen_policy_capacity_checks_passed':checks,'synthetic_n':n,
  'synthetic_patient_groups':len(set(groups)),'synthetic_centers':3,'synthetic_bootstrap_replicates':2000,
  'synthetic_bootstrap_seconds':elapsed,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
  'fixed_prediction_bootstrap_receipt':result['receipt'],'source_files':files,
  'E_code_lock_sha256':code_lock_before,'active_scientific_code_unchanged':True,
  'resource_snapshot_before_launch':snap,
  'skill_source_hashes':{f:sha(f) for f in skills},
  'skill_scope':'Formula-code alignment and traceable numerical qualification only; not a full manuscript or independent review.',
  'clinical_data_or_final_model_scores_read':False,'clinical_model_fitted':False,
  'route_models_locked':False,'selective_policy_locked':False,
  'batch_selective_policy_bootstrap_adapter_qualified':False,
  'calibration_CI_or_final_evaluation_loader_qualified':False,
  'entire_research_complete':False}
 atomic(OUT/'FUNCTIONAL_QUALIFICATION.json',rec)
 print(json.dumps({k:v for k,v in rec.items() if k not in ['source_files','skill_source_hashes','fixed_prediction_bootstrap_receipt','resource_snapshot_before_launch']},ensure_ascii=False))
if __name__=='__main__':main()
