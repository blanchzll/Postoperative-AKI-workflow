"""Finite, resumable model-lock supervisor. Waits for E; never reads final evaluation scores."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sys,os,json,hashlib,time,fcntl,subprocess,traceback
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/common6_model_lock_20261024_v1'
E=ROOT/'outputs/common6_internal_assembly_20261024_v1'
PYTHON=ROOT/'envs/prediction_gpu/bin/python'
OWNED_WORKERS={}
OWNED_GUARD=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):return json.loads(Path(p).read_text())

def atomic(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with q.open('w') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 q.replace(p)

def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

def pick(rows):
 best=max(float(r['AP_mean']) for r in rows);rows=[r for r in rows if float(r['AP_mean'])>=best-1e-12]
 best=min(float(r['Brier_mean']) for r in rows);rows=[r for r in rows if float(r['Brier_mean'])<=best+1e-12]
 return min(rows,key=lambda r:json.dumps({k:r[k] for k in ['line','cohort','stage','branch','family']},sort_keys=True,separators=(',',':')))

AB=ROOT/'outputs/common6_development_20261024_v1'
SIX={'Age','Sex','PreopCr','PreopBUN','PreopAlb','PreopHb'}
def check():
 for f,h in read(OUT/'CODE_LOCK.json').items():assert sha(CODE/f)==h
 for f,h in read(OUT/'DEPENDENCIES.json').items():assert sha(f)==h
def configure(line):
 import runtime as r
 r.SOURCE=AB/'inputs'/line
 r.BASE=AB/'runs'/line
 r.D=E
 r.require_e()
 return r
def selection():
 import pandas as pd
 import runtime as r
 r.require_e();r.checked(E/'calibrated_internal/ACCEPTANCE.json')
 table=E/'calibrated_internal/CALIBRATED_ELEVEN_FAMILY_SUMMARY.csv'
 df=pd.read_csv(table).assign(cohort='preop',stage='preop',branch='full')
 assert len(df)==22 and df.groupby('line').size().eq(11).all()
 roots={x['line']:x for x in read(E/'SCOPE_MANIFEST.json')['roots'] if x['context']=='final'}
 routes=[];jobs={};winners=[]
 for line,g in df.groupby('line',sort=True):
  r=configure(line);w=pick(g.to_dict('records'));winners.append(w)
  root=roots[line];family=w['family']
  methodfile=r.BASE/'selection'/line/'preop/final/METHOD_SELECTION.json'
  t={'line':line,'cohort':'preop','stage':'preop','branch':'full','family':family,'train_rows':root['train_rows'],
   'method':read(methodfile)['selected_method'],'method_receipt':str(methodfile),'method_receipt_sha256':sha(methodfile),
   'input_source_lock_sha256':sha(r.SOURCE/'SOURCE_LOCK.json'),'feature_csv_sha256':sha(r.SOURCE/'preop_preop_features.csv'),
   'metadata_sha256':sha(r.SOURCE/'preop_metadata.csv')}
  raw=E/'ensemble_results'/line/'preop/preop/full/final';r.checked(raw/'COMPLETE.json')
  oof=raw/'honest_base_oof.csv';z=pd.read_csv(oof,dtype={'group_hash':str},float_precision='round_trip')
  if family in r.FAMILIES:members=[family];ep=None
  else:
   conf=read(raw/'SELECTION.json')['winners'][family];members=r.rank_families(z)[:conf['k']];ep={'k':conf['k'],'C':conf['C']}
  for member in members:
   sp=r.BASE/'search_v2'/line/'preop/preop/full/final'/member/'COMPLETE.json';sr=r.checked(sp);best=sr['selected']
   assert sha(best['prediction_file'])==best['prediction_sha256']
   st={**t,'family':member,'params':best['params'],'search_receipt':str(sp),'search_receipt_sha256':sha(sp)}
   jobs[r.base_id(st)]=st
  routes.append({**t,'role':'champion','members':members,'ensemble_params':ep,'primary_route_winner':True,
   'winner_internal_selection_scores':w,'development_raw_oof':str(oof),'development_raw_oof_sha256':sha(oof),'root':root['id']})
 result={'score_source':str(table),'score_source_sha256':sha(table),'route_winners':winners,
  'selection_rule':'Calibrated candidate mean outer AP then Brier and canonical ID, tolerance 1e-12; inherited V1 family choice.',
  'routes_and_references':routes,'base_jobs':jobs,'public_scores_accessed':False,
  'candidate_scores_not_unbiased_selected_procedure_scores':True,
  'internal_acceptance_sha256':sha(E/'STAGE_INTERNAL_ACCEPTANCE.json')}
 p=OUT/'PRIVATE_MODEL_SELECTION_AND_JOBS.json'
 if p.exists():assert read(p)==result
 else:atomic(p,result)
 return result
def route_selection_oof(t):
 import pandas as pd
 import runtime as r
 d=E/'calibrated_internal'/t['line']/'final/inner_selected_procedure'
 rec=r.checked(d/'COMPLETE.json')
 z=pd.read_csv(d/'PRIVATE_HONEST_CALIBRATION_OOF.csv',dtype={'group_hash':str},float_precision='round_trip')
 assert z.excel_row.is_unique and set(z.excel_row)==set(t['train_rows'])
 return z,{'common6_complete_inner_family_selection':True,'source_receipt':str(d/'COMPLETE.json'),
  'source_receipt_sha256':sha(d/'COMPLETE.json'),'source_lineage':read(d/'LINEAGE.json'),
  'scored_patients_excluded_from_child_selection_and_fitting':True}

def finish_routes(selection):
 import numpy as np,pandas as pd,joblib
 import runtime as r
 locks=[]
 for t in selection['routes_and_references']:
  d=OUT/'locked_routes'/r.route_id(t)/t['role']/t['family'];d.mkdir(parents=True,exist_ok=True)
  if (d/'ACCEPTANCE.json').exists():
   old=read(d/'MODEL_LOCK.json');acc=read(d/'ACCEPTANCE.json')
   assert sha(d/'MODEL_LOCK.json')==acc['lock_sha256']
   assert old['source_selection_sha256']==sha(OUT/'PRIVATE_MODEL_SELECTION_AND_JOBS.json')
   for comp in list(old['base_components'].values())+[old['calibrator']]+([old['meta']] if old['meta'] else [])+[old['calibration_training_scores']]:
    assert sha(comp['path'])==comp['sha256']
   locks.append({'path':str(d/'MODEL_LOCK.json'),'sha256':sha(d/'MODEL_LOCK.json'),'identity':old['identity'],'primary_route_winner':old['primary_route_winner']})
   continue
  source=pd.read_csv(t['development_raw_oof'],dtype={'group_hash':str},float_precision='round_trip')
  m=pd.read_csv(r.SOURCE/(t['cohort']+'_metadata.csv'),dtype={'group_hash':str}).set_index('excel_row',drop=False)
  assert source.excel_row.is_unique and set(source.excel_row)==set(t['train_rows'])
  mm=m.loc[source.excel_row]
  assert source.group_hash.tolist()==mm.group_hash.tolist() and source.y.tolist()==mm.y.tolist()
  assert mm[t['line']+'_partition_v2'].eq('development').all()
  components={};union=set()
  for family in t['members']:
   bd=OUT/'base_fits'/('/'.join([t[k] for k in ['line','cohort','stage','branch']])+'/'+family)
   br=r.checked(bd/'COMPLETE.json');bp=bd/'raw_bundle.joblib'
   components[family]={'path':str(bp),'sha256':sha(bp)}
   union.update(br['input_columns'])
  meta=None
  if t['role']=='champion':
   training,cal_lineage=route_selection_oof(t)
   cal=r.calibrator(training);joblib.dump(cal,d/'calibrator.joblib',compress=3)
   atomic(d/'CALIBRATION_LINEAGE.json',cal_lineage)
  if t['family'] in r.FAMILIES:
   if t['role']!='champion':
    training=source[['excel_row','group_hash','y',t['family']]].rename(columns={t['family']:'prob_raw'})
    cal=r.calibrator(training);joblib.dump(cal,d/'calibrator.joblib',compress=3)
  else:
   ec=r.E/'calibrated_internal/candidate_calibration'/t['line']/t['cohort']/t['stage']/t['branch']/'final'/t['family']
   if t['role']!='champion':
    cal=joblib.load(ec/'calibrator.joblib')
    joblib.dump(cal,d/'calibrator.joblib',compress=3)
    training=pd.read_csv(ec/'PRIVATE_HONEST_CALIBRATION_OOF.csv',dtype={'group_hash':str},float_precision='round_trip')
   assert training.excel_row.is_unique and set(training.excel_row)==set(t['train_rows'])
   tm=m.loc[training.excel_row]
   assert training.group_hash.tolist()==tm.group_hash.tolist() and training.y.tolist()==tm.y.tolist()
   if t['family']=='Stacking':
    meta=r.LogisticRegression(C=t['ensemble_params']['C'],penalty='l2',solver='lbfgs',max_iter=5000,random_state=42)
    with r.threadpool_limits(limits=1):meta.fit(source[t['members']],source.y)
    joblib.dump(meta,d/'meta.joblib',compress=3)
   else:meta=None
  training['prob_calibrated']=r.apply_calibration(cal,training.prob_raw)
  cuts=r.thresholds(training.y.to_numpy(),training.prob_calibrated.to_numpy())
  training.to_csv(d/'PRIVATE_TRAINING_THRESHOLD_SCORES.csv',index=False)
  sp=r.source_spec(t['stage']);columns=[c for c in sp['num']+sp['cat'] if c in union]
  manifest={'version':'common6_source_locked_private_pipeline_v1','identity':{k:t[k] for k in ['line','cohort','stage','branch','role','family']},
   'calibration_scope':('training-child family selection within fixed input branch' if t['role']=='champion' else 'fixed-family controlled reference'),
   'family':t['family'],'members':t['members'],'primary_route_winner':t['primary_route_winner'],
   'input_columns':columns,'actual_input_count':len(columns),'base_components':components,
   'calibrator':{'path':str(d/'calibrator.joblib'),'sha256':sha(d/'calibrator.joblib')},
   'meta':({'path':str(d/'meta.joblib'),'sha256':sha(d/'meta.joblib')} if meta is not None else None),
   'training_thresholds':cuts,'case_key_required_for_stochastic_completion':True,
   'calibration_training_scores':{'path':str(d/'PRIVATE_TRAINING_THRESHOLD_SCORES.csv'),'sha256':sha(d/'PRIVATE_TRAINING_THRESHOLD_SCORES.csv')},
   'calibration_lineage':({'path':str(d/'CALIBRATION_LINEAGE.json'),'sha256':sha(d/'CALIBRATION_LINEAGE.json')} if t['role']=='champion' else None),
   'input_spec_source':str(r.SOURCE/'FEATURE_SPECS.json'),'input_source_sha256':sha(r.SOURCE/'SOURCE_LOCK.json'),
   'training_rows_sha256':r.digest(t['train_rows']),'training_n':len(t['train_rows']),
   'training_groups_sha256':r.digest(sorted(mm.group_hash.tolist())),'model_runtime':str(PYTHON),
   'inference_code_sha256':sha(CODE/'runtime.py'),'both_common6_source_model_locks_required_before_public_scoring':True,
   'source_selection_sha256':sha(OUT/'PRIVATE_MODEL_SELECTION_AND_JOBS.json'),
   'contains_training_information_keep_private':True}
  # Verify complete private pipeline using only full-development training probes.
  raw,meta_rows=r.data(t,r.source_spec(t['stage'],columns))
  probe=raw.loc[raw.isna().any(axis=1)].head(4)
  if len(probe)<4:probe=raw.head(4)
  pm=meta_rows.loc[probe.index];ids=r.keys(pm)
  p,pc=r.route_prediction(manifest,probe,ids)
  rev,rc=r.route_prediction(manifest,probe.iloc[::-1],ids[::-1])
  solo=[r.route_prediction(manifest,probe.iloc[i:i+1],ids[i:i+1]) for i in range(len(probe))]
  np.testing.assert_allclose(p,rev[::-1],atol=1e-12,rtol=1e-12)
  np.testing.assert_allclose(pc,rc[::-1],atol=1e-12,rtol=1e-12)
  np.testing.assert_allclose(p,np.concatenate([v[0] for v in solo]),atol=1e-12,rtol=1e-12)
  np.testing.assert_allclose(pc,np.concatenate([v[1] for v in solo]),atol=1e-12,rtol=1e-12)
  for cut_name in ['training_sensitivity80','training_capacity10','training_capacity20']:
   cut=cuts[cut_name]
   np.testing.assert_array_equal(pc>=cut,rc[::-1]>=cut)
   np.testing.assert_array_equal(pc>=cut,np.concatenate([v[1] for v in solo])>=cut)
  atomic(d/'MODEL_LOCK.json',manifest)
  reloaded=read(d/'MODEL_LOCK.json');rp,rpc=r.route_prediction(reloaded,probe,ids)
  np.testing.assert_allclose(p,rp,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(pc,rpc,atol=1e-12,rtol=1e-12)
  atomic(d/'ACCEPTANCE.json',{'status':'PASS_private_model_route_lock','lock_sha256':sha(d/'MODEL_LOCK.json'),
   'single_batch_reorder_reload_verified':True,'thresholds_training_only':True,'final_scores_accessed':False})
  locks.append({'path':str(d/'MODEL_LOCK.json'),'sha256':sha(d/'MODEL_LOCK.json'),
   'identity':manifest['identity'],'primary_route_winner':manifest['primary_route_winner']})
  del source,training,raw;gc=__import__('gc');gc.collect()
 atomic(OUT/('LINE_LOCK_'+selection['line']+'.json'),{'status':'PASS','time_beijing':stamp(),'models':locks,
  'primary_routes':sum(x['primary_route_winner'] for x in locks),'selection_sha256':sha(OUT/'PRIVATE_MODEL_SELECTION_AND_JOBS.json'),
  'final_model_locked':True,'public_scoring_permitted':False,
  'next_required':'Both common6 source-line locks and public endpoint contract verification',
  'entire_research_complete':False})

def main():
 guard=(OUT/'SUPERVISOR.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);check()
 assert read(OUT/'QUALIFICATION.json')['status']=='PASS_common6_lock_engineering_only'
 atomic(OUT/'LAUNCH.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],
  'time_beijing':stamp(),'code_lock_sha256':sha(OUT/'CODE_LOCK.json')})
 while not (E/'STAGE_INTERNAL_ACCEPTANCE.json').exists():
  check()
  if (E/'FAILURE.json').exists():raise RuntimeError('Upstream common6 internal stage failed')
  atomic(OUT/'status.json',{'state':'waiting_common6_strict_internal_acceptance','pid':os.getpid(),'time_beijing':stamp(),'public_scores_accessed':False})
  time.sleep(30)
 upstream=(E/'SUPERVISOR.lock').open('a')
 while True:
  try:fcntl.flock(upstream,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:time.sleep(10)
 sel=selection();locks=[]
 for line in ['geography','temporal']:
  r=configure(line)
  # Small final component count; sequential fits avoid duplicating imputer memory.
  for name,t in sel['base_jobs'].items():
   if t['line']!=line:continue
   check()
   while int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024<6*1024**3:time.sleep(30)
   atomic(OUT/'status.json',{'state':'fitting_common6_final_components','component':name,'pid':os.getpid(),'time_beijing':stamp(),'public_scores_accessed':False})
   r.fit_base(t)
  check();finish_routes({'line':line,'routes_and_references':[t for t in sel['routes_and_references'] if t['line']==line]})
  locks.extend(read(OUT/('LINE_LOCK_'+line+'.json'))['models'])
 assert len(locks)==2
 for x in locks:
  assert sha(x['path'])==x['sha256']
  m=read(x['path']);assert set(m['input_columns'])==SIX and m['actual_input_count']==6
 check()
 atomic(OUT/'STAGE_COMMON6_MODEL_LOCK_ACCEPTANCE.json',{'status':'PASS_two_common6_source_model_locks','models':locks,
  'code_lock_sha256':sha(OUT/'CODE_LOCK.json'),'internal_acceptance_sha256':sha(E/'STAGE_INTERNAL_ACCEPTANCE.json'),
  'public_scores_accessed':False,'original_9_27_38_models_validated':False,'whole_project_complete':False})
 atomic(OUT/'status.json',{'state':'complete_common6_model_locks_public_scoring_pending','owned_workers':0,'time_beijing':stamp()})
 upstream.close();guard.close()
if __name__=='__main__':
 try:main()
 except Exception:
  atomic(OUT/'FAILURE.json',{'time_beijing':stamp(),'traceback':traceback.format_exc()});raise
