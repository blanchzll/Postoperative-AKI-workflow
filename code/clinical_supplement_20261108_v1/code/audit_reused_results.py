from pathlib import Path
import os,json,hashlib,time
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
P=Path('/home/lei/codex-workspaces/aki-prediction-manuscript');C=P/'correction_runs/full_data_20261025_postslot';O=C/'outputs';A=O/'novelty_strengthening_20261101_v1';D=A/'resource_comparison';R=Path(__file__).resolve().parents[1];H={};start=time.time()
def sha(p):
 p=Path(p);h=hashlib.sha256(p.read_bytes()).hexdigest();H[str(p)]=h;return h
def j(p):sha(p);return json.loads(Path(p).read_text())
def read(p):sha(p);return pd.read_csv(p,float_precision='round_trip')
def npz(p):sha(p);return np.load(p,allow_pickle=False)
def save(n,d):(R/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n')
def check(a,b):np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-10,equal_nan=True)
def main():
 matrix={i['comparison_id']:i for i in j(A/'ANALYSIS_PROTOCOL_LOCK.json')['matrix']};bands=read(P/'writing_runs/method_stats_revision_20261106_v1/resource_analysis_v2/resource_analysis_v2.csv');out=[];approx=[]
 # Independently count events after explicitly expanding group-resampled records. No call to original statistic code.
 for name in ['geography_9v27_champ','temporal_9v27_champ','geography_27v38_champ','temporal_27v38_champ']:
  it=matrix[name];ctx=it['context'];ww=npz(D/(ctx+'_BOOTSTRAP_WEIGHTS_PRIVATE.npz'));cache=npz(D/(ctx+'_PREDICTION_CACHE_PRIVATE.npz'));orig=npz(D/(name+'__calibrated_NEW_FUNCTION_SAMPLES_PRIVATE.npz'));grid=read(D/(name+'__calibrated_GRID.csv'));orders=[]
  for side in ['base','extra_information']:
   pr=Path(it[side]['prediction']);h=hashlib.sha256((str(pr)+'|calibrated').encode()).hexdigest()[:20];pred=read(pr);lookup=pred.set_index('excel_row');lookup.index=lookup.index.astype(str);aligned=lookup.loc[ww['records'].astype(str)];check(aligned.prob_calibrated,cache[h+'_p']);check(aligned.y,ww['y']);orders.append(cache[h+'_order'])
  y=ww['y'].astype(int);weights=ww['weights'];arr=[]
  for weight in np.vstack([np.ones((1,len(y)),int),weights]):
   n=int(weight.sum());K=int(np.ceil(.2*n));b=y[orders[0]].repeat(weight[orders[0]].astype(int));u=y[orders[1]].repeat(weight[orders[1]].astype(int));ub=np.r_[0,np.cumsum(u)];base=int(b[:K].sum());k=K-np.arange(31)*n//100
   # floor(K-c*n/100), preserving exact integer arithmetic rather than floor of floating costs.
   k=np.floor_divide(100*K-np.arange(31)*n,100);feas=(100*K-np.arange(31)*n)>=0;capture=ub[np.maximum(k,0)];delta=(capture-base)*1000/n
   arr.append({'delta':delta,'feasible':feas,'base':base*1000/n})
  raw=np.stack([v['delta'] for v in arr]);feas=np.stack([v['feasible'] for v in arr]);base=np.array([v['base'] for v in arr]);field=list(orig['fields']).index('delta_captured_per1000');stored=orig['samples'].reshape(2000,6,31,8)[:,3,:,field]
  check(np.where(feas[1:],raw[1:],np.nan),stored)
  g=grid[grid.budget_percent.eq(20)];check(g.delta_captured_per1000,np.where(feas[0],raw[0],np.nan))
  for rule in ['fallback','feasible_only','post_spend_zero']:
   bb=np.where(feas[1:],raw[1:],0 if rule=='fallback' else np.nan if rule=='feasible_only' else -base[1:,None]);theta=np.where(feas[0],raw[0],0 if rule=='fallback' else np.nan if rule=='feasible_only' else -base[0]);domain=feas[0]&feas[1:].all(0) if rule=='feasible_only' else np.ones(31,bool)
   sd=np.std(bb[:,domain],axis=0,ddof=1);constant=np.all(bb[:,domain]==bb[:1,domain],axis=0);sd[constant]=0;nonzero=sd>0;maxdev=np.max(np.abs((bb[:,domain][:,nonzero]-theta[domain][None,nonzero])/sd[None,nonzero]),axis=1) if nonzero.any() else np.zeros(2000);q=np.quantile(maxdev,.95);qfamily=np.quantile(maxdev,.9875)
   z=bands[bands.comparison.eq(name)&bands.budget_percent.eq(20)&bands.rule.eq(rule)].sort_values('cost');assert len(z)==31;check(z.point_theta,theta);check(z.loc[domain,'bootstrap_sd'],sd);check(z.singlecurve_q_exact,q);check(z.loc[domain,'singlecurve_band_lo'],theta[domain]-q*sd);check(z.loc[domain,'singlecurve_band_hi'],theta[domain]+q*sd)
   if rule=='fallback':check(z.family_q_exact,qfamily);check(z.fourcurve_family_band_lo,theta-qfamily*sd);check(z.fourcurve_family_band_hi,theta+qfamily*sd)
   approx.append({'comparison':name,'rule':rule,'cost_points_in_joint_domain':int(domain.sum()),'singlecurve_q':float(q),'family_q_9875':float(qfamily) if rule=='fallback' else None,'status':'PASS_fresh_expanded_patient_numeric_rebuild','inference':'fixed_sd max-deviation approximate bands, not exact finite-sample coverage; fourcurve family scope20pct fallback only'})
 # Target-rate null Brier and frozen-source constant; no target-rate calibration disguised as external prediction.
 m=A/'mimic_calendar_v2';z=npz(m/'BOOTSTRAP_INPUT_PRIVATE.npz');y=z['y'];w=z['weights'];den=w.sum(1);pi=(w@y)/den;targetnull=pi*(1-pi);nn=len(y);ybar=float(y.mean());source=read(P/'writing_runs/method_stats_revision_20261106_v1/mimic_analysis/SOURCE_CONSTANTS.csv');old=read(P/'writing_runs/method_stats_revision_20261106_v1/mimic_analysis/NULL_BRIER_BSS.csv');perf=read(P/'writing_runs/method_stats_revision_20261106_v1/mimic_analysis/PERFORMANCE.csv')
 for line in ['geography','temporal']:
  s=float(source.loc[source.line.eq(line),'fixed_rate'].iloc[0]);
  for kind in ['raw','calibrated']:
   p=z[line+'__'+kind];brier=float(np.square(y-p).mean());bs=(w@np.square(y-p))/den
   for reference,pointnull,bootnull in [('target_rate_reestimated',ybar*(1-ybar),targetnull),('fixed_source_development_rate',float(np.mean(np.square(y-s))),(w@np.square(y-s))/den)]:
    candidates=old.loc[old.subgroup.eq('overall')&old.model.eq(line+'__'+kind)&old.reference.eq(reference)]
    if candidates.empty:
     candidates=old.loc[old.subgroup.eq('overall')&old.model.eq(line+'__'+kind)&old.reference.str.contains('source')]
    val={'null_Brier':pointnull,'model_minus_null_Brier':brier-pointnull,'BSS':1-brier/pointnull};bv={'null_Brier':bootnull,'model_minus_null_Brier':bs-bootnull,'BSS':1-bs/bootnull}
    for metric,value in val.items():
     a=candidates[candidates.metric.eq(metric)]
     assert len(a)==1,(reference,metric,candidates.metric.tolist());check(a.estimate.iloc[0],value);ci=np.quantile(bv[metric],[.025,.975]);check(a[['lower95','upper95']].to_numpy()[0],ci)
     out.append({'population':'MIMIC_current759_59','line':line,'probability':kind,'reference':reference,'metric':metric,'estimate':value,'lo':ci[0],'hi':ci[1],'N':nn,'events':int(y.sum()),'independent_numeric_check':'PASS','source_constant':s if 'source' in reference else None,'reference_use':'target observed prevalence is descriptive apparent comparator, reestimated each replicate' if 'target' in reference else 'fixed source training constant, no target adaptation'})
 # Known optimism evidence, preserving boundary between fixed family and full model-selection procedure.
 optimism=[];counts=[]
 src=O/'v4_development_inputs_20261018';sl=j(src/'SOURCE_LOCK.json');assert sha(src/'SOURCE_LOCK.json')=='afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3'
 meta=read(src/'preop_metadata.csv').set_index('excel_row');proc=read(O/'ensemble_calibration_20261020_v1/calibrated_internal/PRIVATE_INNER_SELECTED_CALIBRATED_PREDICTIONS.csv')
 for keys,q in proc.groupby(['line','cohort','stage']):
  counts.append({'line':keys[0],'cohort':keys[1],'stage':keys[2],'N':len(q),'events':int(q.y.sum()),'selected_families':';'.join(sorted(q.selected_family.unique())),'selected_branches':';'.join(sorted(q.selected_branch.unique())),'role':'procedure-selected nested predictions; not interchangeable with each fixed final route'})
 for branch in ['compact','full']:
  ref=O/f'conditional_stability_20261024_v1/references/geography_preop_preop_{branch}';zz=npz(ref/'predictions.npz');a=meta.loc[zz['excel_row'].astype(int)];check(a.y,zz['y']) if 'y' in zz else None
  apparent=roc_auc_score(a.y,zz['cal']);outer=[]
  for fold in range(5):
   folder=O/f'ensemble_continuation_20261018_v1/ensemble_results/geography/preop/preop/{branch}/outer_{fold}';f=folder/'LR_calibrated_outer.csv';rec=j(folder/'COMPLETE.json');assert sha(f)==rec['files'][f.name];d=read(f);root=rec['signature']['root'];assert set(meta.loc[root['train_rows'],'group_hash']).isdisjoint(set(d.group_hash));outer.append(d)
  cv=pd.concat(outer).set_index('excel_row');assert set(cv.index)==set(a.index);check(cv.y,meta.loc[cv.index,'y']);outerauc=roc_auc_score(cv.y,cv.prob_calibrated if 'prob_calibrated' in cv else cv.p)
  optimism.append({'branch':branch,'fixed_family':'LR','N':len(a),'events':int(a.y.sum()),'development_apparent_AUROC':apparent,'outer_fixed_family_AUROC':outerauc,'apparent_minus_outer':apparent-outerauc,'full_selection_optimism_corrected':False,'meaning':'Different apparent/fixed-family CV estimands; does not estimate winner-selection optimism for whole workflow'})
 pd.DataFrame(out).to_csv(R/'results/MIMIC_BRIER_REVERIFIED.csv',index=False);pd.DataFrame(approx).to_csv(R/'results/ORIGINAL_GRID_BANDS_REVERIFIED.csv',index=False);pd.DataFrame(optimism).to_csv(R/'results/DEVELOPMENT_OPTIMISM_AUDIT.csv',index=False);pd.DataFrame(counts).to_csv(R/'audit/NESTED_PROCEDURE_PREDICTION_AUDIT.csv',index=False)
 save('audit/REUSED_RESULT_INPUT_HASHES.json',H);save('audit/REUSED_RESULT_VERIFICATION.json',{'status':'PASS_FRESH_NUMERIC_SOURCE_BOUND_REVIEW','seconds':time.time()-start,'MIMIC_Brier_rows':len(out),'band_curves':len(approx),'model_selection_optimism_correction':False,'new_fits':0,'files_unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in H.items())});print('REUSED_RESULTS_REVERIFIED',flush=True)
if __name__=='__main__':main()
