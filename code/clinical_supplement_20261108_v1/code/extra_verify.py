from pathlib import Path
import os,json,time
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np,pandas as pd
from scipy.special import expit
from scipy.optimize import root
from scipy.stats import rankdata
from fractions import Fraction
R=Path(__file__).resolve().parents[1];P=R.parents[1];O=P/'correction_runs/full_data_20261025_postslot/outputs';S=P/'outputs/full_data_audit_20261025_v1/candidate_release/clinical';start=time.time()
def read(p):return pd.read_csv(p,float_precision='round_trip')
def check(a,b,tol=1e-8):np.testing.assert_allclose(a,b,atol=tol,rtol=tol,equal_nan=True)
registry=read(R/'private/REGISTRY_OUTCOMES.csv').set_index('excel_row');source=read(S/'preop_cohort.csv').set_index('excel_row');M=read(O/'public_eligibility_refinement_20261022_v1/PRIVATE_COMMON6_METADATA.csv').set_index('record_key');prof=read(O/'public_eligibility_refinement_20261022_v1/PRIVATE_FIRST_OPERATION_PROFILE.csv').set_index('op_id');cache={}
def sample(label,outcome):
 if label not in cache:
  if label.startswith('Chinese'):
   _,line,k=label.split('_');branch='compact' if k=='9' else 'full';pr=read(next((O/f'formal_evaluation_20261023_v1/{line}_preop/private_predictions').glob(f'preop_{branch}_champion_*/PREDICTIONS.csv'))).set_index('excel_row');x=registry.loc[pr.index];g=pr.group_hash.to_numpy();p=pr.prob_calibrated.to_numpy();vals={name:x[col].to_numpy(float) for name,col in [('recorded90d_death','X'),('recorded_RRT','U'),('recorded30d_reoperation','V'),('recorded30d_readmission','W'),('total_LOS_days','O'),('postoperative_LOS_days','P')]}
  else:
   _,line,_=label.split('_');pr=read(O/f'common6_public_evaluation_20261024_v1/{line}/PRIVATE_PREDICTIONS.csv').set_index('record_key');mm=M.loc[pr.index];f=prof.loc[mm.op_id];g=pr.group_hash.to_numpy();p=pr.prob_calibrated.to_numpy();death=f.inhosp_death_time.notna()&(f.inhosp_death_time>=f.opend_time)&(f.inhosp_death_time<=f.discharge_time);vals={'recorded_inhospital_death':death.to_numpy(int),'recorded_postwindow_RRT':mm.post_window_RRT_positive.to_numpy(int),'operation_end_to_discharge_days':(f.discharge_time-f.opend_time).to_numpy()/1440}
  cache[label]=(g,p,vals)
 g,p,vals=cache[label];y=vals[outcome];ok=np.isfinite(y)&np.isfinite(p);return g[ok],p[ok],y[ok]
assoc=read(R/'results/HARD_OUTCOME_ASSOCIATIONS.csv');regchecks=0
for _,row in assoc.iterrows():
 if pd.isna(row.get('OR')):continue
 g,p,y=sample(row.population_model,row.outcome);X=np.column_stack([np.ones(len(p)),np.log2(np.clip(p,1e-8,1))]);fit=root(lambda b:X.T@(y-expit(X@b)),[0.,1.]);b=fit.x;assert np.linalg.norm(X.T@(y-expit(X@b)))<1e-5
 mu=expit(X@b);bread=np.linalg.inv(X.T@(X*(mu*(1-mu))[:,None]));un,idx=np.unique(g,return_inverse=True);score=X*(y-mu)[:,None];cluster=np.column_stack([np.bincount(idx,weights=score[:,k]) for k in range(2)]);V=bread@(cluster.T@cluster)@bread*(len(un)/(len(un)-1))*((len(y)-1)/(len(y)-2));se=np.sqrt(V[1,1]);vals=np.exp([b[1],b[1]-1.959963984540054*se,b[1]+1.959963984540054*se]);check(vals,[row.OR,row.lo,row.hi],1e-4);regchecks+=3
los=read(R/'results/LOS_ASSOCIATIONS.csv');loschecks=0
for _,row in los.iterrows():
 g,p,y=sample(row.population_model,row.outcome);ok=y>=0;p=p[ok];y=y[ok];a=rankdata(p);b=rankdata(y);rho=np.corrcoef(a,b)[0,1];check([len(y),np.median(y),np.quantile(y,.25),np.quantile(y,.75),rho],[row.N,row.median_days,row.Q1_days,row.Q3_days,row.Spearman_rho]);loschecks+=5
hard=read(R/'results/HARD_OUTCOME_DISCRIMINATION.csv');hardchecks=0
for _,row in hard.iterrows():
 g,p,y=sample(row.population_model,row.outcome);a=y.sum();b=len(y)-a;auc=(rankdata(p)[y==1].sum()-a*(a+1)/2)/(a*b);check([len(y),a,auc],[row.N,row.events,row.AUROC]);hardchecks+=3
windows=read(R/'results/ORIGINAL_WINDOW_DELETION.csv');windowchecks=0
for _,row in windows.iterrows():
 d=read(R/'private'/f'{row.context}_DECISIONS.csv');s=source.loc[d.excel_row];low=read(R/'results/LOW_RISK_EVENTS.csv');th=low[(low.context==row.context)&low.role.eq('base')&low.rule.eq('training_sensitivity80')].iloc[0].threshold;mask=np.ones(len(d),bool) if row.population=='all_evaluation' else d.pbase.to_numpy()<th
 a=np.array([pd.notna(v) and Fraction(str(v))>=Fraction(3,2)*Fraction(str(b)) for v,b in zip(s.pod1,s.base)]);b=np.array([pd.notna(v) and Fraction(str(v))>=Fraction(3,2)*Fraction(str(bb)) for v,bb in zip(s.pod23,s.base)]);known=(a|b)&(s.y.to_numpy()==1);kept=b if row.window_deleted=='POD1' else a;lost=known&~kept&mask;check([mask.sum(),(known&mask).sum(),lost.sum()],[row.N,row.known_relative_SCr_positive_AKI_events,row.lost_known_relative_confirmation]);windowchecks+=3
 # Independently expand each resampled group for the conditional loss ratio and its percentile interval.
 w=np.load(R/'private'/f'{row.context}_RESAMPLING.npz',allow_pickle=False)['weights'];vals=[]
 for ww in w[1:]:
  idx=np.repeat(np.arange(len(d)),ww);den=int((known&mask)[idx].sum());num=int(lost[idx].sum())
  if den:vals.append(num/den)
 bounds=np.quantile(vals,[.025,.975]) if vals else [np.nan,np.nan];check(bounds,[row.lost_fraction_lo,row.lost_fraction_hi]);windowchecks+=2
(R/'audit/EXTRA_INDEPENDENT_VERIFICATION.json').write_text(json.dumps({'status':'PASS_ALTERNATIVE_SOURCE_MATCHED_IMPLEMENTATIONS','association_OR_and_cluster_covariance_checks':regchecks,'LOS_rank_correlation_and_summary_checks':loschecks,'hard_outcome_rank_point_checks':hardchecks,'Chinese_window_checks':windowchecks,'seconds':time.time()-start,'methods':'likelihood-score root solver + hand cluster sandwich; rank correlation; exact fraction window criteria; expanded bootstrap ratios','not_independent_human_review':True},indent=2));print('EXTRA_VERIFICATION_PASS')
