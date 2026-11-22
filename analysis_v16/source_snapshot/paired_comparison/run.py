from pathlib import Path
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import sys,json,hashlib,csv,itertools,concurrent.futures,time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
ROOT=Path(os.environ["AKI_WORKSPACE"])
SOURCE=ROOT/'correction_runs/full_data_20261025_postslot'
OUT=ROOT/'writing_runs/v16_continuation_20261117/paired_comparison_20261117'
sys.path.insert(0,str(SOURCE/'work/final_evaluation_engine_20261022_v1'))
from fixed_prediction_metrics import FixedRisk,_design,group_weights
EVAL=SOURCE/'outputs/formal_evaluation_20261023_v1'
METRICS=['AUROC','AP','Brier','observed_expected','sensitivity','PPV','captured_per1000','missed_per1000','false_alerts_per1000']

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(context,model):
    p=EVAL/context/'private_predictions'/model/'PREDICTIONS.csv'
    complete=p.parent/'COMPLETE.json'
    v=json.loads(complete.read_text())
    df=pd.read_csv(p,dtype={'excel_row':str,'group_hash':str,'center':str})
    assert not df.excel_row.duplicated().any()
    assert df.prob_calibrated.between(0,1).all()
    return df,{'file':str(p.relative_to(ROOT)),'sha256':digest(p),'acceptance_sha256':digest(complete),'model':model}

def run(line,scope):
    start=time.monotonic();dest=OUT/f'{line}_{scope}';dest.mkdir(exist_ok=True)
    refs={};provenance=[]
    ids={
        'nine':('preop','preop_compact_champion_LR' if line=='geography' else 'preop_compact_champion_Voting'),
        'full27':('preop','preop_full_champion_LR' if line=='geography' else 'preop_full_champion_Voting'),
        'nine_LR':('preop','preop_compact_reference_LR'),
        'full27_LR':('preop','preop_full_reference_LR')}
    if scope=='common_paired':
        ids.update({'endop38':('paired','endop_full_champion_LR' if line=='geography' else 'endop_full_champion_Voting'),
                    'paired27':('paired','preop_full_champion_Voting' if line=='geography' else 'preop_full_champion_SGBT'),
                    'paired27_LR':('paired','preop_full_reference_LR'),
                    'endop38_LR':('paired','endop_full_reference_LR')})
    for name,(cohort,model) in ids.items():
        refs[name],source=load(line+'_'+cohort,model);provenance.append({'route':name,**source})
    anchor=refs['nine'] if scope=='preop' else refs['endop38']
    anchor=anchor.sort_values('excel_row').reset_index(drop=True)
    predictions={}
    for name,df in refs.items():
        joined=df.set_index('excel_row').loc[anchor.excel_row].reset_index()
        for col in ['group_hash','center','y','surgery_date']:
            assert joined[col].tolist()==anchor[col].tolist(),(line,scope,name,col)
        predictions[name]=joined.prob_calibrated.to_numpy()
    yy,gg,rr,cc,strata,models=_design(anchor.y,anchor.group_hash,anchor.excel_row,anchor.center,predictions)
    pairs=[('full27','nine','main'),('full27_LR','nine_LR','same_LR')]
    if scope=='common_paired':
        pairs += [('endop38','nine','main'),('endop38','full27','main'),('endop38','paired27','original_paired_development'),('endop38_LR','paired27_LR','same_LR_stage')]
    fields=METRICS
    def score(model,w=None):
        base=model.score(w)
        result={}
        for q in [.1,.2]:
            c=model.capacity(q,w,base)
            result[q]={metric:c[metric] for metric in fields}
        return result
    points={name:score(m) for name,m in models.items()}
    draws={(a,b,q,metric):[] for a,b,_ in pairs for q in [.1,.2] for metric in fields}
    rng=np.random.default_rng(20261017);proof=hashlib.sha256();checks=0
    for rep in range(2000):
        w=group_weights(strata,len(yy),rng);proof.update(w.astype('<i8').tobytes())
        scores={name:score(m,w) for name,m in models.items()}
        if rep in [0,999,1999]:
            idx=np.repeat(np.arange(len(yy)),w)
            for name,m in models.items():
                independent=[roc_auc_score(yy[idx],m.p[idx]),average_precision_score(yy[idx],m.p[idx]),brier_score_loss(yy[idx],m.p[idx])]
                assert np.max(np.abs(np.array(independent)-np.array([scores[name][.2][f] for f in ['AUROC','AP','Brier']])))<1e-12
                # Explicit repeated records, with the source tie order preserved.
                expanded_order=np.repeat(m.order,w[m.order]);k=int(np.ceil(.2*len(idx)))
                assert abs(1000*yy[expanded_order[:k]].sum()/len(idx)-scores[name][.2]['captured_per1000'])<1e-12
                checks+=4
        for a,b,_ in pairs:
            for q in [.1,.2]:
                for metric in fields:
                    va=scores[a][q][metric];vb=scores[b][q][metric]
                    if va is not None and vb is not None and np.isfinite(va) and np.isfinite(vb):draws[a,b,q,metric].append(float(va-vb))
    pointrows=[]
    for name,m in models.items():
        for q in [.1,.2]:
            pointrows.append({'line':line,'scope':scope,'model':name,'n':len(yy),'events':int(yy.sum()),'review_fraction':q,**m.capacity(q)})
    diffrows=[]
    for a,b,role in pairs:
        for q in [.1,.2]:
            for metric in fields:
                arr=draws[a,b,q,metric];ci=np.quantile(arr,[.025,.975],method='linear')
                diffrows.append({'line':line,'scope':scope,'role':role,'difference':a+' MINUS '+b,'review_fraction':q,'metric':metric,'estimate':points[a][q][metric]-points[b][q][metric],'lower95':ci[0],'upper95':ci[1],'valid_replicates':len(arr),'requested_replicates':2000})
    pd.DataFrame(pointrows).to_csv(dest/'POINT_ESTIMATES.csv',index=False)
    pd.DataFrame(diffrows).to_csv(dest/'PAIRED_DIFFERENCES.csv',index=False)
    # Reconcile overlapping pairs with accepted original direct-difference intervals.
    accepted=pd.read_csv(EVAL/(line+('_preop' if scope=='preop' else '_paired'))/'PAIRED_DIFFERENCES.csv')
    lookup={r['route']:r['model'] for r in provenance}
    reconciled=0
    for row in diffrows:
        a,b=row['difference'].split(' MINUS ')
        if scope=='common_paired' and (a.startswith('nine') or b.startswith('nine') or a.startswith('full27') or b.startswith('full27')):continue
        oldmetric=row['metric']
        if oldmetric in ['sensitivity','PPV','captured_per1000','missed_per1000','false_alerts_per1000']:oldmetric=f"capacity{int(row['review_fraction']*100)}_{oldmetric}"
        old=accepted[(accepted.model==lookup[a]+' MINUS '+lookup[b])&(accepted.metric==oldmetric)]
        if len(old)==1:
            for f in ['estimate','lower95','upper95']:assert abs(row[f]-float(old.iloc[0][f]))<1e-10,(line,scope,row,old.iloc[0].to_dict())
            reconciled+=1
    receipt={'status':'PASS_DIRECT_PAIRED_BOOTSTRAP_AND_INDEPENDENT_METRIC_CHECK','line':line,'scope':scope,'n':len(yy),'events':int(yy.sum()),'groups':len(set(gg)),'replicates':2000,'seed':20261017,'resample_weight_sha256':proof.hexdigest(),'independent_scalar_checks':checks,'accepted_difference_rows_reconciled':reconciled,'row_identity_and_outcome_alignment':'PASS','sources':provenance,'fixed_predictions_no_new_model_fits':True,'interval':'pointwise percentile 95%, hospital-stratified patient-group bootstrap','seconds':time.monotonic()-start}
    (dest/'ACCEPTANCE.json').write_text(json.dumps(receipt,indent=2))
    return {k:receipt[k] for k in ['line','scope','n','events','status','accepted_difference_rows_reconciled','seconds']}

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    preflight={'cpu_affinity':len(os.sched_getaffinity(0)),'meminfo':{l.split(':')[0]:l.split(':')[1].strip() for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemAvailable:','MemTotal:'))},'scope':'Four finite independent population jobs; no estimator fitting; BLAS1; memory footprints below 1 GiB aggregate expected.'}
    (OUT/'RESOURCE_PREFLIGHT.json').write_text(json.dumps(preflight,indent=2))
    protocol={'systems':['V16 fixed clinical nine','V16 full preoperative27','V16 operation-end38'],'comparison_direction':'second larger-information system minus first; explicit row labels','replicates':2000,'seed':20261017,'metrics':METRICS,'review_fractions':[.1,.2],'sampling':'patients within hospital, all operations retained, same sample across systems','models':'fixed source calibrated predictions; no target fitting or new predictor selection','uncertainty':'conditional on locked systems and observed hospital mixture; pointwise intervals, no multiplicity adjustment','training_population':'full-preoperative nine/27 carry forward into the exact paired target records. Original paired-development27 and same-LR comparisons retained separately.'}
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(run,line,scope) for line in ['geography','temporal'] for scope in ['preop','common_paired']]
        for future in concurrent.futures.as_completed(futures):print(json.dumps(future.result()),flush=True)
