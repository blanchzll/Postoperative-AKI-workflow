from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import csv, hashlib, json, os, sys
from functools import partial
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd

ROOT=Path(os.environ["AKI_WORKSPACE"])
RUN=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'writing_runs/v16_continuation_20261117/paired_comparison_20261117'
V16=ROOT/'writing_runs/jno_positioning_20261121/package/02_Analysis/complete_input_scale/outputs/evaluation'
sys.path.insert(0,str(ROOT/'correction_runs/full_data_20261025_postslot/work/final_evaluation_engine_20261022_v1'))
from fixed_prediction_metrics import _design, group_weights
Q=np.arange(5,31,dtype=np.int64)
ROUTES=('nine','full27','endop38')
PAIRS=(('full27','nine'),('endop38','full27'),('endop38','nine'))
V16_NAMES=dict(zip(ROUTES,('original_nine','original_full27','original_endop38')))
SEED=20261017
BOOT=2000

def digest(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()

def capture(y, order, weights, capacities):
    ordered_weights=weights[order]
    cumulative_records=np.cumsum(ordered_weights)
    cumulative_events=np.cumsum(ordered_weights*y[order])
    last=np.searchsorted(cumulative_records,capacities,side='left')
    return cumulative_events[last]-(cumulative_records[last]-capacities)*y[order[last]]

def diff(a,b,n):return 1000.0*a/n-1000.0*b/n

def minimum_for_target(y, order, weights, target):
    if target == 0:
        return 0
    ordered_weights=weights[order]
    records=np.cumsum(ordered_weights)
    events=np.cumsum(ordered_weights*y[order])
    last=np.searchsorted(events,target,side='left')
    assert y[order[last]]==1 and events[last]>=target
    return int(records[last]-(events[last]-target))

def run(line):
    p=PRIOR/f'{line}_common_paired'
    lock=json.loads((p/'ACCEPTANCE.json').read_text())
    frozen=pd.read_csv(p/'PAIRED_DIFFERENCES.csv')
    orig_points=pd.read_csv(p/'POINT_ESTIMATES.csv')
    base=pd.read_csv(V16/'PAIRED_DIFFERENCES.csv')
    dfs={}
    fingerprints={}
    for route in ROUTES:
        source=next(s for s in lock['sources'] if s['route']==route)
        file=ROOT/source['file']
        assert digest(file)==source['sha256']
        assert digest(file.with_name('COMPLETE.json'))==source['acceptance_sha256']
        dfs[route]=pd.read_csv(file,dtype={'excel_row':str,'group_hash':str,'center':str}).set_index('excel_row',verify_integrity=True)
        fingerprints[route]=source['sha256']
    anchor=dfs['endop38'].sort_index()
    n=len(anchor)
    assert n==lock['n']
    assert int(anchor.y.sum())==lock['events'] and set(anchor.y)=={0,1}
    predictions={}
    for route,data in dfs.items():
        aligned=data.loc[anchor.index]
        pd.testing.assert_frame_equal(aligned[['group_hash','center','y','surgery_date']],anchor[['group_hash','center','y','surgery_date']])
        assert aligned.prob_calibrated.between(0,1).all()
        predictions[route]=aligned.prob_calibrated.to_numpy()
    y, groups, _, _, strata, models=_design(anchor.y,anchor.group_hash,anchor.index,anchor.center,predictions)
    assert len(set(groups))==lock['groups'] and lock['seed']==SEED and lock['replicates']==BOOT
    k=(Q*n+99)//100
    points={name:capture(y,m.order,np.ones(n,dtype=np.int64),k) for name,m in models.items()}
    draws={pair:np.empty((BOOT,len(Q))) for pair in PAIRS}
    inverse_draws={pair:np.empty(BOOT) for pair in PAIRS}
    anchor_index=int(np.flatnonzero(Q==20)[0])
    rng=np.random.default_rng(SEED)
    draw_hash=hashlib.sha256()
    for rep in range(BOOT):
        w=group_weights(strata,n,rng)
        draw_hash.update(w.astype('<i8').tobytes())
        nn=int(w.sum())
        kk=(Q*nn+99)//100
        found={name:capture(y,m.order,w,kk) for name,m in models.items()}
        if rep in (0,999,1999):
            for name,m in models.items():
                explicit=np.repeat(m.order,w[m.order])
                np.testing.assert_array_equal(found[name],np.cumsum(y[explicit])[kk-1])
        for a,b in PAIRS:
            draws[(a,b)][rep]=diff(found[a],found[b],nn)
            target=found[b][anchor_index]
            minimum=minimum_for_target(y,models[a].order,w,target)
            inverse_draws[(a,b)][rep]=1000.0*(minimum-kk[anchor_index])/nn
            if rep in (0,999,1999):
                order=models[a].order
                explicit=np.repeat(order,w[order])
                expected=0 if target==0 else int(np.searchsorted(np.cumsum(y[explicit]),target))+1
                assert minimum==expected
    assert draw_hash.hexdigest()==lock['resample_weight_sha256']
    rows=[]
    for a,b in PAIRS:
        estimate=diff(points[a],points[b],n)
        lo,hi=np.quantile(draws[(a,b)],[.025,.975],axis=0,method='linear')
        for idx,q in enumerate(Q):
            results=[float(estimate[idx]),float(lo[idx]),float(hi[idx])]
            if q in (10,20):
                old=frozen[(frozen.role=='main') & (frozen.difference==f'{a} MINUS {b}') & (frozen.review_fraction==q/100) & (frozen.metric=='captured_per1000')]
                v16=base[(base.line==line)&(base.scope=='common_paired')&(base.model==V16_NAMES[a])&(base.reference==V16_NAMES[b])&(base.metric==f'capacity{q}_captured_per1000')]
                for frame in (old,v16):
                    assert len(frame)==1,(line,q,a,b)
                    np.testing.assert_allclose(results,frame[['estimate','lower95','upper95']].to_numpy()[0],atol=1e-10,rtol=0)
                for name in (a,b):
                    row=orig_points[(orig_points.model==name)&(orig_points.review_fraction==q/100)].iloc[0]
                    assert row.reviewed_n==k[idx] and row.captured_n==points[name][idx]
            rows.append({'evaluation':line,'review_pct':int(q),'eligible_operations':n,'recorded_AKI':int(y.sum()),'operations_selected_per_model':int(k[idx]),'model':V16_NAMES[a],'reference':V16_NAMES[b],'model_AKI_selected':int(points[a][idx]),'reference_AKI_selected':int(points[b][idx]),'difference_per_1000':results[0],'lower_95_pointwise':results[1],'upper_95_pointwise':results[2]})
    inverse=[]
    for a,b in PAIRS:
        k_base=int(k[anchor_index])
        target=int(points[b][anchor_index])
        minimum=minimum_for_target(y,models[a].order,np.ones(n,dtype=np.int64),target)
        lower,upper=np.quantile(inverse_draws[(a,b)],[.025,.975],method='linear')
        inverse.append({'evaluation':line,'eligible_operations':n,
                        'model':V16_NAMES[a],'reference':V16_NAMES[b],
                        'reference_capacity_pct':20,'reference_records_assessed':k_base,
                        'target_AKI_selected':target,'model_records_assessed_at_equal_AKI':minimum,
                        'review_difference_records':minimum-k_base,
                        'review_difference_per_1000':1000.0*(minimum-k_base)/n,
                        'lower_95_pointwise':float(lower),'upper_95_pointwise':float(upper)})
    return rows,inverse,{'evaluation':line,'n':n,'events':int(y.sum()),'group_weight_hash':draw_hash.hexdigest(),'frozen_prediction_input_sha256':fingerprints,'anchors_10_20_exact':True,'explicit_expansion_verified':True}

if __name__=='__main__':
    contract=json.loads((RUN/'PROTOCOL.json').read_text())
    assert list(Q)==contract['review_capacities_pct'] and contract['seed']==SEED and contract['replicates']==BOOT
    with ProcessPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,('geography','temporal')))
    rows=[row for block,_,_ in results for row in block]
    inverse_rows=[row for _,block,_ in results for row in block]
    assert len(inverse_rows)==6
    assert len(rows)==156
    out=RUN/'results'
    out.mkdir(mode=0o700,exist_ok=True)
    with (out/'CAPACITY_GRID.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out/'EQUAL_CAPTURE_WORKLOAD.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(inverse_rows[0]))
        writer.writeheader()
        writer.writerows(inverse_rows)
    summary=[]
    for line in ('geography','temporal'):
        for a,b in PAIRS:
            subset=[r for r in rows if (r['evaluation'],r['model'],r['reference'])==(line,V16_NAMES[a],V16_NAMES[b])]
            summary.append({'line':line,'contrast':f'{a}-{b}','delta_min':min(r['difference_per_1000'] for r in subset),'delta_max':max(r['difference_per_1000'] for r in subset),'ci_excludes_zero_pct':[r['review_pct'] for r in subset if r['lower_95_pointwise']>0 or r['upper_95_pointwise']<0], 'points_10_20_30':[r for r in subset if r['review_pct'] in (10,20,30)]})
    receipt={'status':'PASS','rows':len(rows),'anchor_estimate_and_ci_checks':12,'cohorts':[receipt for _,_,receipt in results],'equal_capture_workload_rows':len(inverse_rows),'patterns':summary,'new_model_fits':0,'replicates_per_cohort':BOOT}
    (out/'ANALYSIS_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for row in summary:
        print(row['line'],row['contrast'],round(row['delta_min'],2),round(row['delta_max'],2),row['ci_excludes_zero_pct'],[(r['review_pct'],r['model_AKI_selected']-r['reference_AKI_selected'],round(r['lower_95_pointwise'],2),round(r['upper_95_pointwise'],2)) for r in row['points_10_20_30']])

