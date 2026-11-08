"""Read-only adapter for accepted predictions; does not refit or score raw patient inputs."""
from pathlib import Path
import sys
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
sys.path.insert(0,str(ROOT/'work/formal_evaluation_20261023_v1'))
import common as base
import numpy as np,pandas as pd,joblib
from decision_math import actions,route
OUT=ROOT/'outputs/decision_workload_20261024_v1'
def load(line,cohort):
    a,b=base.gate()
    final=base.read(base.OUT/'STAGE_FINAL_EVALUATION_ACCEPTANCE.json')
    assert final['status']=='PASS_four_heldout_partitions_fixed_models_and_locked_policies'
    dest=base.OUT/(line+'_'+cohort);receipt=base.read(dest/'COMPLETE.json')
    for fn,h in receipt['files'].items():assert base.sha(dest/fn)==h
    meta=None;pred={};thresholds={};eligible={};lookup={};primary={};locks={};refs={}
    for ref in a['models']:
        lock=base.read(ref['path']);i=lock['identity']
        if (i['line'],i['cohort'])!=(line,cohort):continue
        nm='_'.join(i[k] for k in ['stage','branch','role','family'])
        folder=dest/'private_predictions'/nm;rec=base.read(folder/'COMPLETE.json')
        assert rec['signature']['model_lock']==ref['sha256'] and base.sha(folder/'PREDICTIONS.csv')==rec['sha256']
        d=pd.read_csv(folder/'PREDICTIONS.csv',dtype={'group_hash':str},float_precision='round_trip')
        m=d.drop(columns=['prob_raw','prob_calibrated'])
        if meta is None:meta=m
        else:assert meta.equals(m)
        pred[nm]=d.prob_calibrated.to_numpy();thresholds[nm]=lock['training_thresholds']
        eligible[nm]=(i['stage']=='preop');lookup[ref['path']]=nm;locks[nm]=lock
        refs[str(folder/'PREDICTIONS.csv')]=rec['sha256']
        if ref['primary_route_winner']:primary[(i['stage'],i['branch'])]=nm
    assert (len(meta),int(meta.y.sum()))==base.EXPECTED[(line,cohort)]
    pol={}
    for ref in b['policy_manifests']:
        lock=base.read(ref['path']);i=lock['identity']
        if (i['line'],i['cohort'])!=(line,cohort):continue
        nm=f"policy_{i['upgrade_fraction']}_{i['mode']}_{i['kind']}"
        assert base.sha(lock['calibrator']['path'])==lock['calibrator']['sha256']
        pol[nm]={'lock':lock,'base':lookup[lock['base_model']['path']],'upgrade':lookup[lock['upgrade_model']['path']],'calibrator':joblib.load(lock['calibrator']['path'])}
        thresholds[nm]=lock['training_thresholds'];eligible[nm]=(cohort=='preop')
    base_nm=primary[('preop','compact' if cohort=='preop' else 'full')]
    up_nm=primary[('preop' if cohort=='preop' else 'endop','full')]
    comps=[(up_nm,base_nm)]
    if cohort=='paired':
        for fam in {locks[base_nm]['family'],locks[up_nm]['family']}:
            pre=[nm for nm,l in locks.items() if l['identity']['role']=='reference' and l['identity']['stage']=='preop' and l['family']==fam]
            end=[nm for nm,l in locks.items() if l['identity']['role']=='reference' and l['identity']['stage']=='endop' and l['family']==fam]
            assert len(pre)==len(end)==1;comps.append((end[0],pre[0]))
    for nm,it in pol.items():
        comps.extend([(nm,base_nm),(nm,up_nm)])
        i=it['lock']['identity']
        if i['mode']=='capacity' and i['kind']=='selective':comps.append((nm,f"policy_{i['upgrade_fraction']}_capacity_random"))
    return meta,pred,thresholds,pol,eligible,list(dict.fromkeys(comps)),refs
def qualify():
    proof=[]
    for line in ['geography','temporal']:
        for cohort in ['preop','paired']:
            m,pr,th,pol,eligible,comp,refs=load(line,cohort)
            y=m.y.to_numpy();g=m.group_hash.astype(str).tolist();r=m.excel_row.astype(str).tolist()
            allpr=dict(pr)
            for nm,it in pol.items():_,allpr[nm],_=route(it,pr[it['base']],pr[it['upgrade']],g,r)
            original=base.read(base.OUT/(line+'_'+cohort)/'EVALUATION.json')
            oldth=pd.read_csv(base.OUT/(line+'_'+cohort)/'LOCKED_THRESHOLD_BURDEN.csv',float_precision='round_trip')
            for nm,p in allpr.items():
                q=actions(y,p,g,r,th[nm])
                for fraction in [10,20]:
                    for field in ['reviewed_n','captured_n','missed_n','false_alerts_n']:
                        assert q['assessment_capacity'+str(fraction)][field]==original['point'][nm]['capacity'+str(fraction)+'_'+field]
                for rule in q:
                    if rule.startswith('training_'):
                        old=oldth.loc[(oldth.model==nm)&(oldth.rule==rule)]
                        assert len(old)==1
                        for field in ['reviewed_n','captured_n','missed_n','false_alerts_n']:assert q[rule][field]==old.iloc[0][field]
            if cohort=='paired':assert all(not eligible[nm] for nm in pol)
            proof.append({'partition':line+'_'+cohort,'models':len(pr),'policies':len(pol),'n':len(m),'events':int(y.sum()),'capacity_and_threshold_counts_match_accepted_results':True,'prediction_sources':refs})
    base.atomic(OUT/'SOURCE_ADAPTER_QUALIFICATION.json',{'status':'PASS_26_model_and_24_policy_frozen_predictions_replayed','proof':proof,'clinical_scenario_results_generated':False,'no_refitting_or_reselection':True})
if __name__=='__main__':qualify()
