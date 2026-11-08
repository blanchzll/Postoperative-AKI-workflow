#!/usr/bin/env python3
"""Source-bound reporting audit. Reuses accepted results; performs no statistical refits."""
from __future__ import annotations
import argparse, ast, csv, datetime as dt, hashlib, io, json, math, os, resource, struct, time, zipfile
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def read_json(path): return json.loads(Path(path).read_text())
def read_csv(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))
def write_json(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_csv(path,rows,columns=None):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns or list(rows[0]));w.writeheader();w.writerows(rows)
def npy_header_from_zip(path,name):
    with zipfile.ZipFile(path) as z, z.open(name) as f:
        if f.read(6)!=b'\x93NUMPY': raise ValueError('Not an NPY member')
        major,minor=f.read(2)
        n=struct.unpack('<H' if major==1 else '<I',f.read(2 if major==1 else 4))[0]
        return ast.literal_eval(f.read(n).decode('latin1'))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',type=Path,default=Path('/home/lei/codex-workspaces/aki-prediction-manuscript'));parser.add_argument('--audit-root',type=Path,default=Path(__file__).resolve().parents[1]);a=parser.parse_args()
    started=time.time();P=a.project_root;R=a.audit_root;S=P/'correction_runs/clinical_supplement_20261108_v1';W=P/'writing_runs/github_publication_20261108_v1/package'
    req=read_json(R/'REQUEST.json');proto=read_json(R/'PROTOCOL_LOCK.json')
    assert sha(R/'REQUEST.json')==proto['request_sha256']
    assert (proto['new_statistical_models'],proto['new_patient_level_bootstraps'],proto['new_parameterized_scenarios'])==(0,0,0)
    expected_accept='0fb3693e3e4b395f32d2d7d880af5f38a9a895855241b881171b08849588067e'
    assert sha(S/'ACCEPTANCE.json')==expected_accept
    acceptance=read_json(S/'ACCEPTANCE.json');manifest=read_json(S/'SOURCE_AND_OUTPUT_MANIFEST.json')
    failures=[]
    for path,expected in manifest['unchanged_owning_source_hashes'].items():
        if not Path(path).is_file() or sha(path)!=expected: failures.append(path)
    for entry in manifest['files']:
        p=S/entry['path']
        if not p.is_file() or sha(p)!=entry['sha256']: failures.append(str(p))
    assert not failures,failures
    assert sha(S/'SOURCE_AND_OUTPUT_MANIFEST.json')==acceptance['source_output_manifest_sha256']
    for key,path in [('scientific_review_sha256','SCIENTIFIC_REVIEW.json'),('question_matrix_sha256','QUESTION_EVIDENCE_ANALYSIS_MATRIX.csv'),('claim_matrix_sha256','SCIENCE_CONCLUSION_MATRIX.csv')]:
        assert sha(S/path)==acceptance[key]
    write_json(R/'audit/FROZEN_SOURCE_RECHECK.json',{'status':'PASS','prior_acceptance_sha256':expected_accept,'owning_inputs_unchanged':len(manifest['unchanged_owning_source_hashes']),'prior_outputs_unchanged':len(manifest['files']),'source_manifest_sha256':sha(S/'SOURCE_AND_OUTPUT_MANIFEST.json'),'failed_files':failures,'source_models_or_labels_modified':False})
    conversation=read_json(R/'inputs/WORKBUDDY_LATEST_CONVERSATION.json');index=read_json(R/'inputs/MESSAGE_INDEX.json')
    assert len(index)==len(conversation['messages'])==32
    for message,entry in zip(conversation['messages'],index):
        text='\n\n'.join(c['text'] for c in message['content'] if c.get('type')=='text' and isinstance(c.get('text'),str))
        assert text==(R/'inputs'/entry['file']).read_text()
        assert sha(R/'inputs'/entry['file'])==entry['sha256']
    assert sha(R/'inputs/WORKBUDDY_LATEST_CONVERSATION.json')==req['latest_review_source']['sha256']
    write_json(R/'audit/LATEST_REVIEW_EXTRACTION_VERIFICATION.json',{'status':'PASS','messages':32,'source_sha256':sha(R/'inputs/WORKBUDDY_LATEST_CONVERSATION.json'),'last_message_sha256':sha(R/'inputs/31_assistant.md'),'text_only_extraction':True,'tool_calls_not_executed_as_instructions':True,'raw_shared_tool_outputs_public_release':False})
    # Inspect the retained sampling matrix header, without reading patient identifiers.
    sample_headers=[]
    for context,n in [('geography_preop',1159),('geography_paired',1158),('temporal_preop',1683),('temporal_paired',1011)]:
        path=S/'private'/f'{context}_RESAMPLING.npz';h=npy_header_from_zip(path,'weights.npy');assert h['shape']==(2001,n)
        sample_headers.append({'context':context,'shape':list(h['shape']),'bootstrap_replicates_excluding_point_row':2000,'dtype':h['descr'],'sampling_file_sha256':sha(path)})
    patient_run=read_json(S/'audit/PATIENT_ANALYSIS_RUN.json');outcome_run=read_json(S/'audit/OBSERVATION_OUTCOME_RUN.json');independent=read_json(S/'audit/INDEPENDENT_NUMERIC_VERIFICATION.json')
    assert patient_run['B']==outcome_run['B']==2000
    assert all(x['replicates_rebuilt']==2000 for x in independent['contexts'])
    write_json(R/'audit/BOOTSTRAP_PROVENANCE_CORRECTION.json',{'status':'CORRECTED_REVIEW_STATEMENT','review_inaccuracy':'Latest WorkBuddy message says the clinical supplement used zero new patient-level bootstraps.','verified_fact':'The accepted clinical supplement used 2,000 patient-group draws per evaluation context; the observation/outcome stage also records B=2000. Alternative expanded-record computations rebuilt all 2,000 draws in each of four contexts.','current_audit_new_bootstraps':0,'retained_sample_headers':sample_headers,'patient_run_sha256':sha(S/'audit/PATIENT_ANALYSIS_RUN.json'),'outcome_run_sha256':sha(S/'audit/OBSERVATION_OUTCOME_RUN.json'),'independent_verification_sha256':sha(S/'audit/INDEPENDENT_NUMERIC_VERIFICATION.json'),'new_predictive_model_fits_in_clinical_supplement':0,'human_expert_panel':False})
    # Source table supplies every metric; no inference from numerical magnitude.
    source=W/'02_Code_and_aggregate_data/aggregate_data/tables/Table_S19_Stage_comparisons.csv';original=P/'writing_runs/probabilistic_integration_20261105_v1/package/02_Code_and_aggregate_data/aggregate_data/tables/Table_S19_Stage_comparisons.csv'
    assert sha(source)==sha(original)
    rows=read_csv(source);assert len(rows)==28
    etables=(W/'04_Editable_sources/ETABLES.md').read_text();fragment=etables.split('## eTable 4.',1)[1].split('## eTable 5.',1)[0]
    old=[line for line in fragment.splitlines() if line.startswith('| ') and not line.startswith('| Evaluation') and not line.startswith('| ---')]
    assert len(old)==28
    mapped=[];before=[]
    contexts={'geography_paired':'Geographic paired','temporal_paired':'Temporal paired'}
    for i,(s,line) in enumerate(zip(rows,old),1):
        cells=[x.strip() for x in line.strip().strip('|').split('|')]
        assert cells[0]==contexts[s['context']] and cells[1]==s['label']
        for idx,key in enumerate(['estimate','lower95','upper95'],2):
            assert format(float(s[key]),'.3g')==cells[idx],(i,key,s[key],cells[idx])
        metric=s['metric'];capacity='—';unit='Probability-scale difference'
        if metric=='AUROC': label='AUROC';unit='AUROC difference'
        elif metric=='Brier':label='Brier score'
        elif metric in ['capacity10_captured_per1000','capacity20_captured_per1000']:
            label='Captured events';capacity='10%' if '10_' in metric else '20%';unit='Events per 1,000 operations'
        else:raise ValueError(metric)
        row={'source_row':i,'evaluation':contexts[s['context']],'comparison':s['label'],'metric':label,'review_capacity':capacity,'unit':unit,'estimate':s['estimate'],'lower95':s['lower95'],'upper95':s['upper95'],'source_metric':metric,'source_model_comparison':s['model'],'source_context':s['context'],'valid_replicates':s['valid_replicates'],'source_csv_sha256':sha(source)}
        mapped.append(row);before.append({'source_row':i,'metric':metric,'old_numbers_match_source_rounding':True,'metric_missing_in_current_table':True,'wrong_uniform_20_percent_footnote':True})
    write_csv(R/'results/ETABLE4_SOURCE_MAPPED.csv',mapped)
    headers=['Evaluation','Comparison','Metric','Capacity','Unit','Difference','Lower 95% CI','Upper 95% CI']
    lines=['## eTable 4. Stage-system and same-family performance and event-capture comparisons','', '| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for s in mapped:lines.append('| '+' | '.join(str(s[k]) for k in ['evaluation','comparison','metric','review_capacity','unit','estimate','lower95','upper95'])+' |')
    lines+=['','Differences are end-of-operation minus preoperative values. AUROC and Brier score differences are dimensionless; positive AUROC and negative Brier differences favor the end-of-operation system. Captured-event differences are events per 1,000 operative records at the indicated 10% or 20% batch review capacity. These are separate performance estimands. Intervals are percentile 95% confidence intervals from 2,000 hospital-stratified patient-group bootstrap samples. Geographic paired evaluation contains 1,158 records/72 events; temporal paired evaluation contains 1,011/36. The temporal non-overlap comparison excludes previously overlapping records (738 records/29 events).','', 'Stage-system comparisons are Voting (preoperative) versus LR (end of operation) geographically, and SGBT versus Voting temporally. Same-family comparisons retain the named model family at both stages. LR denotes logistic regression; SGBT, stochastic gradient boosting. Source CSV model identifiers and source-defined precision are retained in ETABLE4_SOURCE_MAPPED.csv.','']
    (R/'docs/ETABLE4_REPLACEMENT.md').write_text('\n'.join(lines))
    write_json(R/'audit/ETABLE4_MAPPING_VERIFICATION.json',{'status':'PASS_UNIQUE_SOURCE_MAPPING_REPORTING_REPAIR_REQUIRED','current_package_table':'writing_runs/github_publication_20261108_v1/package/04_Editable_sources/ETABLES.md','source_csv':str(source),'source_csv_sha256':sha(source),'source_rows':28,'metrics':sorted(set(x['metric'] for x in rows)),'all_old_numeric_cells_match_explicit_source_rounding':True,'metric_and_capacity_were_missing':True,'incorrect_title_and_footnote':True,'new_table_numeric_values_changed':False,'mapped_csv_sha256':sha(R/'results/ETABLE4_SOURCE_MAPPED.csv'),'replacement_md_sha256':sha(R/'docs/ETABLE4_REPLACEMENT.md'),'original_package_modified':False,'row_checks':before})
    checks=[]
    def check(label,left,right,source_ref,tol=2e-9):
        assert math.isclose(float(left),float(right),abs_tol=tol,rel_tol=tol),(label,left,right)
        checks.append({'check':label,'left':left,'right':right,'tolerance':tol,'status':'PASS','source':source_ref})
    for name in ['LOW_RISK_EVENTS.csv','MIMIC_LOW_RISK_EVENTS.csv','MIMIC_DELETION_SENSITIVITY.csv','FIXED_TRIGGER_COMPARISONS.csv','HARD_OUTCOME_DISCRIMINATION.csv']:
        for i,row in enumerate(read_csv(S/'results'/name),1):
            ref=f'clinical_supplement_20261108_v1/results/{name}:row{i}'
            if name=='LOW_RISK_EVENTS.csv':
                check(f'{name}:{i}:rate',row['observed_low_risk_event_rate'],int(row['low_risk_events'])/int(row['low_risk_n']),ref)
                check(f'{name}:{i}:missed_fraction',row['events_missed_fraction'],int(row['low_risk_events'])/int(row['events_total']),ref)
            elif name=='MIMIC_LOW_RISK_EVENTS.csv' and int(row['low_risk_n']):
                check(f'{name}:{i}:rate',row['low_risk_rate'],int(row['low_risk_events'])/int(row['low_risk_n']),ref)
            elif name=='MIMIC_DELETION_SENSITIVITY.csv':
                if int(row['N'])==0:continue
                check(f'{name}:{i}:event_accounting',row['original_known_events'],int(row['retained_known_events'])+int(row['lost_known_events_lower_bound']),ref)
                check(f'{name}:{i}:test_accounting',row['original_measurements'],int(row['remaining_measurements'])+int(row['deleted_measurements']),ref)
                if int(row['original_known_events']):check(f'{name}:{i}:lost_fraction',row['lost_known_fraction'],int(row['lost_known_events_lower_bound'])/int(row['original_known_events']),ref)
                check(f'{name}:{i}:deleted_fraction',row['deleted_fraction'],int(row['deleted_measurements'])/int(row['original_measurements']),ref)
            elif name=='FIXED_TRIGGER_COMPARISONS.csv':
                check(f'{name}:{i}:event_accounting',row['events'],int(row['captured_events'])+int(row['missed_events']),ref)
                check(f'{name}:{i}:capture_per1000',row['capture_per1000'],1000*int(row['captured_events'])/int(row['N']),ref)
            elif name=='HARD_OUTCOME_DISCRIMINATION.csv':check(f'{name}:{i}:event_rate',row['event_rate'],int(row['events'])/int(row['N']),ref)
    brier=read_csv(S/'results/MIMIC_BRIER_REVERIFIED.csv');p=59/759
    for row in brier:
        if row['reference']=='target_rate_reestimated' and row['metric']=='null_Brier':check('MIMIC:'+row['line']+':'+row['probability']+':null_Brier',row['estimate'],p*(1-p),'MIMIC_BRIER_REVERIFIED.csv')
    kdigo=read_json(S/'audit/MIMIC_KDIGO_CLOCK_AUDIT.json')
    check('Timed_component_overlap',kdigo['strict_measured_scr_component'],kdigo['strict_and_calendar']+kdigo['strict_outside_calendar'],'audit/MIMIC_KDIGO_CLOCK_AUDIT.json')
    check('Frozen_calendar_overlap',kdigo['frozen_calendar_events'],kdigo['strict_and_calendar']+kdigo['calendar_without_strict'],'audit/MIMIC_KDIGO_CLOCK_AUDIT.json')
    write_csv(R/'results/AUDIT_NUMERIC_CHECKS.csv',checks)
    write_json(R/'audit/AUDIT_RUN.json',{'status':'PASS_SOURCE_NUMERIC_AND_MAPPING_AUDIT','utc':dt.datetime.now(dt.timezone.utc).isoformat(),'seconds':time.time()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'source_input_hash_checks':len(manifest['unchanged_owning_source_hashes']),'prior_output_hash_checks':len(manifest['files']),'message_checks':32,'eTable4_rows':28,'aggregate_numeric_checks':len(checks),'new_model_fits':0,'new_bootstrap_replicates':0,'new_simulation_scenarios':0,'new_patient_data':0,'manuscript_edits_by_this_script':0,'script_sha256':sha(__file__),'protocol_sha256':sha(R/'PROTOCOL_LOCK.json'),'request_sha256':sha(R/'REQUEST.json')})
    print(json.dumps(read_json(R/'audit/AUDIT_RUN.json'),indent=2))
if __name__=='__main__':main()
