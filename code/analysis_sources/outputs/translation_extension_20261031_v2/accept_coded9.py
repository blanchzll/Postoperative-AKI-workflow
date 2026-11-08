from pathlib import Path
import os,json,csv,hashlib,datetime,subprocess
from zoneinfo import ZoneInfo
R=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');O=R/'outputs';T=O/'translation_extension_20261031_v2';V1=O/'translation_extension_20261031_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);assert not p.exists(),str(p);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
assert not (T/'ACCEPTANCE.json').exists()
now=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat();pins=read(T/'CODE_HISTORY_AND_MODEL_PRELOCK.json');deps=dict(pins['source_files_sha256'])
for p,h in deps.items():assert sha(p)==h,p
models=read(O/'model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json')
for it in models['models']:
 assert sha(it['path'])==it['sha256'];deps[it['path']]=it['sha256']
 m=read(it['path'])
 for rec in list(m['base_components'].values())+[m['calibrator']]+([m['meta']] if m.get('meta') else []):assert sha(rec['path'])==rec['sha256'];deps[rec['path']]=rec['sha256']
for line in ['geography','temporal']:
 p=O/'common6_public_evaluation_20261024_v1'/line
 assert sha(p/'PRIVATE_PREDICTIONS.csv')==read(p/'PREDICTION_ACCEPTANCE.json')['prediction_sha256']
 source=p/'PRIVATE_PREDICTIONS.csv';deps[str(source)]=sha(source)
 a=T/line/'PREDICTION_ACCEPTANCE.json';assert sha(T/line/'PRIVATE_PREDICTIONS.csv')==read(a)['prediction_sha256']
for n,h in read(T/'COMPUTATION_ACCEPTANCE.json')['files'].items():assert sha(T/n)==h
for n,d in read(T/'FIGURE_RENDER_RECEIPT.json')['files'].items():assert sha(T/n)==d['sha256']
for p,d in read(O/'post_reanalysis_process_review_v1/MANUSCRIPT_PROTECTION.json')['files'].items():assert sha(p)==d['sha256']
sl=O/'v4_development_inputs_20261018/SOURCE_LOCK.json';assert sha(sl)=='afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3'
for name,h in read(sl)['input_sha256'].items():assert sha(sl.parent/name)==h
assert sha(V1/'ACCEPTANCE.json')=='0430bcec06a75797df508dbaba9d0d0370783a72e74927d0bd0f707a9b1d16c9'
for n,d in read(V1/'SOURCE_AND_ARTIFACT_MANIFEST.json')['new_artifacts'].items():assert sha(V1/n)==d['sha256'],n
for it in read(O/'final_analysis_delivery_20261025_v1/CURRENT_COMPLETE_FIGURE_INVENTORY.json')['current_images']:assert sha(it['path'])==it['sha256']
num=read(T/'INDEPENDENT_NUMERICAL_VERIFICATION.json');assert num['status'].startswith('PASS') and num['independent_patient_bootstrap_replicates']==2000
assert read(T/'SCIENTIFIC_REVIEW.json')['status'].startswith('PASS') and read(T/'VISUAL_REVIEW_ACCEPTANCE.json')['status'].startswith('PASS')
observed=[]
for unit in ['aki-public-coded9-sensitivity-20261031-v2.service','aki-public-coded9-independent-20261031-v2.service']:
 s=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','ActiveState','-p','ExecMainStatus','-p','Result','--no-pager'],capture_output=True,text=True)
 x=dict(z.split('=',1) for z in s.stdout.splitlines() if '=' in z);assert x.get('MainPID')=='0' and x.get('ActiveState')=='inactive' and x.get('ExecMainStatus')=='0',(unit,x);observed.append({'unit':unit,**x})
save(T/'OWNED_PROCESS_CLOSURE.json',{'status':'ALL_NEW_OWNED_UNITS_EXITED_ZERO','time_Asia_Shanghai':now,'units':observed,'plot_SSH_native_session_exited0':True,'configured_memory_max_GiB':3,'actual_prediction_bootstrap_peak_RSS_MiB':read(T/'COMPUTATION_ACCEPTANCE.json')['peak_RSS_MiB'],'actual_independent_check_peak_RSS_MiB':num['peak_RSS_MiB'],'GPU_or_other_jobs_or_global_services_changed':False,'no_exclusive_lease_or_heartbeat_reactivated':True})
pointer=O/'reanalysis_control/TRANSLATION_EXTENSION_CURRENT.json';oldpointer=read(pointer);assert oldpointer['acceptance_sha256']==sha(V1/'ACCEPTANCE.json')
save(T/'PREVIOUS_CURRENT_POINTER.json',oldpointer)
artifacts={}
for p in sorted(T.rglob('*')):
 if p.is_dir():os.chmod(p,0o700)
 elif p.is_file() and p.name!='RUNNER.lock':
  os.chmod(p,0o600);name=str(p.relative_to(T));artifacts[name]={'sha256':sha(p),'bytes':p.stat().st_size,'patient_derived_or_training_private':('PRIVATE' in p.name or p.suffix=='.npz'),'not_automatically_public':True}
save(T/'SOURCE_AND_ARTIFACT_MANIFEST.json',{'status':'PINNED_CURRENT_CODED_HISTORY_PUBLIC9_SUPPLEMENT','source_sha256':deps,'new_artifacts':artifacts,'original_source_models26_manuscripts92_and64_figures_verified_unchanged':True,'v1_all103_artifacts_unchanged':True,'six_existing_predictions_reused_not_rescored':True,'no_new_predictive_fits_or_target_calibration':True})
accept={'status':'PASS_FROZEN_NINE_INPUT_PUBLIC_CODED_HISTORY_SUPPLEMENT','time_Asia_Shanghai':now,'public_n':6043,'public_events':253,'public_nine_input_model_evaluation_performed':True,'latest_user_heart_definition_confirmed':True,'latest_user_no_code_zero_rule_implemented_before_scores':True,'source_China_recode_or_refit_required':False,'new_predictive_fits':0,'new_target_calibration_updates':0,'full_nine_input_clinical_measurement_equivalence_claimed':False,'known_provider_code_suppression_retained':True,'raw_and_source_calibrated_views':8,'paired_workflow_comparisons':4,'patient_bootstrap_repetitions':2000,'independent_all2000_key_metric_and_threshold_CI_check':True,'independent_numeric_assertions':num['numeric_assertions'],'max_abs_difference_including_diagnostic_optimizer':num['max_abs_difference_including_separate_calibration_optimizer'],'actual_new_figure_reviewed':1,'protected_manuscript_files_unchanged':92,'source_model_manifests_unchanged':26,'old64_figures_and_v1_results_unchanged':True,'scientific_reading_actual_same_Codex_not_external_expert':True,'v1_superseded_scope_statements':['HeartDisease definition unknown','No code-zero nine-input proxy evaluation allowed'],'v1_numeric_results_and_other_conclusions_retained':True,'six_input_public_evidence_remains_primary':True,'no_manuscript_prose_edited_or_reviewed':True,'no_new_thread_outbound_message_or_automation':True,'entry':'MANUSCRIPT_HANDOFF_ZH.md','files':{n:sha(T/n) for n in ['USER_CLARIFICATION_AND_SCOPE_AMENDMENT.json','CODE_HISTORY_AND_MODEL_PRELOCK.json','PRE_SCORE_LOCK.json','HISTORY_PROXY_INPUT_ACCEPTANCE.json','POINT_ESTIMATES.csv','INTERVALS.csv','PAIRED_DIFFERENCES_VS_SIX_RF.csv','NINE_SOURCE_THRESHOLD_BURDEN.csv','INDEPENDENT_NUMERICAL_VERIFICATION.json','VISUAL_REVIEW_ACCEPTANCE.json','SCIENTIFIC_REVIEW.json','CONCLUSION_UPDATE_MATRIX.csv','RESULTS_AND_SCOPE_ZH.md','MANUSCRIPT_HANDOFF_ZH.md','SOURCE_AND_ARTIFACT_MANIFEST.json','OWNED_PROCESS_CLOSURE.json']}}
save(T/'ACCEPTANCE.json',accept)
current={'status':accept['status'],'directory':str(T),'handoff':str(T/'MANUSCRIPT_HANDOFF_ZH.md'),'acceptance':str(T/'ACCEPTANCE.json'),'acceptance_sha256':sha(T/'ACCEPTANCE.json'),'time_Asia_Shanghai':now,'previous_version':{'directory':str(V1),'acceptance_sha256':sha(V1/'ACCEPTANCE.json'),'retained_numeric_evidence':True},'latest_definition':'HeartDisease excludes hypertension, which is separate; no presurgical released code0 by user rule; known suppression disclosed.','original37_handoff_not_replaced':True,'writing_thread_id':'01a18e2b-074e-7d03-b737-c5e5d6b32074','analysis_thread_id':'01a170c0-3584-74e2-b44c-4d0711ee1860','no_outbound_message_sent':True}
tmp=pointer.with_name(pointer.name+'.tmp');tmp.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,pointer)
for n,h in accept['files'].items():assert sha(T/n)==h
assert read(pointer)['acceptance_sha256']==sha(T/'ACCEPTANCE.json')
print(json.dumps({'status':accept['status'],'acceptance_sha256':sha(T/'ACCEPTANCE.json'),'directory':str(T),'new_fits':0,'originals_unchanged':True,'current_pointer_updated':True},ensure_ascii=False),flush=True)
