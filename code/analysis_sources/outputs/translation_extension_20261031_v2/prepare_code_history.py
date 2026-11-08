"""Explicit user-confirmed code-zero public history proxy; never a clinically adjudicated negative."""
from pathlib import Path
import csv,gzip,json,os,hashlib,datetime,fcntl,statistics,resource
from zoneinfo import ZoneInfo
R=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');O=R/'outputs';T=O/'translation_extension_20261031_v2';S=Path('/home/lei/physionet.org/files/inspire/1.4.2')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);assert not p.exists(),str(p);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 T.mkdir(mode=0o700,exist_ok=True);guard=(T/'RUNNER.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (T/'HISTORY_PROXY_INPUT_ACCEPTANCE.json').exists()
 now=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
 # Category lists fixed before new diagnosis count construction or model scoring.
 card={'I01'}|{f'I{i:02}' for i in range(5,10)}|{f'I{i:02}' for i in range(20,53)}|{f'Q{i:02}' for i in range(20,25)}
 codes={'Diabetes':set('E10 E11 E12 E13 E14'.split()),'Hypertension':set('I10 I11 I12 I13 I15'.split()),'HeartDisease':card}
 references=[
 {'url':'https://physionet.org/content/inspire/1.4/','read':'Methods anonymization and Diagnosis sections','use':'Released diagnoses are truncated and specified diagnoses removed. Local1.4.2 changelog/schema/exclusion list are exact version evidence; v1.4 page is supporting documentation, not a claim of newer downloaded data.'},
 {'url':'https://www.cdc.gov/heart-disease-stroke-atlas/data-sources/index.html','read':'ICD-code groups for heart disease and hypertension','use':'Supports distinction of cardiac and hypertensive groups. Not an exact transplant of mortality/principal-diagnosis epidemiological phenotype.'},
 {'url':'https://www.cms.gov/icd10m/fy2025-version42.1-fullcode-cms/fullcode_cms/P0152.html','read':'Cardiac congenital and valvular diagnoses','use':'Supports I01/I05-I09/valvular/congenital cardiac category identity. Detailed codes cannot be reconstructed beyond three-character released categories.'}]
 models=[]
 for line,fam in [('geography','LR'),('temporal','Voting')]:
  p=O/'model_lock_20261022_v1/locked_routes'/line/'preop/preop/compact/champion'/fam/'MODEL_LOCK.json';m=json.loads(p.read_text());assert len(m['input_columns'])==9 and m['actual_input_count']==9
  models.append({'line':line,'model_lock':str(p),'model_lock_sha256':sha(p),'identity':m['identity'],'components':{v['path']:v['sha256'] for v in list(m['base_components'].values())+[m['calibrator']]}})
 sources=[S/'diagnosis.csv.gz',S/'icd10_excluded.csv',S/'schema.csv',S/'CHANGELOG.txt',O/'public_eligibility_refinement_20261022_v1/PRIVATE_COMMON6_FEATURES.csv',O/'public_eligibility_refinement_20261022_v1/PRIVATE_COMMON6_METADATA.csv',O/'public_eligibility_refinement_20261022_v1/PRIVATE_FIRST_OPERATION_PROFILE.csv',O/'public_eligibility_refinement_20261022_v1/COMMON6_INPUT_CONTRACT.json',O/'translation_extension_20261031_v1/ACCEPTANCE.json',Path(__file__)]
 prelock={'status':'LOCKED_BEFORE_CODE_PROXY_CONSTRUCTION_AND_ANY_NINE_INPUT_PREDICTION','time_Asia_Shanghai':now,'analysis_thread_id':'01a170c0-3584-74e2-b44c-4d0711ee1860','user_clarifications':[{'text':'HeartDisease 实际记录的是“高血压以外的心脏病”','supersedes':'Earlier broad cardiovascular definition including hypertension.'},{'text':'没有诊断编码就是无病史','implementation':'0 means no released matching diagnostic code strictly before operation start; operational coded-history rule, not clinical verification of absence.'}],'coded_history_rule':{'clock':'chart_time < opstart_time, all available prior observations for same subject; never future backfill','positive':'At least one released category in fixed list before prediction time','zero':'No released matching category before prediction time, including redacted/unrecorded histories; disclosed proxy assumption','category_lists':{k:sorted(v) for k,v in codes.items()},'hypertension_separate':'Do not set HeartDisease1 from I10-I15 alone, including I11/I13 in this expressly category-based sensitivity. This is a proxy classification decision, not clinical equivalence.'},'redaction_limit':'Local exclusion file removes I27,I42,I47,I49 andQ family among others; zeros are not proof of no clinical disease. No redacted codes imputed or reconstructed.','scope':'Nine-input code-history proxy sensitivity using frozen China9 champion pipelines. Not full semantic qualification/clinical validation of nine disease-history measurements.','cohort_n':6043,'events':253,'source_models':models,'source_files_sha256':{str(p):sha(p) for p in sources},'old_source_data_models_and_results_unchanged':True,'no_fits_search_target_recalibration_new_patients':True,'comparison':'Within same6043 patients, each nine-input source model versus its corresponding original common6RF workflow, separately raw/cal. Different family/preprocessing; not pure three-history increment.','bootstrap':{'patient_groups':True,'replicates':2000,'seed':20261017,'conditional_fixed_models':True},'source_thresholds_unchanged':True,'clinical_loss_margin_delta7_unchanged':True,'references':references,'research_API_calls':0,'no_remote_patient_data_sent':True}
 save(T/'CODE_HISTORY_AND_MODEL_PRELOCK.json',prelock)
 meta=list(csv.DictReader((O/'public_eligibility_refinement_20261022_v1/PRIVATE_COMMON6_METADATA.csv').open()));profile={x['op_id']:x for x in csv.DictReader((O/'public_eligibility_refinement_20261022_v1/PRIVATE_FIRST_OPERATION_PROFILE.csv').open())}
 ops={m['subject_id']:profile[m['op_id']] for m in meta};assert len(ops)==6043
 states={pid:{k:0 for k in codes} for pid in ops};flags={pid:{k:False for k in codes} for pid in ops};matched_codes={k:{} for k in codes};rows=0
 with gzip.open(S/'diagnosis.csv.gz','rt') as f:
  for z in csv.DictReader(f):
   rows+=1;pid=z['subject_id']
   if pid not in ops:continue
   try:at=float(z['chart_time']);start=float(ops[pid]['opstart_time'])
   except ValueError:continue
   if not at<start:continue
   code=z['icd10_cm'].strip().upper()
   for key,valid in codes.items():
    if code in valid:
     states[pid][key]=1;flags[pid][key]=True
     matched_codes[key][code]=matched_codes[key].get(code,0)+1
 features=list(csv.DictReader((O/'public_eligibility_refinement_20261022_v1/PRIVATE_COMMON6_FEATURES.csv').open()));bykey={m['record_key']:m for m in meta}
 assert len(features)==6043
 with (T/'PRIVATE_CODED_NINE_FEATURES.csv').open('w') as f:
  fields=list(features[0])+list(codes);w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for z in features:
   m=bykey[z['record_key']];z.update({k:str(states[m['subject_id']][k]) for k in codes});w.writerow(z)
 # Keep source patient IDs and code audit only in private table.
 with (T/'PRIVATE_HISTORY_FLAGS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['record_key','subject_id','op_id',*codes]);w.writeheader()
  for m in meta:w.writerow({**{k:m[k] for k in ['record_key','subject_id','op_id']},**states[m['subject_id']]})
 counts={k:{'one':sum(v[k] for v in states.values()),'zero':sum(1-v[k] for v in states.values())} for k in codes}
 save(T/'HISTORY_MAPPING_AUDIT.json',{'status':'PASS_EXPLICIT_CODE_ZERO_PROXY_INPUTS_NOT_CLINICAL_NEGATIVE_VERIFICATION','n':6043,'events':253,'diagnosis_rows_scanned':rows,'counts':counts,'preop_matching_code_record_counts':matched_codes,'excluded_file_text':(S/'icd10_excluded.csv').read_text(encoding='utf-8-sig'),'all_features_cohort_unchanged':True,'new_nonzero_cardiac_codes_beyond_IHD_HF_retained':True,'source_HeartDisease_and_hypertension_are_separate':'Current China data3710: hypertension1=1148, HeartDisease1=313, hypertension1/HeartDisease0=974; consistent with latest direct user definition, not a reason to recode China.'})
 save(T/'HISTORY_PROXY_INPUT_ACCEPTANCE.json',{'status':'PASS_EXPLICIT_CODE_HISTORY_PROXY_ONLY','prelock_sha256':sha(T/'CODE_HISTORY_AND_MODEL_PRELOCK.json'),'n':6043,'events':253,'counts':counts,'files':{n:sha(T/n) for n in ['PRIVATE_CODED_NINE_FEATURES.csv','PRIVATE_HISTORY_FLAGS.csv','HISTORY_MAPPING_AUDIT.json']},'full_nine_input_semantic_qualification':False,'nine_code_proxy_sensitivity_allowed':True,'no_future_codes_no_case_exclusion_no_outcome_selection':True,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
 print(json.dumps({'n':6043,'events':253,'counts':counts,'qualification':'CODE_PROXY_SENSITIVITY_ONLY'},ensure_ascii=False),flush=True)
 guard.close()
if __name__=='__main__':main()
