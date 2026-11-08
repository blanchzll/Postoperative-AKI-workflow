"""Bounded patient-level eligibility audit; never fabricates surgery clocks or model scores."""
from pathlib import Path
import sys,os,json,time,hashlib,gzip,re,traceback,urllib.request,collections
import numpy as np,pandas as pd
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');OUT=ROOT/'outputs/public_patient_qualification_20261024_v1'
M=Path('/home/lei/physionet.org/files/mimic/hosp');E=Path('/home/lei/physionet.org/files/eicu/eicu-collaborative-research-database-2.0')
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 q=Path(p).with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def status(dataset,phase,**kw):save(OUT/(dataset+'_status.json'),{'phase':phase,'pid':os.getpid(),**kw})
def capture_doc(url,name):
 dest=OUT/'source_documents';dest.mkdir(exist_ok=True)
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'AKI research eligibility audit'}),timeout=60) as r:data=r.read()
  p=dest/name;p.write_bytes(data)
  return {'url':url,'file':str(p),'sha256':sha(p),'status':'retrieved'}
 except Exception as ex:return {'url':url,'status':'retrieval_failed','error':type(ex).__name__+': '+str(ex)}
def mimic():
 sources={str(M/f):{'bytes':(M/f).stat().st_size,'mtime_ns':(M/f).stat().st_mtime_ns} for f in ['d_icd_procedures.csv.gz','procedures_icd.csv.gz','diagnoses_icd.csv.gz','patients.csv.gz','d_labitems.csv.gz','labevents.csv.gz']}
 d=pd.read_csv(M/'d_icd_procedures.csv.gz',dtype={'icd_code':str})
 title=d.long_title.fillna('')
 stomach=title.str.contains(r'^(?:Excision|Resection) of Stomach,',case=False,regex=True)
 colon=title.str.contains(r'^(?:Excision|Resection) of (?:Large Intestine|Right Large Intestine|Left Large Intestine|Cecum|Ascending Colon|Transverse Colon|Descending Colon|Sigmoid Colon|Rectum|Rectosigmoid)',case=False,regex=True)
 approach=title.str.contains(r'(?:Open Approach|Percutaneous Endoscopic Approach)$',case=False,regex=True)
 dic={}
 for row in d.itertuples():
  code=str(row.icd_code).replace('.','')
  site=None
  if row.icd_version==9:
   if any(code.startswith(k) for k in ['435','436','437','438','439']):site='gastric'
   elif any(code.startswith(k) for k in ['457','458','484','485','486']):site='colorectal'
  else:
   i=row.Index
   if approach.loc[i] and stomach.loc[i]:site='gastric'
   elif approach.loc[i] and colon.loc[i]:site='colorectal'
  if site:dic[(str(row.icd_code),int(row.icd_version))]=site
 selected=d[[ (str(c),int(v)) in dic for c,v in zip(d.icd_code,d.icd_version)]].copy()
 selected['site']=[dic[(str(c),int(v))] for c,v in zip(selected.icd_code,selected.icd_version)]
 selected.to_csv(OUT/'MIMIC_PROCEDURE_CODEBOOK.csv',index=False)
 pp=[]
 for ch in pd.read_csv(M/'procedures_icd.csv.gz',dtype={'icd_code':str},chunksize=200000):
  site=[dic.get((str(c),int(v))) for c,v in zip(ch.icd_code,ch.icd_version)]
  ch['site']=site;q=ch[ch.site.notna()]
  if len(q):pp.append(q)
 p=pd.concat(pp,ignore_index=True);assert p.chartdate.notna().all()
 hadms=set(p.hadm_id);diagnoses={}
 for ch in pd.read_csv(M/'diagnoses_icd.csv.gz',usecols=['hadm_id','icd_code','icd_version'],dtype={'icd_code':str},chunksize=250000):
  ch=ch[ch.hadm_id.isin(hadms)]
  for r in ch.itertuples(index=False):
   c=str(r.icd_code).replace('.','').upper();site=None
   if int(r.icd_version)==9:
    if c.startswith('151'):site='gastric'
    elif c.startswith('153') or c.startswith('1540') or c.startswith('1541'):site='colorectal'
   else:
    if c.startswith('C16'):site='gastric'
    elif c.startswith(('C18','C19','C20')):site='colorectal'
   if site:diagnoses.setdefault(int(r.hadm_id),set()).add(site)
 p=p[[s in diagnoses.get(int(h),set()) for h,s in zip(p.hadm_id,p.site)]].copy()
 people=pd.read_csv(M/'patients.csv.gz',usecols=['subject_id','gender','anchor_age','anchor_year'])
 p=p.merge(people,on='subject_id',validate='many_to_one');p['operation_date']=pd.to_datetime(p.chartdate)
 p['Age']=p.anchor_age+p.operation_date.dt.year-p.anchor_year
 p=p[p.Age>=18].sort_values(['subject_id','operation_date','seq_num'],kind='stable').drop_duplicates('subject_id',keep='first').reset_index(drop=True)
 p.to_csv(OUT/'PRIVATE_MIMIC_FIRST_TARGET_OPERATIONS.csv',index=False)
 status('MIMIC','candidate_cohort_ready',n=len(p),major_procedure_rows=len(selected),anatomically_concordant_admissions=int(p.hadm_id.nunique()))
 labs=pd.read_csv(M/'d_labitems.csv.gz')
 target={'creatinine':'PreopCr','urea nitrogen':'PreopBUN','albumin':'PreopAlb','hemoglobin':'PreopHb'}
 labs=labs[(labs.fluid.str.lower()=='blood')&labs.label.str.lower().isin(target)].copy()
 labs['feature']=labs.label.str.lower().map(target);labs.to_csv(OUT/'MIMIC_LAB_CODEBOOK.csv',index=False)
 item=dict(zip(labs.itemid,labs.feature));subjects=set(p.subject_id);dates=p.set_index('subject_id').operation_date
 collected=[];seen=0
 for ch in pd.read_csv(M/'labevents.csv.gz',usecols=['subject_id','itemid','charttime','valuenum','valueuom'],chunksize=250000):
  seen+=len(ch);ch=ch[ch.subject_id.isin(subjects)&ch.itemid.isin(item)]
  if len(ch):
   ch=ch.copy();ch['feature']=ch.itemid.map(item);ch['time']=pd.to_datetime(ch.charttime,errors='coerce');ch['operation_date']=ch.subject_id.map(dates)
   keep=(ch.time>=ch.operation_date-pd.Timedelta(days=30))&(ch.time<ch.operation_date+pd.Timedelta(days=8))
   ch=ch[keep]
   if len(ch):collected.append(ch)
  if seen%5000000==0:status('MIMIC','streaming_candidate_window_labs',n=len(p),rows_scanned=seen,windowed_labs=sum(len(q) for q in collected))
 z=pd.concat(collected,ignore_index=True) if collected else pd.DataFrame()
 z.to_csv(OUT/'PRIVATE_MIMIC_WINDOWED_LABS.csv.gz',index=False,compression='gzip')
 coverage=[];units=z.groupby(['feature','valueuom'],dropna=False).size().rename('measurements').reset_index()
 units.to_csv(OUT/'MIMIC_OBSERVED_UNITS.csv',index=False)
 for r in p.itertuples():
  q=z[z.subject_id==r.subject_id];numeric=pd.to_numeric(q.valuenum,errors='coerce')
  q=q[numeric.notna()&(numeric>0)]
  rec={'subject_id':int(r.subject_id),'Age':float(r.Age),'Sex_available':r.gender in ['M','F'],'site':r.site}
  before=q[q.time<r.operation_date];after=q[(q.time>=r.operation_date+pd.Timedelta(days=1))&(q.time<r.operation_date+pd.Timedelta(days=8))]
  for feature in target.values():rec[feature+'_before_day0']=bool((before.feature==feature).any())
  rec['postoperative_calendar_day1_7_SCr']=bool((after.feature=='PreopCr').any())
  rec['day0_SCr_ambiguous']=bool(((q.time>=r.operation_date)&(q.time<r.operation_date+pd.Timedelta(days=1))&(q.feature=='PreopCr')).any())
  coverage.append(rec)
 c=pd.DataFrame(coverage);c.to_csv(OUT/'PRIVATE_MIMIC_MEASUREMENT_COVERAGE.csv',index=False)
 names=[k for k in c if k.startswith('Preop') or k in ['Sex_available','postoperative_calendar_day1_7_SCr','day0_SCr_ambiguous']]
 stats={k:int(c[k].sum()) for k in names}
 stats['baseline_and_later_SCr']=int((c.PreopCr_before_day0&c.postoperative_calendar_day1_7_SCr).sum())
 stats['all_six_observed_and_later_SCr']=int(c[['Sex_available']+[f+'_before_day0' for f in target.values()]+['postoperative_calendar_day1_7_SCr']].all(axis=1).sum())
 for name,r in sources.items():
  f=Path(name);assert f.stat().st_size==r['bytes'] and f.stat().st_mtime_ns==r['mtime_ns'];r['sha256']=sha(f)
 save(OUT/'MIMIC_PATIENT_AUDIT.json',{'status':'PASS_patient_level_candidate_and_measurement_audit','n_first_adult_target_operations':len(p),'coverage':stats,
  'source_files':sources,'procedure_date_precision':'calendar_day_only','exact_end_of_surgery_available_in_examined_tables':False,
  'conclusion_original9_27_38_and_selective':'Not qualified as faithful original end-of-operation or selective-strategy evaluation',
  'conclusion_common6':'Preoperative common6 can be assembled conservatively before operation day; exact original7day timing and onset coverage remain nonidentical; a separately named calendar-window support evaluation would require its own endpoint lock',
  'model_scores_generated':False,'AKI_event_count_not_claimed_without_qualified_endpoint':True})
 status('MIMIC','patient_level_audit_complete',n=len(p),coverage=stats)
def eicu():
 docs=[capture_doc('https://eicu.mit.edu/eicutables/treatment/','eicu_treatment.html'),capture_doc('https://eicu.mit.edu/eicutables/diagnosis/','eicu_diagnosis.html')]
 p=pd.read_csv(E/'patient.csv.gz',usecols=['patientunitstayid','patienthealthsystemstayid','gender','age','hospitalid','apacheadmissiondx','unitadmitsource','unitvisitnumber','uniquepid'])
 ages=pd.to_numeric(p.age.astype(str).str.replace('>','',regex=False),errors='coerce');p=p[ages>=18];adults=set(p.patientunitstayid)
 cancer=set();diagnosis_labels=collections.Counter()
 for ch in pd.read_csv(E/'diagnosis.csv.gz',usecols=['patientunitstayid','diagnosisstring','icd9code','diagnosisoffset'],chunksize=250000):
  ch=ch[ch.patientunitstayid.isin(adults)];ss=ch.diagnosisstring.fillna('').str.lower();cc=ch.icd9code.fillna('').str.replace('.','',regex=False).str.upper()
  code=cc.str.contains(r'(?<!\d)(?:151\d*|153\d*|154[01]\d*)(?!\d)|C(?:16|18|19|20)\d*',regex=True)
  site=ss.str.contains(r'stomach|gastric|colon|colorectal|rectal|rectum',regex=True);mal=ss.str.contains(r'cancer|carcinoma|malignan|neoplasm',regex=True)
  q=ch[code|(site&mal)]
  cancer.update(q.patientunitstayid);diagnosis_labels.update(q.diagnosisstring)
 surgery=set();treatments=[];all_resection_stays=set()
 pattern=r'gastrectom|colectom|hemicolectom|proctectom|(?:gastric|stomach|colon|colorectal|rectal|rectum).{0,30}resection|abdominoperineal|low anterior resection'
 for ch in pd.read_csv(E/'treatment.csv.gz',usecols=['patientunitstayid','treatmentstring','treatmentoffset'],chunksize=250000):
  q=ch[ch.treatmentstring.fillna('').str.contains(pattern,case=False,regex=True)]
  all_resection_stays.update(q.patientunitstayid)
  q=q[q.patientunitstayid.isin(cancer)]
  if len(q):treatments.append(q);surgery.update(q.patientunitstayid)
 t=pd.concat(treatments,ignore_index=True) if treatments else pd.DataFrame(columns=['patientunitstayid','treatmentstring','treatmentoffset'])
 candidates=p[p.patientunitstayid.isin(surgery)].copy();candidates.to_csv(OUT/'PRIVATE_EICU_CANDIDATE_STAYS.csv',index=False);t.to_csv(OUT/'PRIVATE_EICU_RESECTION_DOCUMENTATION.csv',index=False)
 pd.DataFrame(diagnosis_labels.most_common(),columns=['diagnosis_string','records']).to_csv(OUT/'EICU_TARGET_DIAGNOSIS_CODEBOOK.csv',index=False)
 t.groupby('treatmentstring',dropna=False).size().rename('records').reset_index().to_csv(OUT/'EICU_RESECTION_TERMS.csv',index=False)
 status('eICU','candidate_documentation_ready',adult_cancer_stays=len(cancer),cancer_with_resection_documentation=len(candidates))
 items={'creatinine':'PreopCr','bun':'PreopBUN','albumin':'PreopAlb','hgb':'PreopHb','hemoglobin':'PreopHb'}
 measurements=[];seen=0
 if surgery:
  for ch in pd.read_csv(E/'lab.csv.gz',usecols=['patientunitstayid','labresultoffset','labname','labresult','labmeasurenamesystem'],chunksize=250000):
   seen+=len(ch);ch=ch[ch.patientunitstayid.isin(surgery)&ch.labname.fillna('').str.lower().isin(items)]
   if len(ch):measurements.append(ch)
   if seen%5000000==0:status('eICU','streaming_candidate_labs',rows_scanned=seen,matched_records=sum(len(q) for q in measurements))
 z=pd.concat(measurements,ignore_index=True) if measurements else pd.DataFrame(columns=['patientunitstayid','labname','labresultoffset'])
 z.to_csv(OUT/'PRIVATE_EICU_CANDIDATE_LABS.csv.gz',index=False,compression='gzip')
 coverage={feature:int(z.loc[z.labname.astype(str).str.lower().map(items)==feature,'patientunitstayid'].nunique()) for feature in ['PreopCr','PreopBUN','PreopAlb','PreopHb']}
 save(OUT/'EICU_PATIENT_AUDIT.json',{'status':'PASS_patient_level_documentation_and_measurement_audit','adult_stays':len(p),'adult_target_cancer_stays':len(cancer),
  'all_resection_documentation_stays':len(all_resection_stays),'cancer_and_resection_candidate_stays':len(candidates),
  'candidate_people':int(candidates.uniquepid.nunique()),'candidate_hospitals':int(candidates.hospitalid.nunique()),'any_ICU_offset_measurement_coverage':coverage,
  'documentation_sources':docs,'source_files':{str(E/f):{'bytes':(E/f).stat().st_size,'sha256':sha(E/f)} for f in ['patient.csv.gz','diagnosis.csv.gz','treatment.csv.gz','lab.csv.gz']},
  'operation_start_or_end_event_time_validated':False,
  'conclusion':'Not qualified for current preoperative/end-of-operation target: diagnosis and treatment offsets indicate record entry, not surgery clocks; ICU-offset labs cannot be presumed preoperative',
  'absence_of_treatment_not_interpreted_as_absence':True,'AKI_labels_or_model_scores_generated':False})
 status('eICU','patient_level_audit_complete',candidate_stays=len(candidates),any_ICU_offset_measurement_coverage=coverage)
if __name__=='__main__':
 try:
  if sys.argv[1]=='MIMIC':mimic()
  else:eicu()
 except Exception:save(OUT/(sys.argv[1]+'_FAILURE.json'),{'traceback':traceback.format_exc(),'no_unchanged_retry':True});raise
