from pathlib import Path
import sys,json
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'engine'))
from core import *
import pandas as pd
def build(line):
 sp=spec_for('preop');assert sp['num']==['Age','PreopCr','PreopBUN','PreopAlb','PreopHb'] and sp['cat']==['Sex']
 m=pd.read_csv(INPUT/'preop_metadata.csv',dtype={'group_hash':str}).set_index('excel_row',drop=False)
 expected={'geography':(2551,133),'temporal':(2024,139)}[line]
 assert (len(m),int(m.y.sum()))==expected and m[line+'_partition_v2'].eq('development').all()
 x=pd.read_csv(INPUT/'preop_preop_features.csv')
 assert set(x)=={'excel_row',*sp['num'],*sp['cat']} and set(x.excel_row)==set(m.index)
 jobs=[]
 for context in ['final']+['outer_'+str(i) for i in range(5)]:
  for method in METHODS:
   deps=[]
   for p in parts(line,'preop',context):
    tr=m.loc[p['train_excel_rows']];va=m.loc[p['assessment_excel_rows']]
    assert set(tr.group_hash).isdisjoint(va.group_hash)
    t=pair_task(line,'preop','preop',context,p,method)
    tid='impute_'+line+'_'+digest(t)[:20];deps.append(tid)
    jobs.append({'id':tid,'worker':'preprocess.py','task':t,'dest':str(cache_dir(t)),'depends':[],'line':line,'kind':'impute'})
   for family in FAMILIES:
    t={'kind':'coarse','line':line,'cohort':'preop','context':context,'method':method,'family':family}
    jobs.append({'id':'coarse_'+line+'_'+digest(t)[:20],'worker':'coarse.py','task':t,'dest':str(OUT/'coarse'/line/'preop'/context/method/family),'depends':deps,'line':line,'kind':'coarse'})
 assert len(jobs)==432
 atomic(OUT/'A_TASKS.json',jobs)
 atomic(OUT/'INPUT_BOUNDARY_QUALIFICATION.json',{'status':'PASS_six_input_development_only_all_context_group_boundaries','line':line,'n':len(m),'events':int(m.y.sum()),'tasks':len(jobs),'input_lock_sha256':sha(INPUT/'SOURCE_LOCK.json')})
if __name__=='__main__':build(sys.argv[1])
