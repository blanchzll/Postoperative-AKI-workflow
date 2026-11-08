from pathlib import Path
import run as w
import numpy as np,pandas as pd
def main():
 w.check();c=w.candidates();proof=[]
 for line in ['geography','temporal']:
  for cohort in ['preop','paired']:
   old,pred,th,pol,eligible,comparisons,refs=w.adapter.load(line,cohort)
   before=old.copy(deep=True)
   for scenario in w.SCENARIOS:
    m,p,affected=w.transform(old,pred,set(c.excel_row),scenario)
    assert old.equals(before)
    count=int(affected.sum())
    assert len(m)==len(old)-(count if scenario.startswith('exclude') else 0)
    assert int(m.y.sum())==int(old.y.sum())-count
    ix=old.set_index('excel_row').index.get_indexer(m.excel_row)
    for name in p:np.testing.assert_array_equal(p[name],pred[name][ix])
    proof.append({'line':line,'cohort':cohort,'scenario':scenario,'n':len(m),'events':int(m.y.sum()),'affected_n':count,'frozen_predictions_preserved':True})
 # Synthetic tied risk / repeated group scenario and unchanged main labels.
 m=pd.DataFrame({'excel_row':[1,2,3,4],'group_hash':['a','a','b','c'],'y':[1,0,1,1]})
 p={'p':np.array([.1,.2,.8,.1])}
 q,v,mask=w.transform(m,p,{1,4},w.SCENARIOS[0]);assert q.excel_row.tolist()==[2,3] and q.y.sum()==1
 q,v,mask=w.transform(m,p,{1,4},w.SCENARIOS[1]);assert q.y.tolist()==[0,0,1,0] and m.y.tolist()==[1,0,1,1]
 # Metric implementation reused unchanged from accepted main evaluation.
 w.check();w.base.atomic(w.OUT/'QUALIFICATION.json',{'status':'PASS_outcome_sensitivity_source_and_transform',
  'source_candidates':34,'proof':proof,'source_prediction_receipts_verified':True,'main_labels_unchanged':True,
  'synthetic_transform_passed':True,'code_lock_sha256':w.base.sha(w.OUT/'CODE_LOCK.json'),'new_performance_scores_computed':False})
if __name__=='__main__':main()
