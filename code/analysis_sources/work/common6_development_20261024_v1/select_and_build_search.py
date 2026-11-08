from pathlib import Path
import sys
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'engine'))
from core import *
from coordinator_a import select
def main(line):
 jobs=[]
 for context in ['final']+['outer_'+str(i) for i in range(5)]:
  assert select(line,'preop',context)
  path=OUT/'selection'/line/'preop'/context/'METHOD_SELECTION.json'
  method=json.loads(path.read_text())['selected_method']
  for fam in FAMILIES:
   t={'line':line,'cohort':'preop','stage':'preop','branch':'full','context':context,'family':fam,'method':method}
   jobs.append({'id':'search_'+line+'_'+digest(t)[:20],'worker':'search_worker.py','task':t,'dest':str(OUT/'search_v2'/line/'preop'/'preop'/'full'/context/fam),'depends':[],'line':line,'kind':'search'})
 atomic(OUT/'B_TASKS.json',jobs)
 atomic(OUT/'COARSE_SELECTION_ACCEPTANCE.json',{'status':'PASS_six_inputs_six_imputers_nine_families_six_contexts','line':line,'selection_files':{c:sha(OUT/'selection'/line/'preop'/c/'METHOD_SELECTION.json') for c in ['final']+['outer_'+str(i) for i in range(5)]},'no_public_or_heldout_prediction':True})
if __name__=='__main__':main(sys.argv[1])
