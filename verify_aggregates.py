from pathlib import Path
import csv,json,math
P=Path(__file__).resolve().parent/'aggregate_data'
checks=[]
def read(rel):
 with (P/rel).open() as f:return list(csv.DictReader(f))
def check(name,condition):
 assert condition,name
 checks.append(name)
counts=read('tables/Table_S13_CLINICAL_COUNTS_AT_CAPACITY.csv')
for r in counts:
 n=int(r['n']);q=float(r['review_capacity_percent'])/100
 check(r['context']+r['review_capacity_percent']+' selection',int(r['reviewed_each'])==math.ceil(q*n))
 for k in ['full27','compact9']:
  check(r['context']+k+' capture arithmetic',abs(float(r[k+'_capture_per1000'])-1000*int(r[k+'_captured'])/n)<1e-10)
  check(r['context']+k+' event partition',int(r[k+'_captured'])+int(r[k+'_missed'])==float(r['events']))
grid=read('tables/Table_S13_MARGIN_SENSITIVITY.csv')
for r in grid:
 check('strict unrounded margin '+r['context']+' '+r['review_capacity_percent']+' '+r['delta_capture_per1000'],int(r['bound_strictly_below_margin'])==int(float(r['loss_upper_bound'])<float(r['delta_capture_per1000'])))
joint=read('tables/Table_S13_CROSS_CONTEXT_SENSITIVITY.csv')
for r in joint:
 check('joint rule '+r['review_capacity_percent']+' '+r['delta_capture_per1000'],int(r['both_context_bounds_below_margin'])==(int(r['geography_bound_below_margin']) & int(r['temporal_bound_below_margin'])))
first=min(float(r['delta_capture_per1000']) for r in joint if r['review_capacity_percent']=='20' and r['both_context_bounds_below_margin']=='1')
check('first joint 20 percent grid margin',first==7)
check('grid complete',len(grid)==84 and len(joint)==42)
print(json.dumps({'status':'PASS_SAVED_AGGREGATE_ARITHMETIC','checks':len(checks),'new_fits':0,'new_bootstraps':0}))
