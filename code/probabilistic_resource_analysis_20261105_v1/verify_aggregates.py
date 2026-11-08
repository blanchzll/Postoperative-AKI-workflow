from pathlib import Path
import csv,json,math
P=Path(__file__).resolve().parent/'aggregate_data'
def read(n):
 with (P/n).open() as h:return list(csv.DictReader(h))
rows=read('PRIMARY_20_PERCENT_SUMMARY.csv')
assert len(rows)==20
for r in rows:
 ps=[float(r[k]) for k in ['p_win','p_tie','p_loss','p_infeasible']]
 assert all(0<=p<=1 for p in ps)
 assert abs(sum(ps)-1)<1e-10
 mean=float(r['mean_delta_per1000_fallback'])
 base=float(r['expected_regret_base_per1000'])
 extra=float(r['expected_regret_extra_fallback_per1000'])
 assert abs((base-extra)-mean)<1e-9
 assert float(r['mean_delta_per1000_fallback_patient_lower95'])<=float(r['mean_delta_per1000_fallback_patient_upper95'])
 if '9v27' in r['comparison']:
  assert mean<0 and float(r['mean_delta_per1000_fallback_patient_upper95'])<0
normalized=read('NORMALIZED_ESTIMATES.csv');absolute=read('ABSOLUTE_TIME_ESTIMATES.csv')
assert len(normalized)==2880 and len(absolute)==576
assert all(int(r['requested_replicates'])==2000 for r in normalized)
assert all(int(r['draws'])==262144 and r['precision_pass']=='True' for r in absolute)
assert all(r['patient_sampling_interval']=='NOT_COMPUTED_IN_SUPPORTIVE_COORDINATE_CHECK' for r in absolute)
summary=read('DISTRIBUTION_SENSITIVITY_SUMMARY.csv')
assert len(summary)==4
print(json.dumps({'status':'PASS_AGGREGATE_COUNTS_PROBABILITY_MASS_LOSS_IDENTITY_AND_REPORTED_PRECISION','primary_rows':len(rows),'normalized_rows':len(normalized),'absolute_rows':len(absolute),'new_fits':0,'new_bootstraps':0}))
