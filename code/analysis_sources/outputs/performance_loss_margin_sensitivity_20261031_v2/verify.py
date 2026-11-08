from pathlib import Path
import csv,json,hashlib,math
from decimal import Decimal,getcontext
getcontext().prec=50
R=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');T=Path(__file__).resolve().parent
def readcsv(p):return list(csv.DictReader(Path(p).open()))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
c=json.loads((T/'ANALYSIS_CONTRACT.json').read_text())
for p,h in c['source_sha256'].items():assert sha(p)==h
for fn,h in c['code_sha256'].items():assert sha(T/fn)==h
out=readcsv(T/'MARGIN_SENSITIVITY.csv');joint=readcsv(T/'CROSS_CONTEXT_SENSITIVITY.csv');tips=readcsv(T/'TIPPING_POINT_SUMMARY.csv');support=readcsv(T/'PERFORMANCE_SUPPORT.csv')
assert [len(out),len(joint),len(tips),len(support)]==[84,42,4,6]
checks=[];native_numeric=0;maxerr=Decimal(0)
for context,fname,cname in [('geography_preop','preop_full_champion_LR','preop_compact_champion_LR'),('temporal_preop','preop_full_champion_Voting','preop_compact_champion_Voting')]:
 d=R/'outputs/formal_evaluation_20261023_v1'/context
 p={r['model']:r for r in readcsv(d/'POINT_ESTIMATES.csv')}
 native=[r for r in readcsv(d/'PAIRED_DIFFERENCES.csv') if r['model']==fname+' MINUS '+cname]
 nmap={r['metric']:r for r in native};assert len(nmap)==39
 copied=[{k:v for k,v in r.items() if k!='context'} for r in readcsv(T/'ALL_ORIGINAL_PAIRED_METRICS.csv') if r['context']==context]
 assert native==copied
 for row in native:
  if not row['estimate']:
   assert row['metric']=='upgraded_per1000';continue
  metric=row['metric'];err=abs(Decimal(p[fname][metric])-Decimal(p[cname][metric])-Decimal(row['estimate']))
  maxerr=max(maxerr,err);assert err<Decimal('1e-12');native_numeric+=1
 for quota in [20,10]:
  metric=f'capacity{quota}_captured_per1000';src=nmap[metric]
  subset=[r for r in out if r['context']==context and int(r['review_capacity_percent'])==quota]
  assert sorted(Decimal(r['delta_capture_per1000']) for r in subset)==[Decimal(i)/2 for i in range(21)]
  for row in subset:
   delta=Decimal(row['delta_capture_per1000']);u=Decimal(src['upper95'])
   assert row['loss_full27_minus_compact9']==src['estimate'] and row['lower95']==src['lower95'] and row['upper95']==src['upper95']
   assert int(row['bound_strictly_below_margin'])==int(delta>u)
   assert Decimal(row['upper_minus_delta'])==u-delta
  seq=[int(r['bound_strictly_below_margin']) for r in sorted(subset,key=lambda r:Decimal(r['delta_capture_per1000']))]
  assert seq==sorted(seq)
  t=next(r for r in tips if r['context']==context and int(r['review_capacity_percent'])==quota)
  first=next((Decimal(i)/2 for i in range(21) if Decimal(i)/2>Decimal(src['upper95'])),None)
  assert Decimal(t['first_grid_margin_satisfying_rule'])==first
  pref=f'capacity{quota}';f,co=p[fname],p[cname];N=int(f['n'])
  assert int(f[pref+'_reviewed_n'])==int(co[pref+'_reviewed_n'])==math.ceil(N*quota/100)
  exact=Decimal(1000)*(Decimal(f[pref+'_captured_n'])-Decimal(co[pref+'_captured_n']))/N
  assert abs(exact-Decimal(src['estimate']))<Decimal('1e-12')
  checks.append({'context':context,'quota':quota,'capture_count_delta':str(Decimal(f[pref+'_captured_n'])-Decimal(co[pref+'_captured_n'])),'exact_loss_per1000':str(exact),'first_grid_margin':str(first),'source_upper95':src['upper95']})
 for row in (r for r in support if r['context']==context):
  metric=row['metric'];orig=nmap[metric];s=Decimal(-1) if metric=='Brier' else Decimal(1)
  a=s*Decimal(orig['estimate']);lims=sorted([s*Decimal(orig['lower95']),s*Decimal(orig['upper95'])])
  assert (Decimal(row['compact_loss_estimate']),Decimal(row['compact_loss_lower95']),Decimal(row['compact_loss_upper95']))==(a,*lims)
for row in joint:
 q=int(row['review_capacity_percent']);delta=Decimal(row['delta_capture_per1000'])
 matching=[r for r in out if int(r['review_capacity_percent'])==q and Decimal(r['delta_capture_per1000'])==delta]
 flags={r['context']:int(r['bound_strictly_below_margin']) for r in matching};assert len(flags)==2
 assert int(row['both_context_bounds_below_margin'])==int(all(flags.values()))
 assert int(row['geography_bound_below_margin'])==flags['geography_preop'] and int(row['temporal_bound_below_margin'])==flags['temporal_preop']
for delta in ['5','7']:
 rows=[r for r in out if r['review_capacity_percent']=='20' and Decimal(r['delta_capture_per1000'])==Decimal(delta)]
 expected={'geography_preop':0,'temporal_preop':1} if delta=='5' else {'geography_preop':1,'temporal_preop':1}
 assert {r['context']:int(r['bound_strictly_below_margin']) for r in rows}==expected
prot=json.loads((R/'outputs/post_reanalysis_process_review_v1/MANUSCRIPT_PROTECTION.json').read_text())['files']
assert len(prot)==92
for path,z in prot.items():assert sha(path)==z['sha256'],('PROTECTED_CHANGED',path)
receipt={'status':'PASS_INDEPENDENT_NUMERIC_VERIFICATION','native_paired_numeric_differences_verified':native_numeric,
 'maximum_decimal_error':str(maxerr),'all84_sensitivity_cells_verified':True,'all42_cross_context_cells_verified':True,
 'all78_original_paired_metric_rows_exactly_preserved':True,'all6_support_intervals_correct_orientation':True,
 'integer_capture_reconstructions':checks,'protected92_unchanged':True,'source_and_code_hashes_match_contract':True,
 'verification':'Independent Decimal/integer program in a fresh Python process; no clinical expert or new bootstrap claim.'}
with (T/'NUMERICAL_ACCEPTANCE.json').open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':receipt['status'],'checks':checks,'native_numeric_metrics':native_numeric,'protected92_unchanged':True},ensure_ascii=False),flush=True)
