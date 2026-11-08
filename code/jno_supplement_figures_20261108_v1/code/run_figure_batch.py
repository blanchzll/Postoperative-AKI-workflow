#!/usr/bin/env python3
"""Finite resource-bounded native panel export; no model or statistical jobs."""
from pathlib import Path
import concurrent.futures,datetime,json,os,subprocess,sys,time,hashlib
R=Path(__file__).resolve().parents[1];W=R/'workers';W.mkdir(exist_ok=True)
labels=[f'eFigure_{i}' for i in range(1,19)]
mem={line.split(':',1)[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))}
observed_peak_MiB=209;per_process_reservation_MiB=600;headroom=max(4*1024**3,int(mem['MemAvailable']*.25))
workers=min(len(labels),len(os.sched_getaffinity(0)),max(1,int((mem['MemAvailable']-headroom)/(per_process_reservation_MiB*1024**2))))
receipt={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_affinity':len(os.sched_getaffinity(0)),'MemAvailable_bytes':mem['MemAvailable'],'reserved_headroom_bytes':headroom,'qualification_observed_peak_MiB':observed_peak_MiB,'reserved_MiB_per_worker':per_process_reservation_MiB,'workers':workers,'tasks':18,'threads_per_worker':1,'new_statistics':0}
(R/'audit/BATCH_LAUNCH.json').write_text(json.dumps(receipt,indent=2)+'\n')
script=R/'code/export_jno_panels.py';python=sys.executable;started=time.time()
def run(label):
 root=W/label;root.mkdir(exist_ok=True)
 for n in ['panels','audit','runtime','logs']:(root/n).mkdir(exist_ok=True)
 log=R/'logs'/(label+'.log')
 with log.open('w') as handle:
  proc=subprocess.run([python,str(script),'--root',str(root),'--only',label],stdout=handle,stderr=subprocess.STDOUT)
 return {'figure':label,'returncode':proc.returncode,'log':str(log),'root':str(root)}
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
 outcomes=list(pool.map(run,labels))
receipt.update({'outcomes':outcomes,'seconds':time.time()-started,'all_owned_workers_joined':True})
(R/'audit/BATCH_RUN.json').write_text(json.dumps(receipt,indent=2)+'\n')
failed=[x for x in outcomes if x['returncode']]
if failed:
 print(json.dumps({'status':'PANEL_EXPORT_FAILURES_RETAINED','failed':failed,'workers':workers,'seconds':receipt['seconds']},indent=2));sys.exit(1)
# Retain worker-relative paths in the top-level manifest; no data-dependent rerouting.
combined=None
for label in labels:
 root=W/label;j=json.loads((root/'PANEL_MANIFEST.json').read_text())
 if combined is None:combined={k:v for k,v in j.items() if k not in ['panels','figures','source_csvs','source_scripts']};combined.update(panels=[],figures=[],source_csvs={},source_scripts={})
 for p in j['panels']:
  for k in ['pdf','svg','png']:p[k]=str(Path('workers')/label/p[k])
  combined['panels'].append(p)
 combined['figures'].extend(j['figures']);combined['source_csvs'].update(j['source_csvs']);combined['source_scripts'].update(j['source_scripts'])
combined.update(status='ALL_18_SUPPLEMENTAL_NATIVE_PANELS_EXPORTED_PENDING_COMPOSITION_QA',seconds=receipt['seconds'],workers=workers,export_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),panels_count=len(combined['panels']),figures_count=18)
assert len({(p['figure'],p['panel']) for p in combined['panels']})==len(combined['panels'])
assert len({p['figure'] for p in combined['panels']})==18
(R/'PANEL_MANIFEST.json').write_text(json.dumps(combined,indent=2)+'\n')
print(json.dumps({'status':combined['status'],'panels':len(combined['panels']),'figures':18,'workers':workers,'seconds':receipt['seconds']},indent=2))
