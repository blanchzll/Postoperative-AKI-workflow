#!/usr/bin/env python3
"""Merge completed source-preserving figure exports without relaunching workers."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parents[1];combined=None;byversion={}
for i in range(1,19):
 label=f'eFigure_{i}';p=R/'workers'/label/'PANEL_MANIFEST.json';j=json.loads(p.read_text());assert len(j['figures'])==1
 if combined is None:combined={k:v for k,v in j.items() if k not in ['panels','figures','source_csvs','source_scripts']};combined.update(panels=[],figures=[],source_csvs={},source_scripts={})
 for x in j['panels']:
  for k in ['pdf','svg','png']:x[k]=str(Path('workers')/label/x[k])
  combined['panels'].append(x)
 combined['figures'].extend(j['figures']);combined['source_csvs'].update(j['source_csvs']);combined['source_scripts'].update(j['source_scripts']);byversion[label]=j['script_sha256']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(sha(p)==h for p,h in {**combined['source_scripts'],**combined['source_csvs']}.items())
combined.update(status='ALL_18_SUPPLEMENTAL_NATIVE_PANELS_EXPORTED_PENDING_COMPOSITION_QA',merged_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),panels_count=len(combined['panels']),figures_count=18,worker_export_script_versions=byversion,export_script_sha256=sha(R/'code/export_jno_panels.py'),source_hashes_unchanged=True)
assert combined['panels_count']==53
(R/'PANEL_MANIFEST.json').write_text(json.dumps(combined,indent=2)+'\n');print(json.dumps({'figures':18,'panels':53,'numeric_artists_baseline_matched':sum(x['baseline_native_numeric_match'] for x in combined['panels'])}))
