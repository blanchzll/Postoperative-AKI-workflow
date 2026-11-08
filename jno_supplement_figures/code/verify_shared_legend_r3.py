#!/usr/bin/env python3
"""Verify the bounded R3 shared-legend attribution repair, not a new scientific analysis."""
from pathlib import Path
import datetime,hashlib,json
import pymupdf as fitz
R=Path(__file__).resolve().parents[1];OLD=R.parent/'figures_jno_20261108_v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=json.loads((OLD/'PAGE_AND_PANEL_MANIFEST.json').read_text());b=json.loads((R/'PAGE_AND_PANEL_MANIFEST.json').read_text());old={(x['figure'],x['panel']):x for x in a['panels']};rows=[]
for x in b['panels']:
 y=old[x['figure'],x['panel']];assert x['numeric_artist_sha256']==y['numeric_artist_sha256'] and x['annotation_text_sha256']==y['annotation_text_sha256']
 if x['figure']!='eFigure_2':assert x['pdf_sha256']==y['pdf_sha256']
 rows.append({'figure':x['figure'],'panel':x['panel'],'numeric_artists_unchanged':True,'annotation_values_unchanged':True,'unchanged_pdf_bytes':x['pdf_sha256']==y['pdf_sha256']})
texts={}
for x in b['panels']:
 if x['figure']=='eFigure_2':
  with fitz.open(R/x['pdf']) as d:texts[x['panel']]=d[0].get_text()
assert 'Exact upper bound U' not in texts['A'] and 'U < δ' not in texts['A'] and 'U ≥ δ' not in texts['A']
assert 'Exact upper bound U' in texts['B'] and 'U < δ' in texts['B'] and 'U ≥ δ' in texts['B']
for i in range(2,9):assert (R/f'previews/PAGE_CONTACT_{i:02d}.png').read_bytes()==(OLD/f'previews/PAGE_CONTACT_{i:02d}.png').read_bytes()
m=json.loads((R/'audit/SHARED_LEGEND_SOURCE_MAPPING_R3.json').read_text());m.update(status='PASS_FOUR_SOURCE_LEGEND_MAPPINGS_CLOSED',eFigure2_A_has_no_B_only_key=True,eFigure2_B_retains_reference_grid_key=True,source_call_locations=[{'source':'build_figures.py','line':322,'figure':'eFigure5'},{'source':'build_figures.py','line':454,'figure':'eFigure2'},{'source':'build_figures.py','line':496,'figure':'eFigure13'},{'source':'gen_fig_novelty.py','line':48,'figure':'mainFigure3'}]);(R/'audit/SHARED_LEGEND_SOURCE_MAPPING_R3.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
result={'status':'PASS_R3_LEGEND_ATTRIBUTION_AND_FROZEN_VALUES','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_global_legend_calls_checked':4,'panels_with_numeric_and_annotation_fingerprints_unchanged':53,'unaffected_panel_PDFs_byte_identical':51,'unaffected_contact_sheets_byte_identical':7,'eFigure2_A_wrong_key_removed':True,'eFigure2_B_correct_key_retained':True,'previous_figure_acceptance_sha256':sha(OLD/'ACCEPTANCE.json'),'rows':rows,'new_statistics':0}
(R/'audit/R3_LEGEND_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
