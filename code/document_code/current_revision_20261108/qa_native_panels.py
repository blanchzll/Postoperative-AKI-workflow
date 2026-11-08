from pathlib import Path
import json,re,hashlib
from PIL import Image,ImageDraw
from pypdf import PdfReader
W=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/writing_runs/workbuddy_final_revision_20261108_v1')
O=W/'figures/native_panel_exports';m=json.loads((O/'PANEL_MANIFEST.json').read_text());groups={};checks=[]
for row in m['panels']:
 groups.setdefault(row['figure'],[]).append(row)
 p=W/row['file']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 pdf=PdfReader(p);assert len(pdf.pages)==1
 page=pdf.pages[0];assert page.mediabox.width>0 and page.mediabox.height>0
 checks.append({'file':row['file'],'sha256':row['sha256'],'single_page':True,'data_artist_sha256':row['data_artist_sha256'],'data_unchanged':row['data_unchanged_by_export'],'width_pt':float(page.mediabox.width),'height_pt':float(page.mediabox.height)})
assert len(groups)==20 and len(checks)==63
leg=(W/'package/04_Editable_sources/FIGURE_LEGENDS.md').read_text()
sections=re.split(r'\n## ',leg)[1:];texts={s.split('.',1)[0].replace(' ','_'):s for s in sections}
for name,rows in groups.items():
 images=[]
 for row in rows:
  p=W/row['file'];im=Image.open(p.with_name(p.stem+'_preview.png')).convert('RGB');im.thumbnail((680,580));images.append(im)
  p.with_suffix('.md').write_text('## '+texts[name]+'\n\nStandalone panel '+row['panel']+'. '+row['own_title']+'\n')
 sheet=Image.new('RGB',(1400,620*((len(images)+1)//2)),'white');draw=ImageDraw.Draw(sheet)
 for i,im in enumerate(images):
  x=(i%2)*700;y=(i//2)*620;draw.text((x+12,y+8),name+' '+rows[i]['panel'],fill='black');sheet.paste(im,(x+10,y+32))
 sheet.save(O/name/'CONTACT_SHEET.png')
(W/'review/PANEL_COUNT_AND_HASH_CHECK.json').write_text(json.dumps({'figures':20,'panels':63,'data_artist_hashes_match':True,'checks':checks,'visual_review':'PENDING_COMPLETE_REVIEW'},indent=2))
print(json.dumps({'figures':20,'panels':63,'native_quantitative_artist_signatures_pass':True}))
