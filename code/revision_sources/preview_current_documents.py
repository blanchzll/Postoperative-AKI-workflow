from pathlib import Path
import json,subprocess,os,shlex,hashlib
from PIL import Image,ImageDraw
W=Path(__file__).resolve().parents[1];q=json.loads((W/'review/DOCUMENT_QUALIFICATION.json').read_text())
out=W/'review/page_previews';out.mkdir(exist_ok=True)
cmds=[]
for r in q['selected_preview_pages']:
 cmds.append(['/usr/bin/pdftoppm','-f',str(r['page']),'-l',str(r['page']),'-singlefile','-scale-to','1300','-png','/work/'+r['file'],'/work/review/page_previews/'+r['label']])
shell='\n'.join(' '.join(shlex.quote(s) for s in c) for c in cmds)
r=subprocess.run(['docker','run','--rm','--user',f'{os.getuid()}:{os.getgid()}','--memory','2g','-v',str(W)+':/work','--entrypoint','/bin/sh','codex-docx-render:20260823','-e','-c',shell],capture_output=True,text=True)
assert r.returncode==0,r.stderr
items=q['selected_preview_pages']
for j in range((len(items)+3)//4):
 group=items[j*4:j*4+4];canvas=Image.new('RGB',(1860,2700 if len(group)>2 else 1350),'#dddddd');draw=ImageDraw.Draw(canvas)
 for k,item in enumerate(group):
  im=Image.open(out/(item['label']+'.png')).convert('RGB');im.thumbnail((910,1300));x=(k%2)*930;y=(k//2)*1350;canvas.paste(im,(x,y+25));draw.text((x+5,y+5),item['label']+' page '+str(item['page']),fill='black')
 canvas.save(out/f'PROOF_CONTACT_{j+1}.png')
(W/'review/PAGE_PREVIEW_RUN.json').write_text(json.dumps({'exit_code':0,'pages':len(items),'PDF_hashes':{f:hashlib.sha256((W/f).read_bytes()).hexdigest() for f in {x['file'] for x in items}},'contact_pages':3},indent=2))
print('Fresh selected PDF previews generated',len(items))
