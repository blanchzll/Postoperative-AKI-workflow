from pathlib import Path
import subprocess,os,json
from PIL import Image,ImageOps,ImageDraw
from pypdf import PdfReader
W=Path(__file__).resolve().parents[1];D=W/'review/rendered';O=W/'review/previews'
sp=PdfReader(D/'SUPPLEMENT_1.pdf')
find=lambda txt:next(i+1 for i,p in enumerate(sp.pages) if txt in p.extract_text()[:160])
targets=[('TABLE_1',1),('TABLE_2',1),('COVER_LETTER',1),('MANUSCRIPT',3),('SUPPLEMENT_1',find('eTable 18.')),('SUPPLEMENT_1',find('eFigure 2.'))]
imgs=[]
for num,(name,page) in enumerate(targets):
 pref=f'p{num}'
 cmd=['docker','run','--rm','--name',f'aki-preview-{os.getpid()}-{num}','--network','none','--user',f'{os.getuid()}:{os.getgid()}','--mount',f'type=bind,source={W},target=/work','--entrypoint','/usr/bin/pdftoppm','codex-docx-render:20260823','-f',str(page),'-l',str(page),'-singlefile','-scale-to','1100','-png','/work/review/rendered/'+name+'.pdf','/work/review/previews/'+pref]
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=60);assert r.returncode==0,r.stderr
 im=Image.open(O/(pref+'.png')).convert('RGB');im.thumbnail((500,710));imgs.append((name,page,im.copy()));im.close()
board=Image.new('RGB',(1500,1490),'#d9dde3');draw=ImageDraw.Draw(board)
for i,(name,page,im) in enumerate(imgs):
 x=(i%3)*500;y=(i//3)*745;board.paste(im,(x+(500-im.width)//2,y+25));draw.text((x+12,y+6),f'{name}, page {page}',fill='black');im.close()
board.save(O/'DOCUMENT_CONTACT_SHEET.png');board.close()
print('Preview board generated',len(targets))
