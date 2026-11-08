from pathlib import Path
from pypdf import PdfReader
from PIL import Image,ImageDraw
import subprocess,os,json
W=Path(__file__).resolve().parents[1];O=W/'review/previews';D=W/'review/rendered'
targets=[('MANUSCRIPT',1),('MANUSCRIPT',3),('METHODS_EN',2),('SUPPLEMENTARY_TABLE_1',1),('SUPPLEMENTARY_TABLE_2',1),('SUPPLEMENT_1',50),('SUPPLEMENT_1',51),('COVER_LETTER',1)]
board=Image.new('RGB',(2000,1450),'#e4e4e4');draw=ImageDraw.Draw(board)
for i,(name,page) in enumerate(targets):
 pdf=PdfReader(D/(name+'.pdf'));page=min(page,len(pdf.pages))
 out='qa'+str(i)
 cmd=['docker','run','--rm','--name','aki-novelty-qa-'+str(os.getpid())+'-'+str(i),'--network','none','--memory','1g','--user',str(os.getuid())+':'+str(os.getgid()),'--mount','type=bind,source='+str(W)+',target=/work','--entrypoint','/usr/bin/pdftoppm','codex-docx-render:20260823','-f',str(page),'-l',str(page),'-singlefile','-scale-to','1400','-png','/work/review/rendered/'+name+'.pdf','/work/review/previews/'+out]
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=45);assert r.returncode==0,r.stderr
 with Image.open(O/(out+'.png')) as im:
  im.thumbnail((500,690));x=(i%4)*500;y=(i//4)*725
  board.paste(im,(x+(500-im.width)//2,y+25));draw.text((x+8,y+5),name+', p'+str(page),fill='black')
board.save(O/'DOCUMENT_CONTACT_SHEET.jpg',quality=92)
print('Rendered eight review pages')
