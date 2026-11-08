from pathlib import Path
import os,subprocess,json,time,shutil
W=Path(__file__).resolve().parents[1];P=W/'package'
mem={l.split(':')[0]:l.split(':')[1].strip() for l in Path('/proc/meminfo').read_text().splitlines()}
if int(mem['MemAvailable'].split()[0])<10*1024*1024:raise SystemExit('Insufficient actual memory headroom for document renderer')
files=sorted((P/'01_Journal_files').rglob('*.docx'))
files+=[P/'03_Author_review/MANUSCRIPT_ZH.docx',P/'03_Author_review/MANUSCRIPT_TRACK_CHANGES.docx']
if (P/'03_Author_review/TITLE_PAGE_AUTHOR_REVIEW.docx').exists():files.append(P/'03_Author_review/TITLE_PAGE_AUTHOR_REVIEW.docx')
for prefix in ['INTRODUCTION','METHODS','RESULTS','DISCUSSION','CONCLUSIONS']:
 for lang in ['EN','ZH']:files.append(P/f'04_Editable_sources/Chapters/{prefix}_{lang}.docx')
assert len({p.stem for p in files})==len(files)
out=W/'review/rendered';out.mkdir(exist_ok=True)
args=['docker','run','--rm','--user',f'{os.getuid()}:{os.getgid()}','--memory','8g','-v',str(W)+':/work','-v',str(W/'fonts')+':/usr/share/fonts/truetype/aki_fonts:ro','--entrypoint','/usr/bin/libreoffice','codex-docx-render:20260823','-env:UserInstallation=file:///tmp/aki-doc-20261108','--headless','--convert-to','pdf','--outdir','/work/review/rendered']+['/work/'+str(p.relative_to(W)) for p in files]
start=time.time();run=subprocess.run(args,capture_output=True,text=True)
(W/'review/DOCUMENT_RENDER.log').write_text(run.stdout+'\nSTDERR\n'+run.stderr)
assert run.returncode==0,(run.returncode,run.stderr[-1000:])
records=[]
for source in files:
 pdf=out/(source.stem+'.pdf');assert pdf.exists() and pdf.stat().st_size>5000,pdf
 shutil.copy2(pdf,source.with_suffix('.pdf'))
 records.append({'docx':str(source.relative_to(W)),'pdf':str(source.with_suffix('.pdf').relative_to(W)),'bytes':pdf.stat().st_size})
(W/'review/DOCUMENT_RENDER.json').write_text(json.dumps({'status':'PDF_RENDERED_CONTENT_REVIEW_PENDING','exit_code':run.returncode,'elapsed_seconds':time.time()-start,'MemAvailable_at_launch':mem['MemAvailable'],'files':records,'renderer_image':'codex-docx-render:20260823','container_removed_on_exit':True},indent=2))
print(json.dumps({'exit_code':run.returncode,'PDFs':len(records),'elapsed_seconds':time.time()-start}))
