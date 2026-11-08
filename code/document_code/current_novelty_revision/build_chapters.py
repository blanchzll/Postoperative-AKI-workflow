from pathlib import Path
import runpy,json,shutil,difflib
W=Path(__file__).resolve().parents[1];T=W/'work/text';P=W/'package'
ns=runpy.run_path(str(W/'code/build_documents.py'))
make,md,write=ns['style_doc'],ns['add_md'],ns['write']
D=P/'04_Editable_sources/Chapters';D.mkdir(exist_ok=True)
names={'EN':[('Introduction','INTRODUCTION'),('Methods','METHODS'),('Results','RESULTS'),('Discussion','DISCUSSION'),('Conclusions','CONCLUSIONS')],'ZH':[('前言','INTRODUCTION'),('方法','METHODS'),('结果','RESULTS'),('讨论','DISCUSSION'),('结论','CONCLUSIONS')]}
counts={}
for lang,items in names.items():
 s=(T/('MANUSCRIPT_'+lang+'.md')).read_text()
 for heading,stem in items:
  chunk='## '+heading+'\n'+s.split('## '+heading+'\n',1)[1].split('\n## ',1)[0]
  f=D/(stem+'_'+lang)
  f.with_suffix('.md').write_text(chunk+'\n');d=make();md(d,chunk);write(d,f.with_suffix('.docx'));counts[f.name]=len(chunk)
for n in ['FIGURE_LEGENDS','MAIN_TABLES','ARGUMENT_OUTLINE_ZH']:
 shutil.copy2(T/(n+'.md'),D/(n+'.md'))
for lang in ['EN','ZH']:
 before=(W/'inputs'/('BASELINE_MANUSCRIPT_'+lang+'.md')).read_text()
 after=(T/('MANUSCRIPT_'+lang+'.md')).read_text()
 diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='baseline_'+lang,tofile='revision_'+lang))
 (W/'review'/('MANUSCRIPT_'+lang+'_REVISIONS.diff')).write_text(diff)
 html=difflib.HtmlDiff(wrapcolumn=80).make_file(before.splitlines(),after.splitlines(),fromdesc='Accepted baseline',todesc='Joint-budget revision',context=True,numlines=2)
 (P/'03_Author_review'/('MANUSCRIPT_'+lang+'_COMPARISON.html')).write_text(html)
(W/'review/CHAPTER_BUILD.json').write_text(json.dumps(counts,indent=2))
print('Separate chapters built:',len(counts))
