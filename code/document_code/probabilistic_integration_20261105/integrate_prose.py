from pathlib import Path
import json,re,difflib
W=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/writing_runs/probabilistic_integration_20261105_v1')
P=W/'package';S=P/'04_Editable_sources';OLD=W.parent/'discussion_deepening_20261104_v1/package'
E=json.loads((W/'work/PROSE_EDITS.json').read_text())
logs=[]
def para(t,start,new,label):
 matches=[p for p in t.split('\n\n') if p.startswith(start)]
 assert len(matches)==1,(label,start,len(matches))
 old=matches[0];logs.append({'item':label,'before':old,'after':new})
 return t.replace(old,new,1)
for lang,key in [('EN','en'),('ZH','zh')]:
 e=E[key];t=(S/f'MANUSCRIPT_{lang}.md').read_text()
 if lang=='EN':
  t=para(t,'**Findings:**','**Findings:** '+e['keyfind'],'keypoints EN')
  t=para(t,'**Results:**','**Results:** '+e['abstractResults'],'abstract results EN')
  t=para(t,'**Conclusions and Relevance:**','**Conclusions and Relevance:** '+e['abstractConclusion'],'abstract conclusion EN')
  t=para(t,'We compared frozen nine-',e['intro'],'introduction EN')
  before,rest=t.split('## Results\n',1)
  before=before.replace('\n### Public-source evaluation','\n'+e['method']+'\n\n### Public-source evaluation',1)
  t=before+'## Results\n'+rest
  t=t.replace('\nIncreasing nine-input capacity','\n'+e['result']+'\n\nIncreasing nine-input capacity',1)
  h,end='Discussion','Conclusions'
  t=para(t,'Additional information had different operational value',e['disc1'],'discussion opening EN')
  t=para(t,'For the capture target used here',e['disc3'],'discussion resource interpretation EN')
  t=para(t,'Equal-capture boundaries express that opportunity cost',e['disc4'],'discussion boundary EN')
  t=para(t,'The study has several limitations.',e['limits'],'limitations EN')
  t=para(t,'Joint-budget evaluation quantified',e['conclusion'],'conclusions EN')
  t=t.replace('through November 4, 2026.','through November 5, 2026.')
 else:
  t=para(t,'**发现：**','**发现：** '+e['keyfind'],'keypoints ZH')
  t=para(t,'**结果：**','**结果：** '+e['abstractResults'],'abstract results ZH')
  t=para(t,'**结论与相关性：**','**结论与相关性：** '+e['abstractConclusion'],'abstract conclusion ZH')
  t=para(t,'本研究比较胃肠肿瘤术后AKI的冻结',e['intro'],'introduction ZH')
  before,rest=t.split('## 结果\n',1)
  before=before.replace('\n### 公共来源评价','\n'+e['method']+'\n\n### 公共来源评价',1)
  t=before+'## 结果\n'+rest
  t=t.replace('\n九项输入容量从10%增至20%', '\n'+e['result']+'\n\n九项输入容量从10%增至20%',1)
  h,end='讨论','结论'
  t=para(t,'增加信息的操作价值随评价场景而变化。',e['disc1'],'discussion opening ZH')
  t=para(t,'对于本研究的捕获目标',e['disc3'],'discussion resource interpretation ZH')
  t=para(t,'等捕获边界以指定基础复核容量',e['disc4'],'discussion boundary ZH')
  t=para(t,'本研究存在若干局限。',e['limits'],'limitations ZH')
  t=para(t,'共同预算评价量化了',e['conclusion'],'conclusions ZH')
  t=t.replace('本轮记录更新至2026年11月4日。','本轮记录更新至2026年11月5日。')
 (S/f'MANUSCRIPT_{lang}.md').write_text(t)
 for section,stem in ([('Introduction','INTRODUCTION'),('Methods','METHODS'),('Results','RESULTS'),('Discussion','DISCUSSION'),('Conclusions','CONCLUSIONS')] if lang=='EN' else [('前言','INTRODUCTION'),('方法','METHODS'),('结果','RESULTS'),('讨论','DISCUSSION'),('结论','CONCLUSIONS')]):
  chunk='## '+section+'\n'+t.split('## '+section+'\n',1)[1].split('\n## ',1)[0]
  (S/'Chapters'/f'{stem}_{lang}.md').write_text(chunk.rstrip()+'\n')
 before=(OLD/f'04_Editable_sources/MANUSCRIPT_{lang}.md').read_text()
 (W/f'review/MANUSCRIPT_{lang}_CHANGES.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),t.splitlines(True),fromfile='discussion_deepening_'+lang,tofile='probabilistic_integration_'+lang)))
(W/'review/PROSE_EDIT_LOG.json').write_text(json.dumps(logs,ensure_ascii=False,indent=2))
main=(S/'MANUSCRIPT_EN.md').read_text()
counts={'main_body':len(('Introduction\n'+main.split('## Introduction\n',1)[1].split('## Acknowledgment',1)[0]).split()),'abstract':len(main.split('## Abstract\n',1)[1].split('## Introduction\n',1)[0].split()),'key_points':len(main.split('## Key Points\n',1)[1].split('## Abstract\n',1)[0].split()),'discussion':len(main.split('## Discussion\n',1)[1].split('## Conclusions\n',1)[0].split())}
(W/'review/WORD_COUNTS.json').write_text(json.dumps(counts,indent=2))
print(json.dumps(counts))
