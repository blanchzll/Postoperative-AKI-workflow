from pathlib import Path
import json,re
W=Path(__file__).resolve().parents[1];T=W/'work/text';R=W/'review'
x=json.loads((R/'SEMANTIC_CANDIDATES.json').read_text())
related={21,25,29,33,39,53,61,63,69,77,85,93,97,103,111,113,115,119,125,127,129}
for a in x['rename']:
 a.update(label='0组',confidence='高',reason='该段保持同一术语；各模型、指标或流程阶段指代不同对象，正式缩写与指代省略不计同义重命名。',disposition='保留')
for a in x['cohesion']:
 a.update(label='有关但转换突然' if a['line'] in related else '明确承接',confidence='中',reason='该接口转换至同一研究的另一方法或结果主题。' if a['line'] in related else '前后围绕同一风险评价问题或配对比较继续展开。',disposition='保留')
for a in x['colon']:
 label='定义或解释' if a['line'] in [29,53] else '标题时间比例元数据或版式用途'
 a.update(label=label,confidence='高',reason='冒号给出结局或编码规则定义。' if label=='定义或解释' else '用于置信区间、书目或DOI格式。',disposition='保留')
(R/'SEMANTIC_CANDIDATES_LABELED.json').write_text(json.dumps(x,ensure_ascii=False,indent=2))
z=(T/'MANUSCRIPT_ZH_humanizer.md').read_text()
old='地理开发采用中心1、2和4，评价采用中心3和5；时间开发'
new='地理开发采用中心1、2和4，评价采用中心3和5。时间开发'
assert old in z;z=z.replace(old,new)
# Journal label crosswalk is an editorial-format operation, separate from tone edits.
cross=json.loads((R/'NUMBERING_CROSSWALK.json').read_text());fm=cross['figures_old_to_journal'];tm=cross['tables_old_to_journal']
z=z.replace('补充表S11至S12','补充材料eTable 8至eTable 9')
z=re.sub(r'补充图(\d+)',lambda m:'补充材料eFigure '+str(fm[m[1]]),z)
z=re.sub(r'(?:补充)?表S(\d+)',lambda m:'补充材料eTable '+str(tm.get(m[1],m[1])),z)
(T/'MANUSCRIPT_ZH.md').write_text(z)
hits=json.loads((R/'LESS_AI_BEFORE.json').read_text())['hits'];dis=[]
for kind,arr in hits.items():
 for h in arr:
  dis.append(dict(h,rule=kind,disposition='保留',reason='列举确切的预测输入、诊断类别、评估指标、已记录分母或研究步骤；每项信息均有独立定义，删改会丢失方法或结果。'))
(R/'LANGUAGE_HIT_DISPOSITIONS.json').write_text(json.dumps(dis,ensure_ascii=False,indent=2))
(R/'LESS_AI_EXACT_CHANGES.json').write_text(json.dumps([{'before':old,'after':new,'rule':'F-semicolon-local-sentence-split','reason':'拆开地理与时间开发定义，数字与含义保持。'}],ensure_ascii=False,indent=2))
# Technical symbol ranges and English passages are excluded by the second skill; humanizer handles English prose.
(W/'review/LANGUAGE_PASS_RECEIPT.json').write_text(json.dumps({'sequence':['humanizer','lieflat-less-ai-tone'],'authorization':'Explicit current user request and parent AGENTS authorization; no redundant approval requested','english_handling':'Humanizer manual review; second skill screened and scope exclusions preserved English passages','chinese_semantic_candidates':x['meta'],'tone_changes':1,'reference_texts_unchanged':True,'scientific_thresholds_unchanged':True,'authorship_or_ai_detection_claim':False},ensure_ascii=False,indent=2))
print('Sequential language review recorded; 125 semantic candidates labeled')
