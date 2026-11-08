from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Ellipse,Circle,PathPatch,Rectangle,FancyArrowPatch
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
W=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/writing_runs/probabilistic_integration_20261105_v1')
O=W/'package/06_Graphical_abstract';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
C={'ink':'#203447','muted':'#607487','blue':'#216A9A','teal':'#23867C','coral':'#B95643','pale':'#EFF5F8','line':'#D9E4EA','skin':'#E8B9A3','lightteal':'#EAF4F0'}
fig=plt.figure(figsize=(18,11),facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1800),ylim=(1100,0));ax.axis('off')
def text(x,y,s,size=15,color=None,weight='normal',ha='left',va='top',**kwargs):
 return ax.text(x,y,s,fontsize=size,color=color or C['ink'],fontweight=weight,ha=ha,va=va,linespacing=1.35,**kwargs)
def box(x,y,w,h,fc='white',ec=None,r=16,lw=1):
 patch=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',facecolor=fc,edgecolor=ec or fc,linewidth=lw)
 ax.add_patch(patch);return patch
def line(points,color=None,lw=1.5,alpha=1):
 xx,yy=zip(*points);ax.plot(xx,yy,color=color or C['ink'],lw=lw,solid_capstyle='round',alpha=alpha)
def arrow(x1,y1,x2,y2,color=None,lw=1.6,scale=15):
 ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=scale,lw=lw,color=color or C['muted']))
def path(vertices,codes,fc='none',ec=None,lw=2):
 p=PathPatch(MPath(vertices,codes),facecolor=fc,edgecolor=ec or C['ink'],lw=lw,joinstyle='round',capstyle='round');ax.add_patch(p);return p
def clipboard(x,y,w=64,h=84,color=None):
 color=color or C['blue'];box(x,y,w,h,'white',color,7,1.6)
 box(x+w*.28,y-5,w*.44,12,C['line'],color,3,1)
 for i in range(3):
  line([(x+12,y+25+i*17),(x+w-12,y+25+i*17)],color,1.3)
def clinician(x,y,s=1):
 ax.add_patch(Circle((x,y),12*s,facecolor=C['skin'],edgecolor=C['ink'],lw=1))
 box(x-16*s,y+13*s,32*s,42*s,'white',C['blue'],8,1)
 path([(x-8*s,y+18*s),(x,y+29*s),(x+8*s,y+18*s)],[MPath.MOVETO,MPath.LINETO,MPath.LINETO],ec=C['teal'],lw=1.5)
 line([(x-7*s,y+55*s),(x-7*s,y+73*s)],C['ink'],1.8)
 line([(x+7*s,y+55*s),(x+7*s,y+73*s)],C['ink'],1.8)
 ax.add_patch(Circle((x+5*s,y+38*s),3*s,fill=False,edgecolor=C['teal'],lw=1))
def kidney(cx,cy,flip=1,s=1):
 pts=[(cx-14*flip*s,cy-28*s),(cx-52*flip*s,cy-46*s),(cx-57*flip*s,cy+34*s),(cx-17*flip*s,cy+34*s),(cx+5*flip*s,cy+26*s),(cx-12*flip*s,cy+7*s),(cx-8*flip*s,cy-2*s),(cx+7*flip*s,cy-15*s),(cx+3*flip*s,cy-20*s),(cx-14*flip*s,cy-28*s)]
 codes=[MPath.MOVETO]+[MPath.CURVE4]*9
 path(pts,codes,fc='#DFA18F',ec=C['coral'],lw=1.7)
 line([(cx-7*flip*s,cy+4*s),(cx+4*flip*s,cy+19*s),(cx+7*flip*s,cy+57*s)],C['coral'],1.3)
def hospital(x,y):
 box(x,y,68,60,'white',C['blue'],4,1)
 for yy in [13,31]:
  for xx in [12,29,46]:box(x+xx,y+yy,8,8,C['pale'],C['blue'],1,.7)
 box(x+29,y+45,12,15,C['pale'],C['blue'],1,.8)
# Header.
box(52,38,6,78,C['teal'],r=2)
text(78,37,'Perioperative information for AKI risk review',29,weight='bold')
text(79,91,'Gastrointestinal cancer surgery  |  Multicentre retrospective cohort study',15,C['muted'])
text(55,142,'How much information is useful when clinical review capacity is limited?',19,C['ink'],weight='bold')
line([(55,191),(1745,191)],C['line'],1)
# Three clinical sections.
for x,w,heading in [(55,470,'PATIENTS & CLINICAL SETTING'),(560,610,'INFORMATION & REVIEW CHOICES'),(1210,535,'FINDINGS FOR CLINICAL REVIEW')]:
 text(x,219,heading,13,C['muted'],weight='bold')
text(55,254,'3,710 operations',25,weight='bold')
text(55,295,'5 Chinese centres · 205 recorded AKI events',13,C['muted'])
# Operating room scene.
box(55,335,470,294,C['pale'],r=20)
# Surgical lamp and monitor.
line([(106,352),(106,378),(170,400)],C['muted'],1.5)
ax.add_patch(Ellipse((176,404),73,19,angle=16,facecolor='white',edgecolor=C['muted'],lw=1))
box(412,367,81,58,'white',C['ink'],5,1)
line([(420,400),(433,400),(441,388),(448,410),(456,396),(465,400),(484,400)],C['teal'],1.2)
line([(451,427),(451,494)],C['muted'],1.5)
# Patient and table.
box(115,488,295,30,'white',C['muted'],8,1)
box(247,518,27,57,C['line'],C['muted'],3,.8)
line([(205,577),(318,577)],C['muted'],2)
ax.add_patch(Ellipse((147,469),44,40,facecolor=C['skin'],edgecolor=C['muted'],lw=1))
path([(127,458),(128,444),(156,441),(169,456)],[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4],fc='#C5E2EA',ec=C['blue'],lw=1)
box(173,452,218,36,'#B6D7E4',C['blue'],10,1)
ax.add_patch(Ellipse((271,467),53,24,facecolor='#E5C4B4',edgecolor=C['blue'],lw=1))
# Clinicians beside patient.
clinician(107,420,.85);clinician(370,390,.9)
text(81,591,'Gastric or colorectal cancer resection',12,C['muted'])
# Clinical renal focus below scene.
kidney(133,704,1,.72);kidney(166,704,-1,.72)
text(220,660,'Postoperative\nrenal assessment',16,weight='bold')
text(220,719,'Recorded AKI and\ncreatinine events',12.5,C['muted'])
# Timeline and clinical input options.
line([(591,288),(1130,288)],C['muted'],1.2)
ax.add_patch(Circle((614,288),5,facecolor=C['blue'],edgecolor='white',lw=1))
ax.add_patch(Circle((1055,288),5,facecolor=C['coral'],edgecolor='white',lw=1))
text(614,256,'Before surgery',13,C['blue'],ha='center')
text(1055,256,'Operation end',13,C['coral'],ha='center')
for x,y,w,n,title,detail,color in [
 (560,326,190,'9','Routine\ninputs','Age, sex, laboratory\ntests and history',C['blue']),
 (768,326,190,'27','Preoperative\ninputs','Additional clinical\ninformation',C['blue']),
 (976,326,194,'38','Operation-end\ninputs','Preoperative +\nintraoperative data',C['coral'])]:
 box(x,y,w,178,'#F4F8FA',C['line'],13,1)
 text(x+17,y+13,n,30,color,weight='bold')
 text(x+17,y+67,title,12.5,weight='bold')
 text(x+17,y+118,detail,11.5,C['muted'])
arrow(750,413,765,413,C['muted'],scale=10);arrow(960,413,973,413,C['muted'],scale=10)
box(560,537,610,218,C['lightteal'],r=18)
text(584,558,'One acquisition-and-review budget',17,C['teal'],weight='bold')
clipboard(590,615,45,59,C['teal'])
text(655,621,'Collect more\ninformation',14,weight='bold')
text(850,634,'or',13,C['muted'],ha='center')
clinician(919,620,.65);clipboard(945,621,31,45,C['teal'])
text(997,621,'Review more\npatients',14,weight='bold')
text(584,711,'Information collection reduces the capacity left for review.',12,C['muted'])
# Clinical result summaries.
box(1210,261,535,234,'#F0F6F9',r=18)
text(1236,282,'9 → 27 inputs',21,C['blue'],weight='bold')
text(1236,326,'Expanded routine-input review',17,weight='bold')
text(1236,362,'Higher mean recorded-event capture\nacross five assumed cost distributions',13.8)
text(1236,443,'Principal 20% review-equivalent budget',11.5,C['muted'])
box(1210,520,535,235,'#FAF1EE',r=18)
text(1236,541,'27 → 38 inputs',21,C['coral'],weight='bold')
text(1236,585,'Operation-end information',17,weight='bold')
text(1236,623,'Resource value depended on\nassumed cost and evaluation setting',13.8)
text(1236,706,'Geographic and temporal results differed',11.5,C['muted'])
# Public validation band.
line([(55,789),(1745,789)],C['line'],1)
text(55,817,'PUBLIC-SOURCE EVALUATION',13,C['muted'],weight='bold')
text(55,855,'Do performance and review burden transfer?',17,weight='bold')
hospital(66,922)
text(156,915,'INSPIRE',17,C['blue'],weight='bold')
text(156,950,'6,043 records · 253 creatinine events',12.5,C['muted'])
text(665,915,'MIMIC-IV',17,C['blue'],weight='bold')
text(665,950,'759 patients · 59 creatinine events',12.5,C['muted'])
text(1168,855,'Transferred cutoffs changed review workload',16,weight='bold')
text(1168,909,'Source-20% cutoffs selected',12.5,C['muted'])
text(1168,941,'36.8% and 27.7%',23,C['blue'],weight='bold')
text(1168,985,'of MIMIC-IV patients',12.5,C['muted'])
# Footnote gives the precise inferential frame without a warning panel.
line([(55,1033),(1745,1033)],C['line'],1)
text(55,1056,'AKI, acute kidney injury. Comparisons used retrospective recorded-event capture and a shared resource pool.',10.5,C['muted'])
text(55,1080,'Cost distributions were analyst-specified. Clinical review depicts the intended use context.',10.5,C['muted'])
for ext in ['svg','pdf']:
 fig.savefig(O/f'GRAPHICAL_ABSTRACT_EN.{ext}',facecolor='white',bbox_inches=None)
fig.savefig(O/'GRAPHICAL_ABSTRACT_EN.png',dpi=300,facecolor='white')
fig.savefig(W/'review/GRAPHICAL_ABSTRACT_PREVIEW.png',dpi=100,facecolor='white')
plt.close(fig)
caption_en='Perioperative information and clinical review capacity after gastrointestinal cancer surgery. The clinical cohort comprised 3,710 operations at five Chinese centres with 205 recorded acute kidney injury events. Frozen systems used nine routine preoperative inputs, 27 preoperative inputs, or 38 operation-end inputs. Acquisition and record review shared a hypothetical resource budget. At a 20% review-equivalent budget, expanded nine-input review had higher mean recorded-event capture than acquisition of 27 inputs under all five examined analyst-specified cost distributions. Operation-end resource value depended on assumed costs and evaluation setting. Public evaluations used separate recorded creatinine endpoints in INSPIRE and MIMIC-IV. Source-20% cutoffs selected 36.8% and 27.7% of MIMIC-IV patients under the geographic and temporal source models. The clinical drawings depict the assessment context. The resource analyses evaluated retrospective capture and did not measure effects of implementing review.'
caption_zh='胃肠肿瘤围术期信息与临床复核容量。临床队列包含中国五中心3710次手术及205例记录性AKI。冻结系统分别使用九项常规术前输入、27项术前输入及38项术毕输入。在共同假设资源预算中，采集与复核共同占用资源。20%复核当量预算下，五种分析者指定成本分布中，扩大九项输入的复核名单均具有更高平均记录事件捕获；术毕信息的资源价值随假设成本及评价场景变化。INSPIRE和MIMIC-IV采用各自记录性肌酐结局，地理及时间源模型的20%切点在MIMIC-IV中选择36.8%和27.7%的患者。临床插图展示预期评估场景；资源分析测量回顾性事件捕获，尚未测量实施复核对患者结局的影响。'
(O/'FIGURE_LEGEND_EN_ZH.md').write_text('# Graphical abstract\n\n'+caption_en+'\n\n'+caption_zh+'\n')
(O/'README_ZH.md').write_text('# 临床场景图形摘要\n\n本图以胃肠肿瘤手术和围术期AKI风险评估为临床场景，展示信息需求、评估时点及复核容量的临床选择。英文SVG可编辑，PDF为矢量稿，PNG为300dpi。图像由可追溯Python绘图代码绘制，临床符号为原创建模示意，未调用外部图库或生成式图像模型。\n\n这是用户新增的独立配套图形摘要，现有五张主图及其编号保持。目标期刊是否使用图形摘要由实际投稿要求和编辑决定。\n')
mapping={'graphic_type':'clinical cohort graphical abstract','user_requested_clinical_scene':True,'title':'Perioperative information for AKI risk review','clinical_population':'Gastrointestinal cancer surgery','values':{'clinical_operations':3710,'clinical_events':205,'centres':5,'INSPIRE_records':6043,'INSPIRE_events':253,'MIMIC_patients':759,'MIMIC_events':59,'MIMIC_source20_burden_percent':[36.8,27.7]},'new_cost_analysis_acceptance_sha256':'8feffdb7623257510a3e3b05489d6bfdfd64329304b9f26796750c0899743819','cost_findings':'All five conditional cost laws; principal20%budget; recorded-event objective','illustration_provenance':'Original matplotlib paths and shapes; no external image assets or generative imagery','figure_library':'Skill read; matching callable library service tools unavailable this turn. No template was retrieved or materialized.','main_figures_renumbered':False}
(O/'SOURCE_BINDING.json').write_text(json.dumps(mapping,indent=2))
print(json.dumps({'exports':['GRAPHICAL_ABSTRACT_EN.svg','GRAPHICAL_ABSTRACT_EN.pdf','GRAPHICAL_ABSTRACT_EN.png'],'clinical_scene':True,'dpi':300,'vector_text_editable':True}))
