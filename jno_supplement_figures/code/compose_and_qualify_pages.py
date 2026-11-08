#!/usr/bin/env python3
"""Compose native Arial10pt panel SVG/PDFs without scaling; inspect every exported PDF."""
from pathlib import Path
import argparse,copy,datetime,hashlib,json,re,xml.etree.ElementTree as ET
import fitz
from PIL import Image,ImageDraw
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=ap.parse_args();R=args.root
j=json.loads((R/'PANEL_MANIFEST.json').read_text());assert j['figures_count']==18 and j['panels_count']==53
SVG='http://www.w3.org/2000/svg';XLINK='http://www.w3.org/1999/xlink';ET.register_namespace('',SVG);ET.register_namespace('xlink',XLINK)
pt=lambda mm:mm*72/25.4
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(s):return re.sub(r'\s+',' ',s).strip()
def inspect_pdf(path):
 d=fitz.open(path);fonts=[];sizes=[];texts=[];images=0;outside=[]
 for pg in d:
  texts.append(pg.get_text());images+=len(pg.get_images())
  for block in pg.get_text('dict')['blocks']:
   for line in block.get('lines',[]):
    for s in line.get('spans',[]):
     if not s.get('text','').strip():continue
     sizes.append(s['size']);fonts.append(s['font']);b=fitz.Rect(s['bbox'])
     if b.x0<-.7 or b.y0<-.7 or b.x1>pg.rect.width+.7 or b.y1>pg.rect.height+.7:outside.append(s['text'])
 assert sizes and min(sizes)>=9.98,(path,'font smaller than10',min(sizes))
 assert all('Arial' in s for s in fonts),(path,sorted(set(fonts)))
 assert not outside,(path,'text outside page',outside)
 out={'file':str(path.relative_to(R)),'pages':len(d),'width_mm':d[0].rect.width/72*25.4,'height_mm':d[0].rect.height/72*25.4,'minimum_text_pt':min(sizes),'fonts':sorted(set(fonts)),'raster_images':images,'text_sha256':hashlib.sha256(norm(' '.join(texts)).encode()).hexdigest(),'text':norm(' '.join(texts)),'sha256':sha(path)}
 d.close();return out
inspections=[]
for x in j['panels']:
 x['pdf_qualification']=inspect_pdf(R/x['pdf']);inspections.append(x['pdf_qualification'])
 root=ET.parse(R/x['svg']).getroot();text_elements=root.findall('.//{'+SVG+'}text');assert text_elements,(x['figure'],x['panel'],'SVG has no editable text');x['svg_editable_text_elements']=len(text_elements)
def compose_svg(items,path,height):
 root=ET.Element('{'+SVG+'}svg',{'width':'170mm','height':f'{height:.8g}mm','viewBox':f'0 0 {pt(170):.9f} {pt(height):.9f}','version':'1.1'})
 ET.SubElement(root,'{'+SVG+'}rect',{'x':'0','y':'0','width':str(pt(170)),'height':str(pt(height)),'fill':'white'})
 offset=0.
 for i,x in enumerate(items):
  inner=ET.parse(R/x['svg']).getroot();ids={el.attrib['id']:f'p{i}_{el.attrib["id"]}' for el in inner.iter() if 'id' in el.attrib}
  for el in inner.iter():
   for key,value in list(el.attrib.items()):
    if key=='id':el.set(key,ids[value])
    else:
     for old,new in ids.items():
      value=value.replace('url(#'+old+')','url(#'+new+')')
      if value=='#'+old:value='#'+new
     el.set(key,value)
  g=ET.SubElement(root,'{'+SVG+'}g',{'transform':f'translate(0,{pt(offset):.9f})','id':f'panel_{x["panel"]}'})
  for child in inner:
   if child.tag.endswith('metadata'):continue
   g.append(copy.deepcopy(child))
  offset+=x['height_mm']
 ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True);assert abs(offset-height)<1e-7
pages=[];figures=[]
for i in range(1,19):
 label=f'eFigure_{i}';panels=[x for x in j['panels'] if x['figure']==label];groups=[];current=[];h=0.
 for x in panels:
  if current and (len(current)==2 or h+x['height_mm']>180.00001):groups.append(current);current=[];h=0
  current.append(x);h+=x['height_mm']
 if current:groups.append(current)
 composite=fitz.open();figure_pages=[]
 for page_no,items in enumerate(groups,1):
  height=sum(x['height_mm'] for x in items);assert height<=220
  stem=R/'pages'/f'{label}_page_{page_no:02d}';svg=stem.with_suffix('.svg');pdf=stem.with_suffix('.pdf');png=stem.with_suffix('.png');compose_svg(items,svg,height)
  doc=fitz.open();pg=doc.new_page(width=pt(170),height=pt(height));offset=0.
  for x in items:
   src=fitz.open(R/x['pdf']);rect=fitz.Rect(0,pt(offset),pt(170),pt(offset+x['height_mm']));pg.show_pdf_page(rect,src,0,keep_proportion=True);src.close();offset+=x['height_mm']
  doc.set_metadata({'title':label+f' page{page_no}','subject':'Frozen aggregate data;Arial10pt native panels;no scaling','creator':'AKI analysis figure code'})
  doc.save(pdf,garbage=4,deflate=True);q=inspect_pdf(pdf);expected=' '.join(x['pdf_qualification']['text'] for x in items);assert norm(expected)==q['text'],(label,page_no,'Composite text altered')
  pix=pg.get_pixmap(matrix=fitz.Matrix(300/72,300/72),alpha=False);pix.set_dpi(300,300);pix.save(png);composite.insert_pdf(doc);doc.close()
  rec={'figure':label,'page':page_no,'svg':str(svg.relative_to(R)),'pdf':str(pdf.relative_to(R)),'png':str(png.relative_to(R)),'width_mm':170,'height_mm':height,'panels':[x['panel'] for x in items],'font_family':'Arial','minimum_text_pt':q['minimum_text_pt'],'fonts':q['fonts'],'pdf_sha256':sha(pdf),'svg_sha256':sha(svg),'png_sha256':sha(png),'text_content_unchanged':True,'raster_images':q['raster_images']}
  pages.append(rec);figure_pages.append(rec);inspections.append(q)
 stem=R/'composites'/label;composite.set_metadata({'title':label,'subject':'Source-preserving JNO supplemental figure','creator':'AKI analysis figure code'});composite.save(stem.with_suffix('.pdf'),garbage=4,deflate=True);composite.close();figures.append({'figure':label,'panels':[x['panel'] for x in panels],'pages':figure_pages,'composite_pdf':str(stem.with_suffix('.pdf').relative_to(R)),'composite_sha256':sha(stem.with_suffix('.pdf'))})
for sheet,start in enumerate(range(0,len(pages),4),1):
 group=pages[start:start+4];canvas=Image.new('RGB',(1440,1640),'#dddddd');draw=ImageDraw.Draw(canvas)
 for k,x in enumerate(group):
  im=Image.open(R/x['png']).convert('RGB');im.thumbnail((700,780));left=(k%2)*720+(720-im.width)//2;top=(k//2)*820+25;canvas.paste(im,(left,top));draw.text(((k%2)*720+10,(k//2)*820+7),x['figure']+f' page{x["page"]}',fill='black')
 canvas.save(R/'previews'/f'PAGE_CONTACT_{sheet:02d}.png')
for x in inspections:x.pop('text',None)
result={'status':'PASS_ACTUAL_ARIAL10_VECTOR_TEXT_AND_PAGE_COMPOSITION_PENDING_VISUAL_REVIEW','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'figures':figures,'pages':pages,'panels':j['panels'],'figures_count':18,'pages_count':len(pages),'panels_count':53,'all_native_panel_width_mm':170,'body_font_pt':10,'actual_font_families':sorted({f for q in inspections for f in q['fonts']}),'actual_minimum_text_pt':min(q['minimum_text_pt'] for q in inspections),'source_scripts':j['source_scripts'],'source_csvs':j['source_csvs'],'export_script_sha256':j['export_script_sha256'],'compose_script_sha256':sha(__file__),'new_statistics':0,'frozen_numerical_artists_and_intervals_unchanged':True,'source_hashes_unchanged':j['source_hashes_unchanged'],'font_files_bundled':False,'body_text_not_scaled':True,'vector_page_assembly_no_rasterization':True,'final_visual_review':'PENDING','contact_sheets':sheet}
(R/'PAGE_AND_PANEL_MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n');(R/'audit/PDF_AND_SVG_QUALIFICATION.json').write_text(json.dumps({'status':'PASS_ACTUAL_PDF_FONT_TEXT_AND_EDITABLE_SVG_CHECKS','pdf_files':len(inspections),'files':inspections,'font_files_bundled':False},indent=2)+'\n')
print(json.dumps({'status':result['status'],'figures':18,'panels':53,'pages':len(pages),'fonts':result['actual_font_families'],'minimum_text_pt':result['actual_minimum_text_pt'],'contact_sheets':sheet},indent=2))
