"""Build calculator source/schema against current accepted locks; never copy old predictions."""
from pathlib import Path
import json,hashlib,shutil,datetime,ast,csv,copy,os
R=Path(__file__).resolve().parents[2];P=R.parents[1];O=R/"outputs/calculator_release_20261025_v1";A=O/"app"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n")
def main():
 assert not A.exists()
 A.mkdir(parents=True,mode=0o700);old=P/"outputs/calculator_release_20261025_v1/app"
 originals={}
 for name in ["engine.py","input_validation.py","cli.py","web.py","index.html"]:
  src=old/name;dst=A/name;s=src.read_text()
  if name=="engine.py":
   assert s.count('physical=Path("/data/codex-workspaces")/self.root.name')==1
   s=s.replace('physical=Path("/data/codex-workspaces")/self.root.name','physical=(self.root/"outputs").resolve()')
  dst.write_text(s);originals[name]={"source_sha256":sha(src),"current_sha256":sha(dst)}
 # Preserve all prediction, imputation, calibration, numeric block, validation and alert equations.
 oldtree=ast.parse((old/"engine.py").read_text());newtree=ast.parse((A/"engine.py").read_text())
 def methods(tree):
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="Calculator")
  return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}
 left,right=methods(oldtree),methods(newtree)
 for key in left:
  if key!="resolve_source":assert left[key]==right[key],key
 fields={}
 old_catalog=read(old/"catalog.json")
 for item in old_catalog.values():
  for field in item["fields"]:
   if field["name"] not in fields:fields[field["name"]]=copy.deepcopy(field)
 spec=read(R/"outputs/v4_development_inputs_20261018/FEATURE_SPECS.json")
 locks=read(R/"outputs/model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json")
 catalog={};dictionary=[];sources={}
 for ref in locks["models"]:
  if not ref["primary_route_winner"]:continue
  assert sha(ref["path"])==ref["sha256"]
  m=read(ref["path"]);i=m["identity"];rid="_".join(i[k] for k in ["line","cohort","stage","branch"])
  assert rid in old_catalog
  schema=[]
  for name in m["input_columns"]:
   f=copy.deepcopy(fields[name])
   if name in spec[i["stage"]]["cat"]:
    assert f["type"]=="category";f["allowed"]=spec[i["stage"]]["cat_levels"][name]
    f["labels"]={v:f.get("labels",{}).get(v,"源编码"+v) for v in f["allowed"]}
   else:assert f["type"]=="number" and name in spec[i["stage"]]["num"]
   schema.append(f);dictionary.append({"route":rid,**f})
  catalog[rid]={"label":({"geography":"地理开发","temporal":"时间开发"}[i["line"]]+" · "+{"preop":"术前","endop":"术毕"}[i["stage"]]+" · "+str(len(schema))+"项"+{"compact":"紧凑","full":"完整","selected":"训练内筛选流程"}[i["branch"]]+" · "+{"preop":"主队列","paired":"配对队列"}[i["cohort"]]),"manifest":str(Path(ref["path"]).relative_to(R)),"manifest_sha256":ref["sha256"],"identity":i,"family":m["family"],"fields":schema,"thresholds":m["training_thresholds"]}
  sources[ref["path"]]=ref["sha256"]
 assert len(catalog)==10
 save(A/"catalog.json",catalog)
 save(A/"VERSION.json",{"calculator_version":"AKI研究版2026.10.29","current_input_sha256":sha(R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json"),"models_refitted_by_calculator":False,"inference":"Original keyed imputation, canonical100 numeric block, original model/source calibration and thresholds","clinical_main":[3710,205],"paired":[2428,138]})
 examples=O/"synthetic_examples";examples.mkdir(mode=0o700)
 for rid,cat in catalog.items():
  source=read(P/"outputs/calculator_release_20261025_v1/synthetic_examples"/(rid+".json"))
  original=source["records"][0];rec={"case_key":"DEMO_current_001"}
  for f in cat["fields"]:
   assert f["name"] in original,"No silently generated missing clinical field"
   rec[f["name"]]=original[f["name"]]
   if rec[f["name"]] is not None and f["type"]=="category":assert rec[f["name"]] in f["allowed"]
  save(examples/(rid+".json"),{"route":rid,"units":{f["name"]:f["unit"] for f in cat["fields"] if f["unit"]},"records":[rec]})
 with (O/"VARIABLE_DICTIONARY.csv").open("w",newline="") as stream:
  writer=csv.DictWriter(stream,fieldnames=["route","name","label","unit","type","lower","upper","allowed","labels"]);writer.writeheader();writer.writerows(dictionary)
 save(O/"CURRENT_PREPARATION_ACCEPTANCE.json",{"status":"PASS_CURRENT_SOURCE_CALCULATOR_CODE_AND_CATALOG_PREPARATION_ONLY","time":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),"source_lock_sha256":sha(R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json"),"all_numeric_prediction_API_function_AST_unchanged":True,"execution_only_path_repair":"Allow the current run outputs symlink's actual resolved /data directory instead of assuming physical directory=self.root.name; copied private package still resolves under its own root.","source_code":originals,"model_manifests":sources,"catalog_sha256":sha(A/"catalog.json"),"synthetic_examples_from_old_explicit_synthetic_values_only":True,"clinical_or_engineering_qualification_run":False,"final_scientific_review_run":False})
 print("Prepared current10-route calculator; numeric API unchanged; no qualification results copied")
if __name__=="__main__":main()
