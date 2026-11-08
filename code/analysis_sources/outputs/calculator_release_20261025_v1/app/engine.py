"""One canonical inference API for CLI and local web. Original states stay private."""
from pathlib import Path
import os,sys,json,hashlib,threading
from collections import OrderedDict
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from input_validation import validate_records
APP=Path(__file__).resolve().parent
VERSION=json.loads((APP/"VERSION.json").read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def parse_json(text):
    def object_pairs(pairs):
        out={}
        for k,v in pairs:
            if k in out:raise ValueError("Duplicate JSON field")
            out[k]=v
        return out
    def invalid_constant(x):raise ValueError("Non-finite JSON number")
    return json.loads(text,object_pairs_hook=object_pairs,parse_constant=invalid_constant)
class Calculator:
    def __init__(self,root=None):
        self.root=Path(root or os.environ.get("AKI_PROJECT_ROOT",APP.parents[2])).resolve()
        self.catalog=json.loads((APP/"catalog.json").read_text())
        self.cache=OrderedDict();self.lock=threading.RLock()
        for name in ["v2_stable_imputation_20261017","python_six_outcomes_20261017","ensemble_calibration_20261020_v1"]:
            sys.path.insert(0,str(self.root/"work"/name))
        # Import only required saved-state classes and the frozen calibration function.
        import stable_imputer
        from ensemble_math import apply_calibration
        self.apply_calibration=apply_calibration
    def resolve_source(self,path):
        p=Path(path)
        if p.is_absolute():
            try:parts=p.parts;p=Path(*parts[parts.index("outputs"):])
            except ValueError:raise ValueError("Unrecognized source-state location") from None
        if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0]!="outputs":
            raise ValueError("Invalid source-state location")
        target=(self.root/p).resolve()
        physical=(self.root/"outputs").resolve()
        if not (target.is_relative_to(self.root) or target.is_relative_to(physical.resolve())):
            raise ValueError("Invalid source-state location")
        return target
    def _load(self,route):
        if route not in self.catalog:raise ValueError("Unknown model route")
        if route in self.cache:
            q=self.cache.pop(route);self.cache[route]=q;return q
        c=self.catalog[route];p=self.resolve_source(c["manifest"])
        if not p.is_file():raise ValueError("Private model states are required for prediction")
        if sha(p)!=c["manifest_sha256"]:raise ValueError("Model manifest checksum mismatch")
        m=json.loads(p.read_text());b={}
        for f,ref in m["base_components"].items():
            q=self.resolve_source(ref["path"])
            if sha(q)!=ref["sha256"]:raise ValueError("Source component checksum mismatch")
            b[f]=joblib.load(q)
        r=m["calibrator"];q=self.resolve_source(r["path"])
        if sha(q)!=r["sha256"]:raise ValueError("Calibration checksum mismatch")
        cal=joblib.load(q);meta=None
        if m.get("meta"):
            r=m["meta"];q=self.resolve_source(r["path"])
            if sha(q)!=r["sha256"]:raise ValueError("Ensemble checksum mismatch")
            meta=joblib.load(q)
        obj=(m,b,cal,meta);self.cache[route]=obj
        while len(self.cache)>2:self.cache.popitem(last=False)
        return obj
    def public_catalog(self):
        return {"version":VERSION["calculator_version"],"routes":{k:{x:v[x] for x in ["label","identity","fields","thresholds"]} for k,v in self.catalog.items()}}
    def validate(self,payload):
        if not isinstance(payload,dict) or set(payload)!={"route","records","units"}:raise ValueError("Expected route, records and units")
        route=payload["route"]
        if not isinstance(route,str) or route not in self.catalog:raise ValueError("Unknown model route")
        return route,validate_records(payload["records"],self.catalog[route]["fields"],payload["units"])
    @staticmethod
    def _canonical_pipeline(pipe,filled):
        outputs=[]
        for start in range(0,len(filled),100):
            z=filled.iloc[start:start+100].reset_index(drop=True);n=len(z)
            if n<100:z=pd.concat([z,pd.concat([z.iloc[-1:]]*(100-n),ignore_index=True)],ignore_index=True)
            outputs.append(np.asarray(pipe.predict_proba(z)[:n,1],dtype=np.float64))
        return np.concatenate(outputs)
    def probabilities(self,route,records):
        """Validated records only; frozen imputation and canonical100 inference."""
        m,b,cal,meta=self._load(route);keys=[r["case_key"] for r in records]
        x=pd.DataFrame([{k:v for k,v in r.items() if k!="case_key"} for r in records])
        fields=self.catalog[route]["fields"]
        for f in fields:
            if f["type"]=="number":x[f["name"]]=pd.to_numeric(x[f["name"]],errors="raise")
            else:x[f["name"]]=pd.array(x[f["name"]],dtype="string")
        filled={};values={}
        for family,bundle in b.items():
            sig=bundle["imputer_sha256"]
            if sig not in filled:filled[sig]=bundle["imputer"].transform(x[bundle["input_columns"]],keys)
            values[family]=self._canonical_pipeline(bundle["pipeline"],filled[sig])
        if m["family"]=="Voting":raw=np.column_stack([values[f] for f in m["members"]]).mean(axis=1)
        elif m["family"]=="Stacking":raw=self._canonical_pipeline(meta,pd.DataFrame({f:values[f] for f in m["members"]}))
        else:raw=values[m["family"]]
        prob=self.apply_calibration(cal,np.asarray(raw,dtype=np.float64))
        if not np.isfinite(prob).all() or ((prob<0)|(prob>1)).any():raise ValueError("Invalid source prediction")
        return raw,prob
    def predict(self,payload):
        route,records=self.validate(payload)
        with self.lock,threadpool_limits(limits=1):
            raw,prob=self.probabilities(route,records)
        cuts=self.catalog[route]["thresholds"];rows=[]
        for i,r in enumerate(records):
            missing=[f["name"] for f in self.catalog[route]["fields"] if r[f["name"]] is None]
            rows.append({"case_key":r["case_key"],"risk":float(prob[i]),"raw_risk":float(raw[i]),
              "alerts":{k:bool(prob[i]>=cuts[k]) for k in ["training_sensitivity80","training_capacity10","training_capacity20"]},
              "missing_fields":missing})
        return {"version":VERSION["calculator_version"],"route":route,"thresholds":cuts,"predictions":rows}
