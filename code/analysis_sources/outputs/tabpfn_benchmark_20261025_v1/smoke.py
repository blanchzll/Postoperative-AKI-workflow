import os
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json, time, pathlib, resource, socket
import numpy as np,torch,tabpfn
from sklearn.datasets import make_classification
from tabpfn import TabPFNClassifier
OUT=pathlib.Path(__file__).resolve().parent
w=json.loads((OUT/'WEIGHT_RECEIPT.json').read_text())
def denied(*a,**kw):raise RuntimeError('Network disabled for model computation')
socket.socket.connect=denied
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
x,y=make_classification(n_samples=160,n_features=9,n_informative=5,random_state=42)
x[::11,0]=np.nan
m=TabPFNClassifier(n_estimators=8,model_path=w['download_path'],device='cuda',inference_precision=torch.float32,random_state=42,n_jobs=1)
t=time.time();m.fit(x[:128],y[:128]);p=m.predict_proba(x[128:]);q=m.predict_proba(x[128:][::-1])[::-1]
assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
assert np.max(abs(p-q))<1e-5
r={'status':'PASS_SYNTHETIC_ONLY','n_train':128,'n_test':32,'n_features':9,'seconds':time.time()-t,'gpu_peak_GiB':torch.cuda.max_memory_allocated()/2**30,'RSS_peak_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'reverse_max_abs_difference':float(np.max(abs(p-q))),'patient_data_used':False,'network_disabled':True,'model_sha256':w['sha256']}
(OUT/'SMOKE.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
