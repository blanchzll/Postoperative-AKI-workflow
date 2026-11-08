"""Finite pinned official MedGemma download. No model load, training or patient inference."""
from pathlib import Path
import datetime,fcntl,hashlib,json,logging,os,shutil,subprocess,time
from huggingface_hub import HfApi,snapshot_download
from huggingface_hub.utils import disable_progress_bars
R=Path(__file__).resolve().parents[2]
O=R/"outputs/llm_supplement_20261028_v1"
MODEL_ID="medgemma_4b_it"
REPO="google/medgemma-1.5-4b-it"
REVISION="91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
TARGET=O/"models"/MODEL_ID
class Deferred(Exception): pass
def stamp(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(path,obj):
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
    os.replace(tmp,path)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""): h.update(block)
    return h.hexdigest()
def main():
    TARGET.mkdir(parents=True,exist_ok=True)
    guard=(O/"MODEL_ASSET_OWNER.lock").open("a")
    try: fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise Deferred("existing_model_asset_owner")
    accepted=TARGET/"MODEL_DOWNLOAD_ACCEPTANCE.json"
    if accepted.exists():
        receipt=json.loads(accepted.read_text())
        assert receipt["repository"]==REPO and receipt["revision"]==REVISION
        assert all(Path(p).is_file() and sha(Path(p))==digest for p,digest in receipt["files"].items())
        print("ALREADY_VERIFIED_MEDGEMMA_ASSETS_NO_DOWNLOAD",flush=True)
        return
    selection=json.loads((O/"MODEL_SELECTION.json").read_text())
    selected=next(m for m in selection["models"] if m["id"]==MODEL_ID)
    assert selected["repository"]==REPO and selected["weight_revision"]==REVISION
    access=json.loads((O/"SOURCE_ACCESS_STATUS.json").read_text())
    access=next(m for m in access["models"] if m["id"]==MODEL_ID)
    assert access["revision"]==REVISION and access.get("config_authorized_access")=="PASS"
    root=R/"model_assets"
    assert os.path.ismount("/data") and str(root.resolve()).startswith("/data/codex-workspaces/aki-prediction-manuscript/")
    status_path=R/"outputs/common6_strict_continuation_20261024_v1/status.json"
    status=json.loads(status_path.read_text())
    if time.time()-status_path.stat().st_mtime>180: raise Deferred("main_resource_status_stale")
    available=int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:")))*1024
    growth=int(status["growth_reserved_bytes"])
    os_headroom=max(4*1024**3,int(available*.25))
    bounded_download_budget=1536*1024**2
    if available-growth-os_headroom<bounded_download_budget: raise Deferred("main_growth_and_OS_headroom")
    main=subprocess.run(["systemctl","--user","show","aki-postslot-full-reanalysis-20261025.service","--property=MainPID,ActiveState"],capture_output=True,text=True,check=True)
    assert "ActiveState=active" in main.stdout and "MainPID=0" not in main.stdout
    info=HfApi(endpoint="https://huggingface.co").model_info(REPO,revision=REVISION,files_metadata=True,timeout=30)
    assert info.sha==REVISION
    weights=[s for s in info.siblings if s.rfilename.endswith(".safetensors")]
    assert {s.rfilename for s in weights}==set(access["official_weight_files"])
    for s in weights:
        lfs=getattr(s,"lfs",None)
        provider=lfs.sha256 if hasattr(lfs,"sha256") else lfs["sha256"]
        assert provider==access["official_weight_files"][s.rfilename]["provider_SHA256"]
        assert s.size==access["official_weight_files"][s.rfilename]["bytes"]
    support=[s for s in info.siblings if s.rfilename.endswith((".json",".txt",".model",".tiktoken",".jinja")) or s.rfilename in ["README.md","LICENSE"]]
    files={s.rfilename:s for s in weights+support}
    assert all(s.size is not None for s in files.values())
    expected_bytes=sum(s.size for s in files.values())
    dest=root/MODEL_ID/REVISION
    dest.mkdir(parents=True,exist_ok=True,mode=0o700)
    assert shutil.disk_usage(dest).free>2*expected_bytes+10*1024**3
    save(TARGET/"STATUS.json",{"status":"DOWNLOADING_OFFICIAL_PINNED_MODEL_ASSETS","PID":os.getpid(),"revision":REVISION,"expected_bytes":expected_bytes,"time":stamp(),"training_started":False,"available_bytes_at_admission":available,"main_growth_reserved_bytes":growth,"OS_headroom_bytes":os_headroom,"download_memory_budget_bytes":bounded_download_budget,"GPU_model_loading_performed":False})
    logging.disable(logging.CRITICAL)
    disable_progress_bars()
    snapshot_download(repo_id=REPO,revision=REVISION,local_dir=str(dest),max_workers=4,allow_patterns=list(files))
    manifest={}
    provider={}
    for name,s in files.items():
        p=dest/name
        assert p.is_file() and p.stat().st_size==s.size,name
        digest=sha(p)
        manifest[str(p)]=digest
        lfs=getattr(s,"lfs",None)
        if lfs:
            expected=lfs.sha256 if hasattr(lfs,"sha256") else lfs["sha256"]
            assert digest==expected,name
            provider[name]=expected
    assert all(s.rfilename in provider for s in weights)
    rec={"status":"PASS_official_pinned_MedGemma_asset_bytes_provider_SHA256","repository":REPO,"revision":REVISION,"local_path":str(dest),"files":manifest,"provider_weight_SHA256":{s.rfilename:provider[s.rfilename] for s in weights},"total_bytes":expected_bytes,"time":stamp(),"actual_GPU_training_qualified":False,"formal_training_started":False,"scientific_comparison_complete":False,"script_sha256":sha(Path(__file__)),"future_loading_boundary":"Pinned local files only; trust_remote_code=False; same clinical text only; qualify P40 supported compute and LoRA training/save-reload before scientific fits. Do not copy BF16 or push_to_hub defaults."}
    save(accepted,rec)
    save(TARGET/"STATUS.json",{"status":"OFFICIAL_ASSETS_VERIFIED_GPU_TRAINING_QUALIFICATION_PENDING","time":stamp(),"download_acceptance_sha256":sha(accepted),"training_started":False})
    print("PASS_MEDGEMMA_OFFICIAL_ASSETS_VERIFIED_GPU_QUALIFICATION_PENDING",flush=True)
if __name__=="__main__":
    try: main()
    except Deferred as e:
        save(TARGET/"STATUS.json",{"status":"WAITING_RESOURCE_OR_EXISTING_OWNER","reason":str(e),"time":stamp(),"training_started":False})
        print("MEDGEMMA_ASSETS_DEFERRED_WITHOUT_ALTERING_MAIN",flush=True)
    except BaseException as e:
        response=getattr(e,"response",None)
        save(TARGET/"DOWNLOAD_FAILURE.json",{"time":stamp(),"error_class":type(e).__name__,"http_status":getattr(response,"status_code",None),"partial_downloads_preserved":True,"do_not_store_signed_URL_or_credentials":True,"clinical_training_started":False})
        raise SystemExit("MedGemma asset retrieval stopped: "+type(e).__name__+"; see sanitized DOWNLOAD_FAILURE.json")
