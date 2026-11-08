"""Fetch a public pinned NGC candidate archive only. Never load weights, execute pickle, or change the selected clinical models."""
from pathlib import Path,PurePosixPath
from concurrent.futures import ThreadPoolExecutor,as_completed
import datetime,fcntl,hashlib,json,logging,os,shutil,tarfile,threading,time
import requests
R=Path(__file__).resolve().parents[2]
O=R/"outputs/llm_supplement_20261028_v1"
T=O/"models/llama3_8b_instruct_ngc_v1"
D=R/"model_assets/llama3_8b_instruct_ngc_v1/1.0"
NAME="8b_instruct_nemo_bf16.nemo"
SIZE=16061388800
API="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/llama-3-8b-instruct-nemo/versions/1.0"
URL=API+"/files/"+NAME
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+".tmp");t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n");os.replace(t,p)
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
    return h.hexdigest()
def main():
    logging.disable(logging.CRITICAL)
    T.mkdir(parents=True,exist_ok=True,mode=0o700)
    own=(T/"CANDIDATE_OWNER.lock").open("a");fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    acc=T/"CANDIDATE_DOWNLOAD_ACCEPTANCE.json"
    if acc.exists():
        d=json.loads(acc.read_text());p=Path(d["archive_path"])
        assert p.stat().st_size==SIZE and sha(p)==d["archive_sha256"]
        print("ALREADY_VERIFIED_NGC_CANDIDATE_NO_REDOWNLOAD",flush=True);return
    save(T/"STATUS.json",{"status":"WAITING_FOR_EXISTING_MODEL_ASSET_OWNER","PID":os.getpid(),"time":now(),"training_started":False})
    guard=(O/"MODEL_ASSET_OWNER.lock").open("a")
    fcntl.flock(guard,fcntl.LOCK_EX)
    assert os.path.ismount("/data") and str(D.resolve()).startswith("/data/codex-workspaces/aki-prediction-manuscript/")
    D.mkdir(parents=True,exist_ok=True,mode=0o700)
    s=json.loads((R/"outputs/common6_strict_continuation_20261024_v1/status.json").read_text())
    available=int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:")))*1024
    remaining=available-s["growth_reserved_bytes"]-max(4*1024**3,int(.25*available))
    if remaining<4*1024**3:
        save(T/"STATUS.json",{"status":"WAITING_FOR_SAFE_DOWNLOAD_RESOURCES","time":now(),"remaining_after_main_bytes":remaining,"training_started":False});return
    assert shutil.disk_usage(D).free>2*SIZE+10*1024**3
    with requests.get(API,headers={"Accept":"application/json"},timeout=(15,30)) as response:
        response.raise_for_status();m=response.json()
    assert m["model"]["isPublic"] and m["model"]["canGuestDownload"]
    assert m["modelVersion"]["versionId"]=="1.0" and m["modelVersion"]["totalSizeInBytes"]==SIZE
    probe=json.loads((O/"references/NGC_LLAMA3_FILE_ACCESS_PROBE.json").read_text())
    etag=next(v for k,v in probe["response_headers"].items() if k.lower()=="etag")
    def resolve():
        with requests.get(URL,headers={"Range":"bytes=0-0","If-Match":etag,"Accept-Encoding":"identity","Connection":"close"},stream=True,timeout=(15,30)) as response:
            response.raise_for_status();assert response.status_code==206
            assert response.headers.get("Content-Range")==f"bytes 0-0/{SIZE}"
            assert response.headers.get("ETag")==etag
            assert len(response.raw.read(2))==1
            return response.url
    location=resolve()
    piece_bytes=64*1024**2
    work=D/"range_parts";work.mkdir(parents=True,exist_ok=True,mode=0o700)
    contract={"source":URL,"NGC_version":"1.0","expected_bytes":SIZE,"provider_ETag":etag,"piece_bytes":piece_bytes,"workers":4,"provider_full_file_SHA256_available":False}
    cp=work/"SOURCE_CONTRACT.json"
    if cp.exists():assert json.loads(cp.read_text())==contract
    else:save(cp,contract)
    save(T/"STATUS.json",{"status":"RECOVERING_PUBLIC_NGC_CANDIDATE_WITH_SHORT_REQUESTS","PID":os.getpid(),"time":now(),"expected_bytes":SIZE,"provider_ETag":etag,"piece_bytes":piece_bytes,"workers":4,"HTTP_request_bytes":8*1024**2,"training_started":False,"selected_HF_checkpoint_changed":False})
    stop=threading.Event();deadline=time.monotonic()+7200
    def get_piece(bounds):
        a,b=bounds;p=work/(str(a)+"_"+str(b)+".part");rpath=p.with_suffix(".COMPLETE.json")
        if rpath.exists():
            r=json.loads(rpath.read_text());assert p.stat().st_size==r["bytes"] and sha(p)==r["sha256"];return r
        loc=location
        logical_size=b-a+1
        while (p.stat().st_size if p.exists() else 0)<logical_size:
            for attempt in range(2):
                if stop.is_set():raise RuntimeError("Owned transfer cancelled")
                assert time.monotonic()<deadline,"Finite transfer budget reached"
                have=p.stat().st_size if p.exists() else 0
                assert have<logical_size
                begin=a+have;end=min(begin+8*1024**2-1,b)
                response=None;started=time.monotonic();header={}
                try:
                    response=requests.get(loc,headers={"Range":f"bytes={begin}-{end}","If-Match":etag,"Accept-Encoding":"identity","Connection":"close"},stream=True,timeout=(15,30))
                    header={"http_status":response.status_code,"content_range":response.headers.get("Content-Range"),"content_length":response.headers.get("Content-Length")}
                    response.raise_for_status();assert response.status_code==206
                    assert response.headers.get("ETag")==etag
                    assert response.headers.get("Content-Range")==f"bytes {begin}-{end}/{SIZE}"
                    assert int(response.headers["Content-Length"])==end-begin+1
                    with p.open("ab") as f:
                        for data in response.iter_content(chunk_size=256*1024):
                            if stop.is_set():raise RuntimeError("Owned transfer cancelled")
                            assert f.tell()+len(data)<=end-a+1;f.write(data)
                        f.flush();os.fsync(f.fileno())
                    assert p.stat().st_size==end-a+1;break
                except Exception as e:
                    save(work/(str(a)+"_"+str(b)+".SHORT_RANGE_"+str(begin)+"_ATTEMPT_"+str(attempt+1)+".json"),{"time":now(),"error_class":type(e).__name__,"http_status":getattr(getattr(e,"response",None),"status_code",None),"requested_start":begin,"requested_end":end,"retained_bytes":p.stat().st_size if p.exists() else 0,"elapsed_seconds":round(time.monotonic()-started,3),"response_headers":header,"signed_URL_saved":False})
                    if attempt==1 or stop.is_set():raise
                    time.sleep(2);loc=resolve()
                finally:
                    if response is not None:response.close()
        assert p.is_file() and p.stat().st_size==b-a+1
        r={"source_start":a,"source_end":b,"bytes":b-a+1,"path":str(p),"sha256":sha(p),"time":now()}
        save(rpath,r);return r
    tasks=[(a,min(a+piece_bytes-1,SIZE-1)) for a in range(0,SIZE,piece_bytes)]
    done=[];pool=ThreadPoolExecutor(max_workers=4,thread_name_prefix="owned-ngc-candidate")
    futures=[pool.submit(get_piece,t) for t in tasks]
    try:
        for future in as_completed(futures):
            done.append(future.result())
            save(T/"DOWNLOAD_PROGRESS.json",{"time":now(),"pieces_complete":len(done),"pieces_total":len(tasks),"completed_bytes":sum(r["bytes"] for r in done),"expected_bytes":SIZE,"provider_ETag":etag,"training_started":False})
    except BaseException:
        stop.set()
        for f in futures:f.cancel()
        pool.shutdown(wait=True,cancel_futures=True);raise
    else:pool.shutdown(wait=True)
    out=D/NAME;assembling=D/(NAME+".assembling")
    with assembling.open("wb") as f:
        for r in sorted(done,key=lambda r:r["source_start"]):
            p=Path(r["path"]);assert p.stat().st_size==r["bytes"] and sha(p)==r["sha256"]
            assert f.tell()==r["source_start"]
            with p.open("rb") as stream:
                for b in iter(lambda:stream.read(4*1024**2),b""):f.write(b)
        f.flush();os.fsync(f.fileno())
    assert assembling.stat().st_size==SIZE
    digest=sha(assembling)
    index=[];tokenizer=[];metadata=[];metadata_dir=T/"archive_metadata";metadata_dir.mkdir(exist_ok=True)
    with tarfile.open(assembling,"r:") as archive:
        for member in archive:
            path=PurePosixPath(member.name)
            assert not path.is_absolute() and ".." not in path.parts
            index.append({"name":member.name,"bytes":member.size,"type":member.type.decode("ascii","replace")})
            if any(t in member.name.lower() for t in ["tokenizer","tiktoken","merges.txt","vocab.json"]):tokenizer.append(member.name)
            selected=path.name in ["model_config.yaml","metadata.json"] or path.name==".zarray" and any(t in member.name for t in ["final_layernorm","linear_qkv.weight","word_embeddings.weight"])
            if member.isfile() and member.size<=256*1024 and selected:
                stream=archive.extractfile(member);assert stream is not None
                payload=stream.read();assert len(payload)==member.size
                target=metadata_dir/(hashlib.sha256(member.name.encode()).hexdigest()[:12]+"_"+path.name)
                target.write_bytes(payload);metadata.append({"member":member.name,"path":str(target),"sha256":hashlib.sha256(payload).hexdigest()})
    save(T/"ARCHIVE_FILE_INDEX.json",{"complete_archive_listing":True,"entries":index,"tokenizer_named_members":tokenizer,"selected_metadata":metadata,"pickle_or_weights_executed":False})
    assert not out.exists()
    os.replace(assembling,out)
    rec={"status":"PASS_NGC_VERSIONED_ARCHIVE_BYTES_ETAG_LOCAL_SHA256_AND_NONEXECUTING_TAR_INSPECTION","time":now(),"NGC_model":"nvidia/nemo/llama-3-8b-instruct-nemo:1.0","canonical_source":URL,"archive_path":str(out),"archive_bytes":SIZE,"archive_sha256":digest,"provider_ETag":etag,"provider_full_file_SHA256_available":False,"source_contract_sha256":sha(cp),"archive_index_sha256":sha(T/"ARCHIVE_FILE_INDEX.json"),"tokenizer_named_members":tokenizer,"exact_tensor_identity_with_selected_HF_commit":"UNVERIFIED","conversion_completed":False,"GPU_training_qualified":False,"formal_training_started":False,"selected_HF_checkpoint_changed":False,"clinical_data_or_main_models_changed":False,"license":"Meta Llama3 Community License per official NVIDIA catalog","script_sha256":sha(Path(__file__))}
    save(acc,rec)
    save(T/"STATUS.json",{"status":"PUBLIC_NGC_CANDIDATE_ARCHIVE_VERIFIED_CONVERSION_AND_TOKENIZER_QUALIFICATION_PENDING","time":now(),"download_receipt_sha256":sha(acc),"training_started":False})
    for name in ["CANDIDATE_DOWNLOAD_FAILURE.json","CANDIDATE_DOWNLOAD_FAILURE_V2.json"]:
        old=T/name
        if old.exists():
            dest=T/"transfer_repair"/("RESOLVED_"+name)
            assert not dest.exists()
            old.rename(dest)
    print("PASS_PUBLIC_NGC_CANDIDATE_ARCHIVE_VERIFIED_NOT_A_TRAINING_ACCEPTANCE",flush=True)
if __name__=="__main__":
    try:main()
    except BaseException as e:
        save(T/"CANDIDATE_DOWNLOAD_FAILURE_V2.json",{"time":now(),"error_class":type(e).__name__,"http_status":getattr(getattr(e,"response",None),"status_code",None),"partial_data_preserved":True,"signed_URL_or_credentials_saved":False,"training_started":False})
        save(T/"STATUS.json",{"status":"NGC_CANDIDATE_DOWNLOAD_STOPPED_FOR_DIAGNOSIS","time":now(),"error_class":type(e).__name__,"training_started":False})
        raise SystemExit("NGC candidate retrieval stopped: "+type(e).__name__+"; inspect sanitized receipt")
