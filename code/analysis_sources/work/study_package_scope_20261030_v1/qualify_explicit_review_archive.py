"""Bounded nonclinical tests of the explicit archive engine; no study packaging."""
from pathlib import Path
import copy, hashlib, importlib.util, json, os, resource, shutil, signal
import stat, sys, tempfile, time, traceback, warnings, zipfile

R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
W=R/"work/study_package_scope_20261030_v1"
O=R/"outputs/study_package_scope_20261030_v1/explicit_archive_engine_qualification_v1"
ENGINE_SHA="9e6ccfeaf435bb40d8a6e6b7d7e220014ec22e1357f2202b03ebe1bd745fdc07"
EVIDENCE=hashlib.sha256(b"NONCLINICAL ENGINEERING FIXTURE; NOT A STUDY CONTENT REVIEW").hexdigest()

def h(data):return hashlib.sha256(data).hexdigest()
def sha(p):return h(Path(p).read_bytes())
def save(p,x):
    with Path(p).open("x") as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+"\n");f.flush();os.fsync(f.fileno())

def main():
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_AS,(128*1024**2,)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(90,)*2)
    signal.alarm(180)
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    if O.exists():raise RuntimeError("Inspect previous qualification; no overwrite")
    O.mkdir(mode=0o700)
    start=time.monotonic();scratch=None;cases=[]
    try:
        assert sha(W/"explicit_review_archive.py")==ENGINE_SHA
        spec=importlib.util.spec_from_file_location("_tested_explicit_archive",W/"explicit_review_archive.py")
        engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
        protected=json.loads((R/"outputs/post_reanalysis_process_review_v1/MANUSCRIPT_PROTECTION.json").read_text())["files"]
        assert len(protected)==92 and all(sha(p)==v["sha256"] for p,v in protected.items())
        scratch=Path(tempfile.mkdtemp(prefix="owned-nonclinical-archive-",dir=O))
        root=scratch/"source";root.mkdir()
        content={"review/summary.txt":b"Nonclinical archive fixture\n",
                 "review/图说明.txt":"非临床文件完整性测试\n".encode(),
                 "review/not_executed.py":b"raise RuntimeError('archive contents must never execute')\n",
                 "private/opaque.bin":bytes(range(256))*16}
        for name,data in content.items():
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        full=[{"source_relative":name,"archive_path":name,"bytes":len(data),"sha256":h(data),
               "classification":"private_only" if name.startswith("private/") else "review_shareable"}
              for name,data in content.items()]
        shared=[v for v in full if v["classification"]=="review_shareable"]
        records={}
        for kind,selection in [("shared_review",shared),("private_review",full)]:
            archive=scratch/(kind+".zip")
            rec=engine.create_archive(root,[root],selection,archive,kind,EVIDENCE)
            extracted=scratch/(kind+"_clean")
            verified=engine.extract_verified(archive,extracted,rec["sha256"],kind,EVIDENCE,
                                             rec["payload_uncompressed_bytes"])
            assert verified["payload_members"]==len(selection)
            assert {str(p.relative_to(extracted)) for p in extracted.rglob("*") if p.is_file()}=={
                v["archive_path"] for v in selection}|{engine.MANIFEST}
            for v in selection:
                assert (extracted/v["archive_path"]).read_bytes()==content[v["source_relative"]]
            records[kind]=rec
            cases.append({"case":kind+"_actual_roundtrip","result":"PASS","payloads":len(selection),
                          "CRC_checked":True,"bytes_and_sha_independently_compared":True})
        serial=0
        def rejects(name,action):
            nonlocal serial
            serial+=1
            try:action()
            except (ValueError,FileExistsError,zipfile.BadZipFile) as e:
                cases.append({"case":name,"result":"EXPECTED_REJECTION","exception":type(e).__name__})
                return
            raise AssertionError("Invalid archive/input was accepted: "+name)
        def target():return scratch/("rejected_create_"+str(serial)+".zip")
        rejects("private_payload_in_shared_archive",lambda:engine.create_archive(root,[root],full,target(),"shared_review",EVIDENCE))
        altered=copy.deepcopy(shared);altered[0]["classification"]="unreviewed"
        rejects("unreviewed_classification",lambda:engine.create_archive(root,[root],altered,target(),"shared_review",EVIDENCE))
        altered=copy.deepcopy(shared);altered[0]["sha256"]="0"*64
        rejects("source_bytes_changed_since_selection",lambda:engine.create_archive(root,[root],altered,target(),"shared_review",EVIDENCE))
        altered=copy.deepcopy(shared);altered[0]["source_relative"]="../outside.txt"
        rejects("source_traversal",lambda:engine.create_archive(root,[root],altered,target(),"shared_review",EVIDENCE))
        restricted=scratch/"other";restricted.mkdir()
        rejects("physical_source_outside_project_allowlist",lambda:engine.create_archive(root,[restricted],shared,target(),"shared_review",EVIDENCE))
        link=root/"link.txt";link.symlink_to(root/shared[0]["source_relative"])
        altered=copy.deepcopy(shared[:1]);altered[0]["source_relative"]="link.txt"
        rejects("source_symlink",lambda:engine.create_archive(root,[root],altered,target(),"shared_review",EVIDENCE))
        old=scratch/"shared_review.zip";old_sha=sha(old)
        rejects("existing_archive_never_overwritten",lambda:engine.create_archive(root,[root],shared,old,"shared_review",EVIDENCE))
        assert sha(old)==old_sha
        selected=records["shared_review"]
        existing=scratch/"already_there";existing.mkdir();(existing/"keep").write_bytes(b"preserve")
        rejects("existing_extraction_never_overwritten",lambda:engine.extract_verified(old,existing,old_sha,"shared_review",EVIDENCE,selected["payload_uncompressed_bytes"]))
        assert (existing/"keep").read_bytes()==b"preserve"
        rejects("wrong_archive_sha",lambda:engine.extract_verified(old,scratch/"wrong_sha","0"*64,"shared_review",EVIDENCE,selected["payload_uncompressed_bytes"]))
        rejects("wrong_content_selection_evidence",lambda:engine.extract_verified(old,scratch/"wrong_selection",old_sha,"shared_review","0"*64,selected["payload_uncompressed_bytes"]))
        rejects("wrong_archive_kind",lambda:engine.extract_verified(old,scratch/"wrong_kind",old_sha,"private_review",EVIDENCE,selected["payload_uncompressed_bytes"]))
        rejects("wrong_expected_uncompressed_size",lambda:engine.extract_verified(old,scratch/"wrong_size",old_sha,"shared_review",EVIDENCE,selected["payload_uncompressed_bytes"]+1))
        data=b"nonclinical-known-payload-for-CRC-test"
        base={"schema":engine.SCHEMA,"archive_kind":"shared_review","selection_evidence_sha256":EVIDENCE,
              "files":{"ok.txt":{"sha256":h(data),"bytes":len(data),"classification":"review_shareable"}}}
        def badzip(name,items=None,manifest=None,raw_metadata=None):
            archive=scratch/(name+".zip")
            m=copy.deepcopy(base if manifest is None else manifest)
            items=[("ok.txt",data,None)] if items is None else items
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",UserWarning)
                with zipfile.ZipFile(archive,"x",zipfile.ZIP_STORED) as z:
                    for path,value,mode in items:
                        info=zipfile.ZipInfo(path)
                        if mode is not None:info.create_system=3;info.external_attr=mode<<16
                        z.writestr(info,value)
                    if raw_metadata is not False:
                        metadata=json.dumps(m).encode() if raw_metadata is None else raw_metadata
                        z.writestr(engine.MANIFEST,metadata)
            return archive,m
        def reject_zip(name,items=None,manifest=None,raw_metadata=None,mutate=None):
            archive,m=badzip(name,items,manifest,raw_metadata)
            if mutate:mutate(archive)
            expected_bytes=sum(v["bytes"] for v in m["files"].values())
            rejects(name,lambda:engine.extract_verified(archive,scratch/("extract_"+name),sha(archive),"shared_review",EVIDENCE,expected_bytes))
        for name,badname in [("path_traversal","../escape.txt"),("absolute_path","/escape.txt"),
                             ("windows_path","C:\\escape.txt"),("noncanonical_name","a//b.txt")]:
            m=copy.deepcopy(base);m["files"]={badname:next(iter(m["files"].values()))}
            reject_zip(name,[(badname,data,None)],m)
        reject_zip("duplicate_zip_member",[("ok.txt",data,None),("ok.txt",data,None)])
        reject_zip("unexpected_member",[("ok.txt",data,None),("extra.txt",b"extra",None)])
        reject_zip("missing_manifest",raw_metadata=False)
        m=copy.deepcopy(base);m["files"]["absent.txt"]=dict(m["files"]["ok.txt"])
        reject_zip("missing_payload",manifest=m)
        m=copy.deepcopy(base);m["files"]["ok.txt"]["sha256"]="0"*64
        reject_zip("payload_sha_mismatch",manifest=m)
        m=copy.deepcopy(base);m["files"]["ok.txt"]["bytes"]+=1
        reject_zip("payload_size_mismatch",manifest=m)
        m=copy.deepcopy(base);m["files"]["ok.txt"]["classification"]="private_only"
        reject_zip("private_content_declared_in_shared_zip",manifest=m)
        reject_zip("symlink_zip_member",[("ok.txt",data,stat.S_IFLNK|0o777)])
        m=copy.deepcopy(base);m["files"]={"directory/":dict(m["files"]["ok.txt"])}
        reject_zip("directory_zip_member",[("directory/",data,stat.S_IFDIR|0o700)],m)
        for name,names in [("case_colliding_members",["a.txt","A.txt"]),
                           ("file_directory_prefix",["a","a/b"]),
                           ("case_colliding_prefix",["A","a/b"])]:
            m=copy.deepcopy(base);m["files"]={x:dict(base["files"]["ok.txt"]) for x in names}
            reject_zip(name,[(x,data,None) for x in names],m)
        m=copy.deepcopy(base);m["files"]={"manifest_sha256.json":dict(m["files"]["ok.txt"])}
        reject_zip("case_colliding_reserved_manifest",[("manifest_sha256.json",data,None)],m)
        raw=json.dumps(base).encode()
        raw=b'{"schema":"'+engine.SCHEMA.encode()+b'",'+raw[1:]
        reject_zip("duplicate_manifest_JSON_key",raw_metadata=raw)
        def corrupt(path):
            blob=path.read_bytes();where=blob.index(data)
            path.write_bytes(blob[:where]+bytes([blob[where]^1])+blob[where+1:])
        reject_zip("actual_payload_CRC_corruption",mutate=corrupt)
        assert not (scratch.parent/"escape.txt").exists()
        assert all(sha(root/name)==h(value) for name,value in content.items())
        assert all(sha(p)==v["sha256"] for p,v in protected.items())
        assert all(k not in sys.modules for k in ["numpy","pandas","sklearn","torch","joblib"])
        test_count=len(cases)
        receipts={"status":"PASS_BOUNDED_ARCHIVE_ENGINEERING_ONLY","engine_sha256":ENGINE_SHA,
                  "actual_cases":cases,"case_count":test_count,"actual_roundtrips":2,
                  "negative_rejection_cases":test_count-2,
                  "nonclinical_fixture_only":True,"study_data_or_models_loaded":False,
                  "full_study_archives_created":False,"content_shareability_review_performed":False,
                  "archive_contents_executed":False,"scientific_or_package_acceptance":False,
                  "existing_analysis_and_92_manuscript_hashes_unchanged":True,
                  "max_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  "seconds":time.monotonic()-start,
                  "fixture_archive_receipts":records}
        save(O/"ENGINEERING_QUALIFICATION.json",receipts)
        shutil.rmtree(scratch);assert not scratch.exists()
        save(O/"OWNED_FIXTURE_CLEANUP.json",{"removed_only_owned_nonclinical_test_copy":str(scratch),
                                           "exists":False,"original_files_removed":0})
        print(json.dumps({"status":receipts["status"],"cases":test_count,"seconds":receipts["seconds"],
                          "max_RSS_bytes":receipts["max_RSS_bytes"],
                          "receipt_sha256":sha(O/"ENGINEERING_QUALIFICATION.json")}))
    except BaseException as e:
        save(O/"FAILURE.json",{"error":repr(e),"traceback":traceback.format_exc(),
                              "scratch_retained":str(scratch),"completed_cases":cases})
        raise
if __name__=="__main__":main()
