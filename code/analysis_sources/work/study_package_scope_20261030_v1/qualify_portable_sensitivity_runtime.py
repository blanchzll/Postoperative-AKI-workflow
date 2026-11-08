"""Finite stdlib provenance/AST preparation for sensitivity18/30 private replay."""
from pathlib import Path
import csv, datetime, hashlib, importlib.util, json, math, os, resource
import shutil, signal, sys, tempfile, time, traceback

R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
B=R/"outputs/study_package_scope_20261030_v1"
W=R/"work/study_package_scope_20261030_v1"
DEST=B/"portable_sensitivity_runtime_preparation_v1"
ADAPTER_SHA="f379945e908056827a96333f017ebb856892405ee50a8c6c380246ef2fe7bc62"
OLD_MAP_SHA="a54e7b5c03a840b5255a5c681fe5d257eedf26a6837716ebbf663ade343cc02a"
ROUTE_MAP_SHA="4e4f89c47dd6fd535e742584c968fc50d88cb1db65f2319556396384a0985cc3"
SOURCE_INVENTORY_SHA="1acf18c0c34761eb4e9b29c2b68abf1366bbc79fe173b09ca1bdbdc839715953"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()
def write(path,value):
    text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    with Path(path).open("x") as f:f.write(text);f.flush();os.fsync(f.fileno())
    return sha(path)
def read(path,expected=None):
    if expected is not None:assert sha(path)==expected,str(path)
    return json.loads(Path(path).read_text())
def rows(path):
    with Path(path).open(newline="") as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames and len(set(reader.fieldnames))==len(reader.fieldnames)
        data=list(reader)
    assert data and all(None not in r and None not in r.values() for r in data)
    return data

def main():
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_AS,(128*1024**2,)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(90,)*2)
    signal.alarm(180)
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    started=time.monotonic()
    if DEST.exists():raise RuntimeError("Inspect existing preparation; never overwrite")
    DEST.mkdir(mode=0o700)
    scratch=None
    try:
        assert sha(W/"portable_sensitivity_runtime.py")==ADAPTER_SHA
        spec=importlib.util.spec_from_file_location("_sensitivity_runtime_preparation",W/"portable_sensitivity_runtime.py")
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        manifest=read(B/"private_path_relocation_v1/PRIVATE_EXACT_PATH_MAP_INTERNAL.json",OLD_MAP_SHA)
        entries={v["path"]:v for v in manifest["files"]}
        assert len(entries)==len(manifest["files"])==608
        route_map=read(B/"private_pipeline_payload_inventory_v1/PRIVATE_PIPELINE_ROUTE_MAPPING_INTERNAL.json",ROUTE_MAP_SHA)
        route_map=[v for v in route_map if v["family"]=="fixed_flow_sensitivity"]
        assert len(route_map)==30
        sources={v["path"]:v for v in read(B/"dependency_disposition_v3/CURRENT_SOURCE_INVENTORY_INTERNAL.json",SOURCE_INVENTORY_SHA)}
        added=[]
        def bind(relative,expected,origin):
            p=R/relative
            assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
            assert sha(p)==expected,relative
            if relative in entries:
                assert entries[relative]["sha256"]==expected
                return
            v={"path":relative,"sha256":expected,"bytes":p.stat().st_size,
               "archive_class":"PRIVATE_ONLY_DO_NOT_SHARE",
               "original_absolute_aliases":sorted({str(p),str(p.resolve())})}
            entries[relative]=v;manifest["files"].append(v)
            added.append({"path":relative,"sha256":expected,"binding_origin":origin})
        def original(relative):
            assert relative in entries,relative
            return read(R/relative,entries[relative]["sha256"])
        for relative,expected in adapter.NATIVE_SOURCES.items():
            assert sources[relative]["sha256"]==expected
            bind(relative,expected,"Existing current-source code inventory")
        source_lock=original(str(adapter.SOURCE/"SOURCE_LOCK.json"))
        assert entries[str(adapter.SOURCE/"SOURCE_LOCK.json")]["sha256"]==adapter.SOURCE_SHA
        for name,h in source_lock["input_sha256"].items():
            relative=str(adapter.SOURCE/name)
            if relative in entries:assert entries[relative]["sha256"]==h
        protected=read(R/"outputs/post_reanalysis_process_review_v1/MANUSCRIPT_PROTECTION.json")["files"]
        assert len(protected)==92 and all(sha(p)==v["sha256"] for p,v in protected.items())
        refdir=R/"outputs/fixed_flow_sensitivity_20261024_v1"
        stage=read(refdir/"STAGE_REFIT_ACCEPTANCE.json")
        assert stage["status"]=="PASS_thirty_fixed_flow_scenario_routes_fitted"
        assert stage["unique_fit_caches"]==18 and len(stage["receipts"])==30
        expected_route_paths={str(R/v["route_receipt"]) for v in route_map}
        assert set(stage["receipts"])==expected_route_paths
        for p,h in stage["receipts"].items():
            rel=str(Path(p).relative_to(R));assert entries[rel]["sha256"]==h and sha(p)==h
        route_index=read(refdir/"ROUTES.json")
        assert len(route_index)==10
        code_lock_rel="outputs/fixed_flow_sensitivity_20261024_v1/CODE_LOCK.json"
        expected_code_sha=set()
        specs=original(str(adapter.SOURCE/"FEATURE_SPECS.json"))
        metadata={}
        for cohort in ["preop","paired"]:
            rel=str(adapter.SOURCE/(cohort+"_metadata.csv"))
            assert sha(R/rel)==entries[rel]["sha256"]
            mm=rows(R/rel);metadata[cohort]={int(v["excel_row"]):v for v in mm}
            assert len(mm)==len(metadata[cohort])
        plan=[];cache_routes={}
        for route in route_map:
            receipt=original(route["route_receipt"])
            assert receipt["status"]=="PASS_fixed_flow_sensitivity_route_fitted"
            assert (receipt["route"],receipt["scenario"])==(route["route"],route["scenario"])
            assert receipt["fit_cache"]==str(R/route["fit_cache"])
            assert receipt["exclude_evaluation_rows"]==[]
            cache=route["fit_cache"]
            cache_rel=str(Path(cache)/"COMPLETE.json")
            c=original(cache_rel);assert entries[cache_rel]["sha256"]==receipt["fit_receipt_sha256"]
            assert c["status"]=="PASS_fixed_choice_fit_and_training_only_calibration"
            sig=c["signature"];assert sig["seed"]==42
            expected_code_sha.add(sig["code_lock_sha256"])
            ref=route_index[route["route"]]
            lock_rel=str(Path(ref["path"]).relative_to(R))
            model=original(lock_rel)
            assert entries[lock_rel]["sha256"]==sig["route_lock"]==ref["sha256"]
            assert model["identity"]==ref["identity"]
            assert model["input_source_sha256"]==adapter.SOURCE_SHA
            assert model["family"] in ["LR","SGBT","Voting"],model["family"]
            assert set(sig["definitions"])==set(model["members"])
            for fam,d in sig["definitions"].items():
                assert set(d["columns"])==set(d["spec"]["num"]+d["spec"]["cat"])
                assert set(d["columns"])<=set(model["input_columns"])
                if route["scenario"]=="omit_ALT":assert "PreopALT" not in d["columns"]
            for name,h in c["files"].items():
                rel=str(Path(cache)/name);assert entries[rel]["sha256"]==h and sha(R/rel)==h
            full_rel=str(Path(cache)/"full/COMPLETE.json")
            full=original(full_rel);assert entries[full_rel]["sha256"]==c["full_receipt"]
            assert full["signature"]=={"fit":sig,"scope":"full_train_predict_original_holdout"}
            for name,h in full["files"].items():
                rel=str(Path(cache)/"full"/name);assert entries[rel]["sha256"]==h and sha(R/rel)==h
            identity=model["identity"];m=metadata[identity["cohort"]]
            ordered=sorted((v for v in m.values() if v[identity["line"]+"_partition_v2"]=="validation"),
                           key=lambda v:(v["group_hash"],int(v["excel_row"])))
            ids=[int(v["excel_row"]) for v in ordered]
            assert ids==sig["assessment_rows"]
            pred_rel=str(Path(cache)/"PRIVATE_HELDOUT_PREDICTIONS.csv")
            pred=rows(R/pred_rel);assert [int(v["excel_row"]) for v in pred]==ids
            for actual,expected in zip(pred,ordered):
                for field in ["group_hash","center","surgery_date","y","stage","geography_partition_v2","temporal_partition_v2"]:
                    assert actual[field]==expected[field],(route["route"],field)
                assert all(math.isfinite(float(actual[k])) and 0<=float(actual[k])<=1
                           for k in ["prob_raw","prob_calibrated"])
            train_ids=sig["train_rows"];fold=sig["fold_assignment"]
            assert len(train_ids)==len(set(train_ids))==len(fold)==c["training_n"]
            assert len(train_ids)>0 and set(fold)=={0,1,2}
            assert all(m[i][identity["line"]+"_partition_v2"]=="development" for i in train_ids)
            assert sum(int(m[i]["y"]) for i in train_ids)==c["training_events"]
            train_groups={m[i]["group_hash"] for i in train_ids}
            assert train_groups.isdisjoint(v["group_hash"] for v in ordered)
            group_folds={}
            for i,k in zip(train_ids,fold):group_folds.setdefault(m[i]["group_hash"],set()).add(k)
            assert all(len(v)==1 for v in group_folds.values())
            group_keys=[v["group_hash"]+"|"+v["excel_row"] for v in pred]
            assert len(group_keys)==len(set(group_keys))
            plan.append({
                "family":"fixed_flow_sensitivity","route":route["route"],"scenario":route["scenario"],
                "identity":identity,"model_lock":lock_rel,"model_lock_sha256":entries[lock_rel]["sha256"],
                "route_receipt":route["route_receipt"],"route_receipt_sha256":entries[route["route_receipt"]]["sha256"],
                "fit_cache":cache,"fit_receipt":cache_rel,"fit_receipt_sha256":entries[cache_rel]["sha256"],
                "full_fit_receipt":full_rel,"full_fit_receipt_sha256":entries[full_rel]["sha256"],
                "private_components":str(Path(cache)/"full/private_components.joblib"),
                "calibrator":str(Path(cache)/"calibrator.joblib"),
                "expected_predictions":pred_rel,"expected_predictions_sha256":entries[pred_rel]["sha256"],
                "native_raw_array":str(Path(cache)/"full/raw.npy"),
                "source_metadata":str(adapter.SOURCE/(identity["cohort"]+"_metadata.csv")),
                "source_features":str(adapter.SOURCE/(identity["cohort"]+"_"+identity["stage"]+"_features.csv")),
                "records":len(pred),"events":sum(int(v["y"]) for v in pred),
                "record_order":"Native group_hash then numeric excel_row; full heldout batch",
                "original_evaluation_exclusion_rows":[],
                "old_evaluation_mask_bug_not_applied_to_prediction_bank":True,
                "atol":1e-12,"rtol":0,"numerical_replay_performed":False,
            })
            cache_routes.setdefault(cache,[]).append(route["route"]+"/"+route["scenario"])
        assert len(cache_routes)==18 and len(plan)==30
        assert len(expected_code_sha)==1
        bind(code_lock_rel,next(iter(expected_code_sha)),"All 18 original fit signatures")
        for name in ["ROUTES.json","STAGE_REFIT_ACCEPTANCE.json"]:
            rel=str(Path("outputs/fixed_flow_sensitivity_20261024_v1")/name)
            bind(rel,sha(R/rel),"Content independently matched to all30 original bound route receipts and model locks")
        manifest.update(parent_path_map_sha256=OLD_MAP_SHA,
                        scope="Original608 private payloads plus signed sensitivity inference sources and route mapping; not a complete study package")
        map_path=DEST/"PRIVATE_PATH_MAP_WITH_SENSITIVITY_SOURCE_INTERNAL.json"
        map_hash=write(map_path,manifest)
        plan_hash=write(DEST/"SENSITIVITY_30_ROUTE_REPLAY_PLAN_INTERNAL.json",plan)
        write(DEST/"UNIQUE_18_FIT_CACHE_ROUTE_MAP_INTERNAL.json",cache_routes)
        scratch=Path(tempfile.mkdtemp(prefix="owned-sensitivity-preparation-",dir=DEST))
        needed=list(adapter.NATIVE_SOURCES)+[str(adapter.SOURCE/"FEATURE_SPECS.json"),str(adapter.SOURCE/"SOURCE_LOCK.json")]
        copied=[]
        for relative in needed:
            p=scratch/relative;p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(R/relative,p);assert sha(p)==entries[relative]["sha256"];copied.append(relative)
        resolver,paths,trees,provenance=adapter.prepare_runtime(scratch,map_path,map_hash)
        assert all(n not in sys.modules for n in ["numpy","pandas","scipy","sklearn","torch","joblib"])
        assert all(sha(p)==v["sha256"] for p,v in protected.items())
        # Verify every original mapped payload again after preparing the adapter.
        for v in manifest["files"]:assert sha(R/v["path"])==v["sha256"]
        shutil.rmtree(scratch);assert not scratch.exists();scratch=None
        receipt={
            "status":"PASS_SOURCE_PATH_AST_PREPARATION_ONLY",
            "created_UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source_lock_sha256":adapter.SOURCE_SHA,"adapter_sha256":ADAPTER_SHA,
            "original_mapped_files":608,"prepared_mapped_files":len(manifest["files"]),
            "added_files":added,"mapped_manifest_sha256":map_hash,"route_plan_sha256":plan_hash,
            "routes":30,"unique_fit_caches":18,
            "total_route_prediction_rows":sum(v["records"] for v in plan),
            "unique_cache_prediction_rows":sum(next(v["records"] for v in plan if v["fit_cache"]==k) for k in cache_routes),
            "current_source_metadata_order_endpoint_and_train_group_boundaries_checked":True,
            "native_provenance":provenance,"actual_copied_inspected_files":copied,
            "original_payloads_and_92_manuscript_hashes_unchanged":True,
            "owned_temporary_copy_removed":True,"numerical_library_imports":0,"model_loads":0,
            "predictions":0,"new_fits":0,"new_bootstrap_replicates":0,
            "numeric_relocation_accepted":False,"study_package_accepted":False,
            "seconds":time.monotonic()-started,"max_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        }
        receipt_sha=write(DEST/"ADAPTER_PREPARATION.json",receipt)
        write(DEST/"RUNTIME_PREPARATION_AND_NEXT_STEP_ZH.md",
              "# 敏感性私有迁址准备\n\n"
              "已绑定30路线/18唯一有效拟合缓存，原完整验证名单、患者组训练边界、校准、原生推理AST和私有依赖均已逐项检查。"
              "这次仅执行标准库源/路径预检，没有数值库、模型、预测或新拟合。608原私有载荷保留；原25/35/45图展示与此无关。\n\n"
              "旧评价HIGH硬编码只影响2个时间评价子集；此处完整预测银行使用原全验证名单，不套旧HIGH。"
              "数值纠正独立按fixed_flow_sensitivity_evaluation_current_20261030_r1执行。"
              "正确排除名单为空的情景不提供额外独立稳健性证据。\n\n"
              "下一步须在正式48+6成功退出、最新联合资源准入的有限worker中实际重放18缓存并映射全部30路线，"
              "原完整批次raw/cal逐值按abs1e-12/rtol0比较，拒绝回读原目录，保留原校准方向和模型精度。"
              "准备不是迁址数值、完整包或科学验收。\n")
        print(json.dumps({"status":receipt["status"],"receipt_sha256":receipt_sha,
                          "mapped_files":len(manifest["files"]),"routes":30,"unique_caches":18,
                          "rows":receipt["total_route_prediction_rows"],"seconds":receipt["seconds"],
                          "max_RSS_bytes":receipt["max_RSS_bytes"]},ensure_ascii=False))
    except BaseException as error:
        write(DEST/"PREPARATION_FAILURE.json",{"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc(),
            "owned_scratch_retained_for_diagnosis":str(scratch) if scratch else None})
        raise
if __name__=="__main__":main()
