"""Grouped development-only six-imputer/nine-family selection. No final evaluation."""
from pathlib import Path
import os,sys,json,time,hashlib,resource,warnings,gc
import numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from threadpoolctl import threadpool_limits
from model_candidates import model_candidates
ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/"outputs/v4_development_inputs_20261018";OUT=ROOT/"outputs/v4_development_pipeline_20261018"
FAMILIES=["LR","DT","RF","KNN","SVM","NB","XGB","SGBT","NNET"]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,v):
    q=p.with_suffix(p.suffix+".tmp");q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n");os.replace(q,p)
def main():
    line,method,family,threads=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
    assert line in ["geography","temporal"] and family in FAMILIES
    dest=OUT/"comparison"/line/method/family;dest.mkdir(mode=0o700,parents=True,exist_ok=True)
    done=dest/"COMPLETE.json"
    if done.exists():
        r=json.loads(done.read_text())
        for f,h in r["outputs_sha256"].items():assert sha(dest/f)==h
        print(json.dumps({"reused":True,"job":[line,method,family]}));return
    start=time.time();spec=json.loads((INPUT/"FEATURE_SPECS.json").read_text())["preop"]
    cols=spec["num"]+spec["cat"];data=[];input_receipts={}
    lock=json.loads((INPUT/"SOURCE_LOCK.json").read_text())
    for f,h in lock["input_sha256"].items():assert sha(INPUT/f)==h
    for fold in range(3):
        d=OUT/line/("inner_"+str(fold))/method
        receipt=json.loads((d/"COMPLETE.json").read_text())
        assert receipt["imputer_received_no_outcome_fields"] and receipt["global_hospital_time_evaluation_not_fitted"]
        assert receipt["source_input_acceptance_sha256"]==sha(INPUT/"INPUT_ACCEPTANCE.json")
        for f,h in receipt["outputs_sha256"].items():assert sha(d/f)==h
        tr=pd.read_csv(d/"train.csv",dtype={c:"string" for c in spec["cat"]}).set_index("excel_row")
        va=pd.read_csv(d/"assessment.csv",dtype={c:"string" for c in spec["cat"]}).set_index("excel_row")
        tm=pd.read_csv(d/"train_metadata.csv",usecols=["excel_row","group_hash","y",line+"_partition_v2"],dtype={"group_hash":str})
        vm=pd.read_csv(d/"assessment_metadata.csv",usecols=["excel_row","group_hash","y",line+"_partition_v2"],dtype={"group_hash":str})
        assert set(tm.group_hash).isdisjoint(vm.group_hash)
        assert tm[line+"_partition_v2"].eq("development").all() and vm[line+"_partition_v2"].eq("development").all()
        assert tr.index.tolist()==tm.excel_row.tolist() and va.index.tolist()==vm.excel_row.tolist()
        assert list(tr.columns)==cols and list(va.columns)==cols
        encode=ColumnTransformer([("numeric",StandardScaler(),spec["num"]),("categorical",OneHotEncoder(categories=[spec["cat_levels"][c] for c in spec["cat"]],handle_unknown="error",sparse_output=False,dtype=np.float64),spec["cat"])],remainder="drop")
        xx=encode.fit_transform(tr);vv=encode.transform(va)
        assert np.isfinite(xx).all() and np.isfinite(vv).all()
        assert tm.y.nunique()==2 and vm.y.nunique()==2
        data.append((xx,tm,vv,vm))
        input_receipts[str(fold)]=sha(d/"COMPLETE.json")
    candidates=model_candidates(seed=42,threads=threads,gpu=False)[family]
    allscores=[];best=None;allnotes=[]
    for name,est in candidates:
        parts=[];foldscores=[];notes=[]
        for fold,(train,tm,valid,vm) in enumerate(data):
            model=clone(est)
            with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as ww:
                warnings.simplefilter("always")
                model.fit(train,tm.y.to_numpy())
                assert list(model.classes_)==[0,1]
                pred=np.asarray(model.predict_proba(valid)[:,1],dtype=float)
            assert pred.shape==(len(vm),) and np.isfinite(pred).all() and np.all((pred>=0)&(pred<=1))
            notes.extend(str(w.message) for w in ww)
            foldscores.append({"fold":fold,"n":len(vm),"events":int(vm.y.sum()),"AP":float(average_precision_score(vm.y,pred)),"Brier":float(brier_score_loss(vm.y,pred)),"AUROC":float(roc_auc_score(vm.y,pred))})
            part=vm[["excel_row","group_hash","y"]].copy();part["prob_raw"]=pred;part["fold"]=fold;parts.append(part)
            del model;gc.collect()
        score={"candidate":name,"mean_fold_AP":float(np.mean([r["AP"] for r in foldscores])),"mean_fold_Brier":float(np.mean([r["Brier"] for r in foldscores])),"mean_fold_AUROC":float(np.mean([r["AUROC"] for r in foldscores])),"fold_metrics":foldscores,"warnings":sorted(set(notes))}
        allscores.append(score);allnotes.extend(notes)
        rank=(-score["mean_fold_AP"],score["mean_fold_Brier"],name)
        if best is None or rank<best[0]:best=(rank,score,pd.concat(parts,ignore_index=True))
    chosen=best[1];predictions=best[2]
    expected=pd.read_csv(INPUT/"preop_metadata.csv")
    expected=expected.loc[expected[line+"_partition_v2"].eq("development")]
    assert predictions.excel_row.is_unique and set(predictions.excel_row)==set(expected.excel_row)
    ymap=expected.set_index("excel_row").y
    np.testing.assert_array_equal(predictions.y.to_numpy(),ymap.loc[predictions.excel_row].to_numpy())
    predictions["method"]=method;predictions["family"]=family;predictions["candidate"]=chosen["candidate"]
    predictions.to_csv(dest/"SELECTED_INNER_OOF.csv",index=False)
    atomic(dest/"CANDIDATE_SCORES.json",{"line":line,"method":method,"family":family,"candidates":allscores,"selected":chosen["candidate"],"selection_rule":"highest mean inner-foldAP; tie lowest meanBrier then candidateName","scores_are_training_selection_not_final_validation":True,"all_probabilities_raw_no_clinical_calibration_claim":True})
    outputs={f:sha(dest/f) for f in ["SELECTED_INNER_OOF.csv","CANDIDATE_SCORES.json"]}
    rec={"line":line,"method":method,"family":family,"selected_candidate":chosen["candidate"],"mean_fold_AP":chosen["mean_fold_AP"],"mean_fold_Brier":chosen["mean_fold_Brier"],"mean_fold_AUROC":chosen["mean_fold_AUROC"],"n_candidates":len(candidates),"fits":len(candidates)*3,"source_lock_sha256":sha(INPUT/"SOURCE_LOCK.json"),"candidate_factory_sha256":sha(Path(__file__).parent/"model_candidates.py"),"protocol_sha256":sha(OUT/"COMPARISON_PROTOCOL.json"),"imputation_receipt_sha256":input_receipts,"group_disjoint_three_inner_folds":True,"final_hospital_or_time_evaluation_used":False,"natural_prevalence_no_SMOTE":True,"train_only_encoding_scaling":True,"backend":"CPU_consistent_all_methods","warnings":sorted(set(allnotes)),"seconds":time.time()-start,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"outputs_sha256":outputs}
    atomic(done,rec)
    print(json.dumps({k:rec[k] for k in ["line","method","family","n_candidates","fits","seconds","peak_rss_bytes"]}),flush=True)
if __name__=="__main__":main()
