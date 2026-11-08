"""Independent saved-result arithmetic and lineage checks; no new model fits."""
from pathlib import Path
import json,pickle,gzip,math,os,subprocess,time
import numpy as np,pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
import analysis_core as ac
import importlib.util
_qspec=importlib.util.spec_from_file_location("resolution_qualified_profile",Path(__file__).with_name("qualify.py"))
_qmod=importlib.util.module_from_spec(_qspec);_qspec.loader.exec_module(_qmod)
independent_profile=_qmod.independent_profile
def current_nonoverlap_comparison(ctx,data):
    identities={}
    for name,route in data[ctx]["cal_routes"].items():
        identities[name]=json.loads(Path(route["manifest_path"]).read_text())["identity"]
    base=("preop","compact") if ctx.endswith("preop") else ("preop","full")
    up=("preop","full") if ctx.endswith("preop") else ("endop","full")
    def match(key):
        hit=[name for name,i in identities.items() if (i["stage"],i["branch"])==key]
        assert len(hit)==1
        return hit[0]
    return match(up)+" MINUS "+match(base)
def main():
    start=time.monotonic();accepted=json.loads((ac.O/"RESULT_ACCEPTANCE.json").read_text())
    for p,h in accepted["outputs_sha256"].items():assert ac.sha(ac.O/p)==h,p
    with (ac.W/"PRIVATE_INPUTS.pkl").open("rb") as f:data=pickle.load(f)
    diag=pd.read_csv(ac.O/"CALIBRATION_POINT_DIAGNOSTICS.csv",float_precision="round_trip")
    par=pd.read_csv(ac.O/"CALIBRATION_FOLD_PARAMETERS.csv",float_precision="round_trip")
    errors=[];coeferr=[];checked=0;groups=0
    for item in accepted["private_crossfitted_prediction_files"]:
        p=Path(item["path"]);assert ac.sha(p)==item["sha256"]
        df=pd.read_csv(p,dtype={"group_hash":str,"excel_row":str},float_precision="round_trip")
        ctx=p.parent.name;route=p.stem;ds=data[ctx]
        assert np.array_equal(df.group_hash,ds["meta"].group_hash)
        yy=df.y.to_numpy(dtype=int);gg=df.group_hash.to_numpy();rr=df.excel_row.to_numpy();fold=df.target_adaptation_fold.to_numpy()
        orig=df.original_V1_prob.to_numpy()
        masks=[]
        for method in ac.METHODS:
            probs=df[method+"_prob"].to_numpy()
            src=diag.loc[(diag.context==ctx)&(diag.route==route)&(diag.method==method)].iloc[0]
            vals={"AUROC":roc_auc_score(yy,probs),"AP":average_precision_score(yy,probs),
                  "Brier":brier_score_loss(yy,probs),"logloss":log_loss(yy,probs),"observed_expected":yy.sum()/probs.sum()}
            rank=np.lexsort((rr,gg,-probs))
            for cap in [10,20]:
                selected=rank[:math.ceil(len(yy)*cap/100)]
                vals[f"capacity{cap}_captured_n"]=yy[selected].sum()
            tp=((probs>=.05)&(yy==1)).sum();fp=((probs>=.05)&(yy==0)).sum()
            vals["NB_0.05"]=(tp-fp*.05/.95)/len(yy)
            for k,v in vals.items():errors.append(abs(float(v)-float(src[k])));checked+=1
            masks.append(df[method+"_adaptation_capacity20_alert"].to_numpy())
            for k in range(3):
                tr=fold!=k;te=~tr;assert set(gg[tr]).isdisjoint(gg[te]);groups+=1
                row=par.loc[(par.context==ctx)&(par.route==route)&(par.method==method)&(par.fold==k)].iloc[0]
                if method=="original_V1":continue
                coef=independent_profile(yy[tr],orig[tr],np.ones(tr.sum()),"offset" if method=="offset_MAP" else "logistic")
                z=np.log(np.clip(orig[te],1e-6,1-1e-6)/(1-np.clip(orig[te],1e-6,1-1e-6)))
                pred=expit(coef[0]+coef[1]*z);coeferr.append(float(np.max(np.abs(pred-probs[te]))))
                assert row.adaptation_records_weighted==tr.sum() and row.adaptation_events_weighted==yy[tr].sum()
        assert all(np.array_equal(masks[0],x) for x in masks[1:]),(ctx,route)
    assert max(errors)<1e-11,max(errors)
    assert max(coeferr)<1e-6,max(coeferr)
    manifest=json.loads((ac.O/"RUN_MANIFEST.json").read_text())
    support_tables={"alt":pd.read_csv(ac.O/"ALT_SUPPORT.csv"),"cal":pd.read_csv(ac.O/"CALIBRATION_UPDATE_SUPPORT.csv"),"nonoverlap":pd.read_csv(ac.O/"NONOVERLAP_SUPPORT.csv")}
    selected={}
    for j in manifest["jobs"]:
        kind,ctx,name,a,b=j
        if kind=="random":continue
        comp="change_in_gain_omit_ALT_minus_control" if kind=="alt" else "logistic_MAP_MINUS_original_V1" if kind=="cal" else current_nonoverlap_comparison(ctx,data)
        metric="Brier" if kind=="cal" else "capacity20_captured_per1000"
        import runner
        x,_=runner.paths(j)
        with gzip.open(x,"rb") as f:draws=pickle.load(f)
        selected.setdefault((kind,ctx,name,comp,metric),[]).extend(d[comp][metric] for d in draws)
    qerrors=[]
    for (kind,ctx,name,comp,metric),values in selected.items():
        df=support_tables[kind]
        row=df.loc[(df.context==ctx)&(df.comparison==comp)&(df.metric==metric)]
        if kind=="cal":row=row.loc[row.route==name]
        assert len(row)==1
        x=np.array(values);assert len(x)==2000 and np.isfinite(x).all()
        lo,hi=np.percentile(x,[2.5,97.5]);qerrors.extend([abs(lo-row.iloc[0].lower),abs(hi-row.iloc[0].upper)])
    assert max(qerrors)<1e-11
    minimums={kind:int(tab.loc[tab.metric=="Brier","valid_bootstraps"].min()) for kind,tab in support_tables.items()}
    assert all(x==2000 for x in minimums.values())
    protected=json.loads((ac.O/"INPUT_ACCEPTANCE.json").read_text())["protected_frozen_files_sha256"]
    assert all(ac.sha(ac.R/p)==h for p,h in protected.items())
    state=json.loads((ac.O/"RUN_STATUS.json").read_text())
    pid=Path("/proc")/str(state["pid"])
    live=pid.exists() and int((pid/"stat").read_text().split()[21])==state["start_ticks"]
    assert not live,"Batch runner still owns resources"
    ac.save(ac.O/"INDEPENDENT_RESULT_VERIFICATION.json",{"status":"PASS_saved_predictions_intervals_folds_and_source_preservation","saved_point_metric_checks":checked,"max_metric_difference":max(errors),"held_group_boundaries":groups,"independent_saved_update_checks":len(coeferr),"max_saved_update_probability_difference":max(coeferr),"independent_bootstrap_effect_CI_checks":len(selected),"max_interval_difference":max(qerrors),"primary_Brier_valid_replicates":minimums,"zero_alert_PPV_missing_denominators_retained_not_coerced":True,"strictly_monotone_adaptation_within_fold_preserves_local_capacity_alerts":True,"protected_source_files_unchanged":len(protected),"runner_owned_resources_exited":True,"code_sha256":ac.sha(Path(__file__)),"seconds":time.monotonic()-start})
    print(json.dumps({"status":"PASS","metric_checks":checked,"max_metric_error":max(errors),"profile_checks":len(coeferr),"max_probability_error":max(coeferr),"CI_checks":len(selected),"seconds":time.monotonic()-start}))
if __name__=="__main__":main()
