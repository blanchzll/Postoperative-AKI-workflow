"""Verify every Python imputation and summarize the completed development-only comparison."""
from run_imputation_jobs import *
import collections,time
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score
NAMES={"mean_mode":"mean-mode（均值/众数）","knn":"KNN","cart":"CART","random_forest":"随机森林","bayes":"Bayes","pmm":"PMM"}
def main():
    manifest=json.loads((BASE/"fold_inputs/fold_manifest.json").read_text());spec=json.loads((BASE/"fold_inputs/feature_spec.json").read_text())
    pairs=[p for p in manifest["pairs"] if p["role"].startswith("development_")]
    selection=json.loads((BASE/"six_by_nine/imputation_selection.json").read_text())
    assert sha(BASE/"cohort.csv")==manifest["source_cohort_sha256"]==selection["source_cohort_sha256"]
    assert selection["geography_validation_used"] is False
    frozen=json.loads((BASE/"code_freeze_manifest.json").read_text())
    for p,h in frozen["files"].items():assert sha(ROOT/p)==h,p
    observed=0;done=0;notes=collections.Counter();missing=[];hist=[]
    for method in METHODS:
        for p in pairs:
            d=BASE/"imputed"/method/p["name"]
            assert json.loads((d/"provenance.json").read_text())==signature(p,method);check_outputs(d,p)
            md=json.loads((d/"metadata.json").read_text());assert md["training_only"] is True and md["engine"]=="Python"
            for w in md["warnings"]:notes[(method,w)]+=1
            missing.append({"method":method,"pair":p["name"],"training_all_missing_columns":md["all_missing_training_columns"]})
            hist.append({"method":method,"pair":p["name"],"updates":md.get("training_update_history",[])})
            frames={}
            for side in ["train","valid"]:
                raw=pd.read_csv(ROOT/p["files"][side+"_features"]["path"],dtype={k:str for k in spec["cat"]},float_precision="round_trip")
                imputed=pd.read_csv(d/(side+".csv"),dtype={k:str for k in spec["cat"]},float_precision="round_trip")
                frames[side]=(raw,imputed)
                for c in raw:
                    known=raw[c].notna();observed+=int(known.sum())
                    if c in spec["cat"]:
                        assert np.array_equal(raw.loc[known,c],imputed.loc[known,c])
                        assert set(imputed[c])<=set(spec["cat_levels"][c])
                    else:
                        assert np.allclose(raw.loc[known,c],imputed.loc[known,c],rtol=1e-12,atol=1e-12)
                        if c!="WeightLoss":assert not imputed.loc[~known,c].lt(0).any()
            if method=="pmm":
                donor_frame=frames["train"][0]
                for side,(raw,imputed) in frames.items():
                    for c in spec["num"]:
                        donors=donor_frame[c].dropna()
                        if len(donors):assert np.isin(imputed.loc[raw[c].isna(),c],donors).all(),(p["name"],side,c)
            done+=1
    assert done==120
    for p in selection["input_prediction_files"]:assert sha(ROOT/p["path"])==p["sha256"]
    ranking=pd.read_csv(BASE/"six_by_nine/imputation_ranking.csv");metrics=pd.read_csv(BASE/"six_by_nine/development_oof_metrics_54.csv")
    assert len(ranking)==6 and len(metrics)==54
    winner=selection["selected_method"];assert winner==ranking.iloc[0].method
    audit={"status":"passed","engine":"Python","pairs_checked":120,"method_family_estimates":54,"outer_family_prediction_files":270,"observed_cells_verified":observed,"true_PMM_training_donor_membership_verified":True,"legal_categories_verified":True,"nonnegative_imputed_numeric_support_verified":True,"signed_WeightLoss_preserved":True,"source_cohort_sha256":sha(BASE/"cohort.csv"),"code_hashes_verified":True,"warnings":[{"method":k[0],"warning":k[1],"n_pairs":v} for k,v in notes.items()],"all_missing_training_columns":[x for x in missing if x["training_all_missing_columns"]],"selected_method":winner,"selection_excludes_heldout_outcomes":True,"time":time.time()}
    atomic(BASE/"integrity_and_method_audit.json",audit)
    atomic(BASE/"training_update_diagnostics.json",{"interpretation":"Descriptive5-sweep changes; stochastic methods need not reach zero. Not a proof of MCMC convergence.","pairs":hist})
    fold_rows=[]
    for method in METHODS:
        for family in sorted(metrics.family.unique()):
            values=[]
            for i in range(5):
                d=pd.read_csv(BASE/"six_by_nine"/method/f"dev_fold_{i}"/family/"predictions.csv")
                values.append(float(average_precision_score(d.y,d.prob_calibrated)))
            fold_rows.append({"method":method,"family":family,"outer_AP_min":min(values),"outer_AP_max":max(values),"outer_AP_mean":np.mean(values)})
    pd.DataFrame(fold_rows).to_csv(BASE/"six_by_nine/outer_AP_stability.csv",index=False)
    comparison=[]
    best=metrics[metrics.method.eq(winner)].set_index("family")
    for method in METHODS:
        if method==winner:continue
        d=metrics[metrics.method.eq(method)].set_index("family");delta=best.ap-d.ap
        comparison.append({"candidate":winner,"comparator":method,"families_higher_AP":int((delta>0).sum()),"families_lower_AP":int((delta<0).sum()),"mean_AP_difference":float(delta.mean())})
    atomic(BASE/"six_by_nine/selection_review.json",{"selected_by_prespecified_rule":winner,"source":"completed Python six-method comparison","comparisons":comparison,"quality_audit_passed":True,"not_universally_superior_by_all_metrics":True,"downstream_main_models_started":False})
    sys_path=str(ROOT/"work/next_validations_20261016")
    import sys;sys.path.insert(0,sys_path)
    from report_format import markdown_table
    rows=[{"方法":NAMES[r.method],"平均AP排名":f"{r.mean_ap_rank:.3f}","平均AP":f"{r.mean_ap:.4f}","平均Brier":f"{r.mean_brier:.4f}"} for r in ranking.itertuples()]
    result=f"# Python六种插补方法比较\n\n按预先确定的开发集九模型平均AP排名标准，首选为 **{NAMES[winner]}**。六种方法均已完成；未使用地理或时间验证集结局来选择。\n\n"
    result+=markdown_table(rows,["方法","平均AP排名","平均AP","平均Brier"])+"\n\n"
    apbest=ranking.sort_values("mean_ap",ascending=False).iloc[0].method
    bbest=ranking.sort_values("mean_brier").iloc[0].method
    result+=f"九模型平均AP最高为{NAMES[apbest]}，平均Brier最低为{NAMES[bbest]}。主选择依据保持为平均AP排名；逐模型差异另表报告，不将排名首选解释为所有模型或指标均优。\n\n"
    result+="开发集2853条记录、99例AKI，使用相同5折外层及3折内层患者分组；120组Python插补、270项外层模型评估，汇总54个方法与模型组合。类别合法性、原有观测值不变和PMM取值来自训练供体已逐项核验。\n\n"
    result+="本次按方法定义修正原Python代码的命名差异：mean-mode确实使用数值均值及分类众数；PMM确实匹配训练供体。Bayes连续变量采用贝叶斯回归后验预测，分类变量采用二项或多项逻辑回归；KNN使用混合距离；CART从叶内训练供体抽样；随机森林分别使用回归和分类森林。各方法的参数、边界处理、代码和随机种子已保存。\n\n"
    result+="数值量除体重变化外使用非负支持。Bayes在后验预测抽样时实施该支持，PMM/CART/KNN/随机森林的供体或训练响应提供相应支持。未用验证集分布设定边界。五次链式更新变化及各外层折AP范围另存，仅作稳定性描述。\n\n"
    result+="本阶段完成的是插补方法选择。完整术前、精简对照和Stacking主模型尚未接续，当前开发集结果是选择条件下的结果；地理与时间验证随后独立开展。\n"
    (BASE/"six_by_nine/IMPUTATION_SELECTION_REPORT_ZH.md").write_text(result)
    print(json.dumps({"status":"passed","pairs":120,"observed_cells":observed,"selected_method":winner,"warnings_groups":len(notes)}),flush=True)
if __name__=="__main__":main()

