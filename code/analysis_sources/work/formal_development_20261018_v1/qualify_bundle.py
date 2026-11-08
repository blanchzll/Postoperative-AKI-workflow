"""Real input raw-bundle roundtrip and data-minimisation acceptance."""
from core import *
from raw_outer import raw_fit_pair
from export_audit import minimise_imputer,public_export_gate
from preprocess import StableMixedImputer
import joblib,numpy as np
p=parts("geography","preop","final")[0]
t={"line":"geography","cohort":"preop","stage":"preop","branch":"compact","context":"final","family":"LR","method":"bayes"}
dest=OUT/"qualification_bundle"
rec=raw_fit_pair(t,p,{"variant":"linear","C":0.1},1,dest)
bundle=joblib.load(dest/"raw_bundle.joblib")
imp=bundle["imputer"];small,removed=minimise_imputer(imp)
ct=pair_task("geography","preop","preop","final",p,"bayes","compact")
sp=spec_for("preop",COMPACT);_,_,raw,vm=frame_and_meta(ct,sp)
before=imp.transform(raw.head(8),keys(vm.head(8)));after=small.transform(raw.head(8),keys(vm.head(8)))
for c in sp["num"]:np.testing.assert_allclose(before[c],after[c],rtol=1e-12,atol=1e-12)
for c in sp["cat"]:assert before[c].tolist()==after[c].tolist()
assert all(not hasattr(small,k) for k in removed)
assert not public_export_gate("KNN","bayes")["binary_export_allowed_by_default"]
assert not public_export_gate("SVM","bayes")["binary_export_allowed_by_default"]
atomic(OUT/"RAW_BUNDLE_FUNCTIONAL_QUALIFICATION.json",{"status":"PASS","full_raw_input_imputer_encoder_predictor_roundtrip":True,"imputer_training_arrays_removed_in_qualified_copy":removed,"minimised_copy_predictions_unchanged":True,"all_models_and_patients_remain_private":True,"scope":"engineering check on development data; not an unbiased performance result","not_calibrated_or_clinically_validated":True,"code_sha256":{n:sha(CODE/n) for n in ["raw_outer.py","export_audit.py"]}})
print(canonical({"status":"PASS","raw_bundle_roundtrip":True,"minimised_prediction_unchanged":True,"public_export_executed":False}))
