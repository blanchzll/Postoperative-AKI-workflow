"""Minimise imputer state without changing inference; public export remains gated."""
from core import *
from preprocess import StableMixedImputer
import copy
def minimise_imputer(imp):
 # Donor methods require patient-level training values; never silently strip and change algorithm.
 if imp.method not in ["bayes","mean_mode","random_forest"]:
  raise ValueError("Research-only model object retains required patient-level donors; public binary export requires separate authorised data handling")
 result=copy.deepcopy(imp);removed=[]
 for attr in ["original_","initial_filled_","train_filled_"]:
  if hasattr(result,attr):delattr(result,attr);removed.append(attr)
 return result,removed
def public_export_gate(family,imputer_method):
 direct_sample_models={"KNN","SVM"}
 if family in direct_sample_models:return {"binary_export_allowed_by_default":False,"reason":"learner retains fitted samples/support vectors; keep hosted model private, release code/schema and synthetic examples"}
 if imputer_method not in ["bayes","mean_mode","random_forest"]:
  return {"binary_export_allowed_by_default":False,"reason":"imputer retains required patient-level donor information"}
 return {"binary_export_allowed_by_default":False,"reason":"candidate for minimised binary after trained-object audit and explicit release review; no automatic publishing"}
