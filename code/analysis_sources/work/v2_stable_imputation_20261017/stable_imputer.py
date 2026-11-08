"""Versioned per-encounter stochastic completion; no ID enters model predictors."""
import hashlib,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"work/python_six_outcomes_20261017"))
from mixed_imputer import MixedImputer,METHODS
class StableMixedImputer(MixedImputer):
    inference_version="v2_encounter_keyed_single_completion"
    @staticmethod
    def _keys(frame,case_keys):
        if case_keys is None:raise ValueError("Explicit stable encounter keys are required")
        keys=[str(v) if v is not None else "" for v in case_keys]
        if len(keys)!=len(frame) or any(not k or k in ["nan","<NA>"] for k in keys):
            raise ValueError("One nonmissing encounter key per row is required")
        if len(set(keys))!=len(keys):raise ValueError("Encounter keys must be unique within a batch")
        return keys
    def fit(self,frame,case_keys=None):
        keys=self._keys(frame,case_keys)
        order=sorted(range(len(keys)),key=lambda i:keys[i])
        super().fit(frame.iloc[order])
        self.fit_order_policy_="canonical encounter-key order"
        self.fit_key_digest_=hashlib.sha256("\n".join(keys[i] for i in order).encode()).hexdigest()
        return self
    def fit_transform(self,frame,case_keys=None):
        self.fit(frame,case_keys)
        return self.transform(frame,case_keys)
    def transform(self,frame,case_keys=None):
        keys=self._keys(frame,case_keys)
        if self.method not in ["cart","bayes","pmm"]:
            return super().transform(frame)
        raw=self._array(frame);z=self._initialize(raw)
        for i,key in enumerate(keys):
            if not np.isnan(raw[i]).any():continue
            payload=f"AKI-imputation-v2|{self.seed}|{key}".encode()
            seed=int.from_bytes(hashlib.sha256(payload).digest()[:16],"little")
            rng=np.random.default_rng(seed);one=z[i:i+1]
            for step in range(self.iterations):
                for j in range(len(self.cols)):
                    if np.isnan(raw[i,j]):
                        one[0,j]=np.asarray(self._predict(self.models_[j],self._design(one,j),j,rng)).reshape(-1)[0]
        if not np.isfinite(z).all():raise ValueError("Nonfinite completion")
        observed=~np.isnan(raw)
        if not np.array_equal(z[observed],raw[observed]):raise AssertionError("Observed values modified")
        return self._frame(z,frame.index)
