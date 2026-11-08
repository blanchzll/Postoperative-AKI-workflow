import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
class NearZeroVariance(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        keep=[]
        for j in range(X.shape[1]):
            _,counts=np.unique(X[:,j],return_counts=True);counts=np.sort(counts)[::-1]
            ratio=counts[0]/counts[1] if len(counts)>1 else np.inf
            keep.append(len(counts)>1 and not (ratio>19 and 100*len(counts)/len(X)<=10))
        self.keep_=np.asarray(keep);assert self.keep_.any();return self
    def transform(self,X):return X[:,self.keep_]
