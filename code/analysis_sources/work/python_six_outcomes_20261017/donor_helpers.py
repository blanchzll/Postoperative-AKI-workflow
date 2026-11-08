"""Training-only PMM and mixed-distance KNN. Critical review replaced a rejected draft."""
import numpy as np
def pmm_donors(observed_prediction,missing_prediction,observed_values,k,rng):
    observed_prediction=np.asarray(observed_prediction,dtype=float)
    missing_prediction=np.asarray(missing_prediction,dtype=float)
    observed_values=np.asarray(observed_values,dtype=float)
    if any(v.ndim!=1 for v in [observed_prediction,missing_prediction,observed_values]):raise ValueError("1D arrays required")
    if not len(observed_prediction) or len(observed_prediction)!=len(observed_values):raise ValueError("Observed donors required")
    if not all(np.isfinite(v).all() for v in [observed_prediction,missing_prediction,observed_values]):raise ValueError("Finite inputs required")
    if int(k)!=k or k<1:raise ValueError("Positive integer k required")
    n=len(observed_prediction);k=min(int(k),n);out=np.empty(len(missing_prediction))
    for i,p in enumerate(missing_prediction):
        # Seeded random ordering breaks boundary ties without favoring early records.
        perm=rng.permutation(n)
        near=perm[np.argsort(np.abs(observed_prediction[perm]-p),kind="stable")[:k]]
        out[i]=observed_values[near[rng.integers(len(near))]]
    return out
def mixed_knn_transform(train_original,train_filled,query,num_indices,cat_indices,k):
    a=np.asarray(train_original,dtype=float);filled=np.asarray(train_filled,dtype=float);q=np.asarray(query,dtype=float)
    if a.ndim!=2 or q.ndim!=2 or a.shape!=filled.shape or a.shape[1]!=q.shape[1] or not len(a):raise ValueError("Aligned nonempty training matrices required")
    if not np.isfinite(filled).all() or np.isinf(a).any() or np.isinf(q).any():raise ValueError("Invalid data")
    if int(k)!=k or k<1:raise ValueError("Positive integer k required")
    num=set(num_indices);cat=set(cat_indices)
    if num&cat or num|cat!=set(range(a.shape[1])):raise ValueError("Numeric/categorical partition required")
    ranges=np.ones(a.shape[1])
    for j in num:
        observed=a[np.isfinite(a[:,j]),j]
        if len(observed):ranges[j]=np.ptp(observed) or 1
    out=q.copy()
    for i,row in enumerate(q):
        predictors=np.flatnonzero(np.isfinite(row))
        for j in np.flatnonzero(np.isnan(row)):
            donors=np.flatnonzero(np.isfinite(a[:,j]))
            if not len(donors):out[i,j]=0.;continue
            distances=np.zeros(len(donors))
            for c in predictors:
                if c==j:continue
                distances+=np.abs(filled[donors,c]-row[c])/ranges[c] if c in num else (filled[donors,c]!=row[c])
            nearest=donors[np.argsort(distances,kind="stable")[:min(int(k),len(donors))]]
            values=a[nearest,j]
            if j in num:out[i,j]=np.median(values)
            else:
                levels,counts=np.unique(values,return_counts=True);out[i,j]=levels[np.argmax(counts)]
    return out

