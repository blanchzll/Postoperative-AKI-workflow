import numpy as np

def summarize_pairs(values):
    if not isinstance(values, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    
    if values.ndim != 5 or values.shape[-1] != 2:
        raise ValueError("Input must have shape (B, R, N, P, 2).")
    
    if values.dtype.kind not in 'bifu' or not np.isfinite(values).all():
        raise ValueError("Input must contain only finite values.")
    
    B, R, N, P, _ = values.shape
    
    if B < 2 or R < 2 or N < 1 or P < 1:
        raise ValueError("B and R must be at least 2.")
    
    mean = values.mean(axis=(0, 1))
    variances = np.var(values, axis=1, ddof=1)
    mc_se = np.sqrt(np.sum(variances, axis=0) / R) / B
    mean_absolute_importance = np.abs(mean).mean(axis=0)
    
    return {
        'mean': mean,
        'mc_se': mc_se,
        'mean_absolute_importance': mean_absolute_importance
    }
