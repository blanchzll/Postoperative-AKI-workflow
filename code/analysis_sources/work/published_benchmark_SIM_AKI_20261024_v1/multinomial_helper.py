import numpy as np

def multinomial_probabilities(X, beta, intercept):
    X = np.asarray(X, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    intercept = np.asarray(intercept, dtype=np.float64)
    
    if X.ndim != 2 or beta.ndim != 2 or intercept.ndim != 1:
        raise ValueError("Invalid input dimensions")
    if X.shape[1] != beta.shape[1] or beta.shape[0] != 2 or intercept.shape[0] != 2:
        raise ValueError("Incompatible dimensions")
    if not (np.isfinite(X).all() and np.isfinite(beta).all() and np.isfinite(intercept).all()):
        raise ValueError("Inputs must contain only finite values")
    
    eta = X @ beta.T + intercept
    eta = np.insert(eta, 0, 0, axis=1)  # Prepend zero column for reference outcome
    
    row_max = np.max(eta, axis=1, keepdims=True)
    stable_eta = eta - row_max
    probabilities = np.exp(stable_eta) / np.sum(np.exp(stable_eta), axis=1, keepdims=True)
    
    return probabilities.astype(np.float64)
