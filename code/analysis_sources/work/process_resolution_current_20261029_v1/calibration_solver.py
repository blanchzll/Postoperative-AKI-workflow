import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit

def _validate(y, p, w):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    if y.ndim != 1 or p.ndim != 1 or w.ndim != 1:
        raise ValueError("y, p, w must be 1-dimensional arrays")
    if not (y.shape == p.shape == w.shape):
        raise ValueError("y, p, w must have the same shape")
    if y.size == 0:
        raise ValueError("arrays must be non-empty")

    if not np.all(np.isfinite(p)):
        raise ValueError("p must contain only finite values")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p must be in [0, 1]")

    if not np.all(np.isfinite(y)):
        raise ValueError("y must contain only finite values")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("y must be binary (0 or 1)")

    if not np.all(np.isfinite(w)):
        raise ValueError("w must contain only finite values")
    if np.any(w < 0.0):
        raise ValueError("w must be non-negative")

    total_w = np.sum(w)
    if total_w <= 0.0:
        raise ValueError("sum of weights must be > 0")
    return y, p, w

def fit_update(y, p, w, kind, sd=2.5):
    y, p, w = _validate(y, p, w)

    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("sd must be positive")

    eps = 1e-6
    p_clipped = np.clip(p, eps, 1.0 - eps)
    z = np.log(p_clipped / (1.0 - p_clipped))

    if kind == 'offset':
        def score(a):
            eta = z + a
            return np.sum(w * (expit(eta) - y)) + a / (sd * sd)

        a_opt = brentq(score, -80.0, 80.0, xtol=1e-12, rtol=1e-12, maxiter=1000)

        return np.array([a_opt, 1.0], dtype=np.float64)

    elif kind == 'logistic':
        def objective(params):
            a, b = params
            eta = a + b * z
            # logaddexp(0, eta) - y * eta
            loss = np.sum(w * (np.logaddexp(0.0, eta) - y * eta))
            penalty = (a * a + (b - 1.0) ** 2) / (2.0 * sd * sd)
            return loss + penalty

        def gradient(params):
            a, b = params
            eta = a + b * z
            # d/da: sum w * (expit(eta) - y) + a / sd^2
            # d/db: sum w * (expit(eta) - y) * z + (b - 1) / sd^2
            residual = expit(eta) - y
            grad_a = np.sum(w * residual) + a / (sd * sd)
            grad_b = np.sum(w * residual * z) + (b - 1.0) / (sd * sd)
            return np.array([grad_a, grad_b], dtype=np.float64)

        x0 = np.array([0.0, 1.0], dtype=np.float64)
        bounds = [(-np.inf, np.inf), (1e-6, np.inf)]

        res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-8})

        coef = np.asarray(res.x, dtype=np.float64)
        # Numerical Newton refinement of the identical convex objective.
        for _ in range(15):
            grad = gradient(coef)
            projected = grad.copy()
            active = coef[1] <= 1e-6 + 1e-10 and grad[1] >= 0
            if active:
                projected[1] = 0.0
            if np.max(np.abs(projected)) <= 1e-7:
                break
            v = w * expit(coef[0] + coef[1]*z) * (1-expit(coef[0] + coef[1]*z))
            hessian = np.array([[v.sum(), np.sum(v*z)], [np.sum(v*z), np.sum(v*z*z)]])
            hessian += np.eye(2)/(sd*sd)
            delta = np.array([-grad[0]/hessian[0,0],0.0]) if active else -np.linalg.solve(hessian,grad)
            step = 1.0
            if delta[1] < 0:
                step = min(step,max(0.0,(coef[1]-1e-6)/(-delta[1])))
            prior = objective(coef)
            accepted = False
            for _ in range(40):
                candidate = coef + step*delta
                candidate[1] = max(1e-6,candidate[1])
                cg = gradient(candidate)
                if candidate[1] <= 1e-6+1e-10:
                    cg[1] = min(0.0,cg[1])
                if objective(candidate) <= prior + 1e-12 and np.max(np.abs(cg)) < np.max(np.abs(projected)):
                    coef = candidate
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
        grad = gradient(coef)
        if coef[1] <= 1e-6 + 1e-10:
            grad[1] = min(0.0,grad[1])
        if not np.isfinite(coef).all() or np.max(np.abs(grad)) > 1e-4:
            raise ValueError("Optimization projected score did not converge")
        if coef[1] < 1e-6:
            raise ValueError("Slope constraint violated")
        return coef

    else:
        raise ValueError("kind must be 'offset' or 'logistic'")

def apply_update(p, coef):
    p = np.asarray(p, dtype=np.float64)
    coef = np.asarray(coef, dtype=np.float64)
    
    if p.ndim != 1:
        raise ValueError("p must be 1-dimensional")
    if coef.shape != (2,):
        raise ValueError("coef must be a 2-element array")
    if not np.all(np.isfinite(p)):
        raise ValueError("p must contain only finite values")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p must be in [0, 1]")
    if not np.all(np.isfinite(coef)):
        raise ValueError("coef must contain only finite values")

    a, b = coef
    eps = 1e-6
    p_clipped = np.clip(p, eps, 1.0 - eps)
    z = np.log(p_clipped / (1.0 - p_clipped))
    eta = a + b * z
    return expit(eta)