"""Version-bound exact distance weighting for an already fitted Euclidean KNN.

Use the fitted classifier's singleton neighbor identities and recompute distances
by direct coordinate subtraction, avoiding cancellation in squared-norm kernels.
No fit, parameter update or alternative neighbor-count selection occurs here.
"""
import numpy as np


def predict_fitted_pipeline(pipe, frame):
    model = pipe.named_steps["model"]
    assert model.weights == "distance" and model.p == 2
    assert model.metric == "minkowski"
    np.testing.assert_array_equal(model.classes_, [0, 1])
    encoded = pipe[:-1].transform(frame)
    probabilities = []
    for row in encoded:
        _, indices = model.kneighbors(row.reshape(1, -1), return_distance=True)
        indices = indices[0]
        delta = model._fit_X[indices] - row
        distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        zero = distance == 0
        weights = zero.astype(float) if zero.any() else 1 / distance
        outcomes = model._y[indices]
        probabilities.append(float(np.sum(weights * outcomes) / np.sum(weights)))
    return np.asarray(probabilities)


def predict_bundle(bundle, raw, keys):
    filled = bundle["imputer"].transform(raw[bundle["input_columns"]], keys)
    return predict_fitted_pipeline(bundle["pipeline"], filled)
