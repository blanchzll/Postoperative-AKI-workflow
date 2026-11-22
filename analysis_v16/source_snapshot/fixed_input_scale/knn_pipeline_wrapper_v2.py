"""Portable deterministic prediction adapter around an unchanged fitted KNN pipe."""
import numpy as np
from knn_exact_query import predict_fitted_pipeline


class ExactEuclideanKNNPipeline:
    def __init__(self, fitted_pipeline):
        self.fitted_pipeline = fitted_pipeline

    def predict_proba(self, frame):
        positive = predict_fitted_pipeline(self.fitted_pipeline, frame)
        return np.column_stack([1 - positive, positive])
