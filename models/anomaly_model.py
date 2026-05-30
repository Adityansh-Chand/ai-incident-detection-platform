from dataclasses import dataclass

import numpy as np


DEFAULT_MEAN = np.array([150.0, 0.5, 0.2, 1000.0])
DEFAULT_STD = np.array([60.0, 1.0, 1.0, 300.0])
DEFAULT_THRESHOLD = 3.0


@dataclass
class AnomalyModel:
    mean: np.ndarray
    std: np.ndarray
    threshold: float = DEFAULT_THRESHOLD

    def score(self, features):
        features = np.asarray(features, dtype=float)
        z_scores = np.abs((features - self.mean) / self.std)
        return float(np.max(z_scores))

    def predict(self, features):
        anomaly_score = self.score(features)
        return {
            "score": anomaly_score,
            "is_anomaly": anomaly_score >= self.threshold,
        }


def fit(records, threshold=DEFAULT_THRESHOLD):
    matrix = np.asarray(records, dtype=float)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    return AnomalyModel(mean=matrix.mean(axis=0), std=std, threshold=threshold)


def default_model():
    return AnomalyModel(DEFAULT_MEAN, DEFAULT_STD)


def score(features):
    return default_model().score(features)


def is_anomaly(features, threshold=DEFAULT_THRESHOLD):
    return score(features) >= threshold
