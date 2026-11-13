"""Calibração de probabilidades com uma partição independente do ajuste do modelo."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def logit_column(probability):
    values = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values)).reshape(-1, 1)


def fit_calibrator(method, probability, target):
    if method == "none":
        return None
    if method == "sigmoid":
        return LogisticRegression(C=1e6, max_iter=1000).fit(logit_column(probability), target)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(probability, target)
    raise ValueError(f"Método de calibração desconhecido: {method}")


def apply_calibrator(method, calibrator, probability):
    values = np.asarray(probability, dtype=float)
    if method == "none":
        return values
    if method == "sigmoid":
        return calibrator.predict_proba(logit_column(values))[:, 1]
    if method == "isotonic":
        return calibrator.predict(values)
    raise ValueError(f"Método de calibração desconhecido: {method}")


class CalibratedCreditModel(ClassifierMixin, BaseEstimator):
    def __init__(self, base_model, method="none"):
        self.base_model = base_model
        self.method = method

    def fit(self, features, target):
        # O pipeline base já foi ajustado; esta etapa não o treina novamente.
        probability = self.base_model.predict_proba(features)[:, 1]
        self.calibrator_ = fit_calibrator(self.method, probability, target)
        self.classes_ = np.array([0, 1])
        return self

    @property
    def named_steps(self):
        return self.base_model.named_steps

    def predict_proba(self, features):
        raw = self.base_model.predict_proba(features)[:, 1]
        probability = apply_calibrator(self.method, self.calibrator_, raw)
        return np.column_stack([1 - probability, probability])

    def predict(self, features):
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)
