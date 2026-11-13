import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from calibration import CalibratedCreditModel, apply_calibrator, fit_calibrator


@pytest.mark.parametrize('method', ['none', 'sigmoid', 'isotonic'])
def test_calibrators_finite_monotonic_and_bounded(method):
    probability = np.linspace(0, 1, 100)
    target = (np.arange(100) % 4 == 0).astype(int)
    fitted = fit_calibrator(method, probability, target)
    output = apply_calibrator(method, fitted, probability)
    assert np.isfinite(output).all()
    assert ((output >= 0) & (output <= 1)).all()
    if method == 'isotonic':
        assert (np.diff(output) >= 0).all()
        assert np.isfinite(apply_calibrator(method, fitted, [-0.1, 1.1])).all()
    if method == 'none':
        np.testing.assert_array_equal(output, probability)


def test_calibration_does_not_refit_base_model():
    values = np.arange(100).reshape(-1, 1)
    target = (values[:, 0] > 50).astype(int)
    base = LogisticRegression().fit(values[:70], target[:70])
    coefficients = base.coef_.copy()
    wrapper = CalibratedCreditModel(base, 'sigmoid').fit(values[::3], target[::3])
    np.testing.assert_array_equal(coefficients, base.coef_)
    output = wrapper.predict_proba(values)
    np.testing.assert_allclose(output.sum(axis=1), 1)
    assert np.array_equal(wrapper.classes_, [0, 1])


def test_unknown_calibration_method_fails():
    with pytest.raises(ValueError):
        fit_calibrator('unknown', [0.1,0.9], [0,1])
