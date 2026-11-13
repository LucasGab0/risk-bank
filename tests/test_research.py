import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
from data import BASE_DIR, DEFAULT_DATA, FEATURES, load_dataset, split_dataset
from neural_experiments import NeuralConfig, probabilities, train_network
from research import bootstrap_intervals, development_split

RESULTS = BASE_DIR / 'results/research'


def test_five_seed_protocol_has_no_overlap():
    features, target = load_dataset(DEFAULT_DATA)
    _, _, reference = split_dataset(features, target, 42)
    for seed in [7,21,42,84,123]:
        split = development_split(features, target, reference, seed)
        saved = np.load(RESULTS / f'split_seed_{seed}.npz')
        for key, indices in split.items():
            np.testing.assert_array_equal(indices, saved[key])
        groups = [set(pd.util.hash_pandas_object(features.iloc[idx], index=False))
                  for idx in [*split.values(),reference]]
        assert sum(len(idx) for idx in split.values()) + len(reference) == len(features)
        for i,group in enumerate(groups):
            for other in groups[i+1:]:
                assert not group & other


@pytest.mark.parametrize('options', [{'dropout':0.3}, {'batch_norm':True}, {'l2':0.001}, {'balance':'weight'}, {'balance':'oversample'}])
def test_neural_training_variants_are_reproducible_and_finite(options):
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 5))
    y = (x[:,0] > 0.4).astype(int)
    config = NeuralConfig('test','P3','test',hidden=(8,4),batch_size=32,**options)
    first, curve, info = train_network(x[:80],y[:80],x[80:],y[80:],config,42,max_epochs=4)
    second, _, _ = train_network(x[:80],y[:80],x[80:],y[80:],config,42,max_epochs=4)
    np.testing.assert_array_equal(probabilities(first,x), probabilities(second,x))
    assert np.isfinite(pd.DataFrame(curve).to_numpy()).all()
    first.eval()
    np.testing.assert_array_equal(probabilities(first,x), probabilities(first,x))
    if options.get('balance') == 'oversample':
        assert info['fit_rows_after_sampling'] == 2 * np.sum(y[:80] == 0)


def test_selection_matches_validation_only_and_protocol_hash():
    protocol_path = RESULTS / 'protocol.json'
    selection = json.loads((RESULTS/'selection_locked.json').read_text())
    assert selection['protocol_sha256'] == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    summary = pd.read_csv(RESULTS/'summary.csv')
    eligible = summary[summary.model.isin(['logistic','xgboost','mlp'])]
    assert selection['selected_application_model'] == eligible.sort_values('average_precision_mean',ascending=False).iloc[0]['model']
    neural = summary[summary.model.str.startswith('nn_')]
    assert selection['best_neural_configuration'] == neural.sort_values('average_precision_mean',ascending=False).iloc[0]['model']
    calibrations = pd.read_csv(RESULTS/'calibration_summary.csv')
    for key, method in selection['calibration_methods'].items():
        assert method == calibrations[calibrations.model == key].sort_values('brier_score_mean').iloc[0]['method']
    runs = pd.read_csv(RESULTS/'experiments.csv')
    assert len(runs) == 140
    assert runs.groupby('model').seed.nunique().eq(5).all()


def test_integrated_calibration_matches_independent_saved_predictions():
    features, _ = load_dataset(DEFAULT_DATA)
    bundle = joblib.load(BASE_DIR/'results/models.joblib')
    for key, model in bundle['models'].items():
        saved = np.load(RESULTS/f'reference_predictions_{key}.npz')
        np.testing.assert_allclose(model.predict_proba(features.iloc[saved['indices']])[:,1], saved['probability'], atol=1e-12)
        assert model.method == bundle['report']['calibration_methods'][key]


def test_cluster_bootstrap_handles_tied_probabilities():
    features, target = load_dataset(DEFAULT_DATA)
    subset = features.iloc[:100]
    probability = np.full(100,0.5)
    result = bootstrap_intervals(subset,target.iloc[:100],probability,0.5,repetitions=50)
    assert result['roc_auc']['low'] == pytest.approx(0.5)
    assert result['roc_auc']['high'] == pytest.approx(0.5)
    assert all(0 <= interval['low'] <= interval['high'] <= 1 for interval in result.values())
