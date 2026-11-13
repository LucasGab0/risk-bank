import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from data import BASE_DIR, DEFAULT_DATA, FEATURES, NUMERIC, TARGET, clean_features, load_dataset, split_dataset
from train import choose_threshold


@pytest.fixture(scope='module')
def dataset():
    return load_dataset(DEFAULT_DATA)


def test_data_source_integrity_and_target_exclusion(dataset):
    source = json.loads((BASE_DIR / 'data/source.json').read_text())
    assert hashlib.sha256(DEFAULT_DATA.read_bytes()).hexdigest() == source['csv_gz_sha256']
    features, target = dataset
    assert features.shape == (30000,23)
    assert target.sum() == 6636
    assert list(features.columns) == FEATURES
    assert 'ID' not in features and TARGET not in features
    assert features['EDUCATION'].isin([1,2,3,4]).all()
    assert features['MARRIAGE'].isin([1,2,3]).all()


def test_partitions_disjoint_grouped_and_reproducible(dataset):
    features, target = dataset
    saved = np.load(BASE_DIR / 'results/split_indices.npz')
    original = split_dataset(features, target, 42)
    assert np.array_equal(saved['test'], original[2])
    parts = [saved[key] for key in saved.files]
    if len(parts) == 3:
        for indices, key in zip(original, ['train', 'validation', 'test']):
            assert np.array_equal(indices, saved[key])
    for indices in parts:
        assert abs(target.iloc[indices].mean() - target.mean()) < 0.01
    assert len(set(np.concatenate(parts))) == len(features)
    profiles = [set(pd.util.hash_pandas_object(features.iloc[idx], index=False)) for idx in parts]
    for i, group in enumerate(profiles):
        for other in profiles[i+1:]:
            assert not group & other


def test_scalers_fitted_only_on_training(dataset):
    features, _ = dataset
    bundle = joblib.load(BASE_DIR / 'results/models.joblib')
    saved = np.load(BASE_DIR / 'results/split_indices.npz')
    train = features.iloc[saved['train']]
    for pipeline in bundle['models'].values():
        scaler = pipeline.named_steps['preprocessor'].named_transformers_['numeric']
        np.testing.assert_allclose(scaler.mean_, train[NUMERIC].mean().values)
        assert scaler.n_samples_seen_ == len(train)


def test_threshold_maximizes_validation_f1():
    labels = np.array([0,0,0,1,1,1])
    probability = np.array([0.1,0.2,0.7,0.4,0.6,0.9])
    threshold = choose_threshold(labels, probability)
    expected = max(f1_score(labels, probability >= t) for t in probability)
    assert f1_score(labels, probability >= threshold) == pytest.approx(expected)


def test_alias_csv_and_excel_loading(tmp_path, dataset):
    features, target = dataset
    indices = np.r_[np.flatnonzero(target == 0)[:30], np.flatnonzero(target == 1)[:30]]
    frame = features.iloc[indices].copy()
    frame.columns = [f'X{i}' for i in range(1,24)]
    frame['Y'] = target.iloc[indices].values
    for extension in ['csv', 'xlsx']:
        path = tmp_path / ('credit.' + extension)
        if extension == 'csv':
            frame.to_csv(path, index=False)
        else:
            frame.to_excel(path, index=False)
        loaded_x, loaded_y = load_dataset(path)
        np.testing.assert_array_equal(loaded_x.values, features.iloc[indices].values)
        np.testing.assert_array_equal(loaded_y.values, target.iloc[indices].values)


@pytest.mark.parametrize('case', ['missing_target','invalid_target','missing_feature','infinite','category','payment','fractional_age','duplicate_column'])
def test_bad_datasets_fail_explicitly(tmp_path, dataset, case):
    features, target = dataset
    frame = features.iloc[:100].copy()
    frame[TARGET] = target.iloc[:100]
    if case == 'missing_target': frame = frame.drop(columns=TARGET)
    if case == 'invalid_target': frame.loc[0, TARGET] = 3
    if case == 'missing_feature': frame = frame.drop(columns='AGE')
    if case == 'infinite':
        frame['LIMIT_BAL'] = frame['LIMIT_BAL'].astype(float)
        frame.loc[0, 'LIMIT_BAL'] = np.inf
    if case == 'category': frame.loc[0, 'SEX'] = 5
    if case == 'payment': frame.loc[0, 'PAY_AMT1'] = -1
    if case == 'fractional_age': frame['AGE'] = frame['AGE'].astype(float); frame.loc[0, 'AGE'] = 25.5
    if case == 'duplicate_column': frame['PAY_1'] = frame['PAY_0']
    path = tmp_path / 'invalid.csv'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError): load_dataset(path)


def test_reports_and_baseline(dataset):
    report = json.loads((BASE_DIR / 'results/metrics.json').read_text())
    assert sum(s['rows'] for s in report['split'].values()) == 30000
    assert report['baseline']['recall'] == 0
    assert report['baseline']['f1'] == 0
    assert not any(report['training_warnings'].values())
    for key in report['test']:
        assert report['test'][key]['roc_auc'] > report['baseline']['roc_auc']
        assert np.asarray(report['test'][key]['confusion_matrix']).sum() == report['split']['test']['rows']
        assert sum(g['n'] for g in report['group_metrics_test'][key] if g['attribute'] == 'SEX') == report['split']['test']['rows']
