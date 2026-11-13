import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data import BASE_DIR
from server import FACTOR_GROUPS, PredictRequest, create_app


def test_readiness_and_selected_model(client):
    assert client.get('/health').json() == {'status': 'ready', 'model_loaded': True, 'selected_model': 'xgboost'}
    catalog = client.get('/models').json()
    assert {m['id'] for m in catalog['models']} == {'logistic', 'xgboost', 'mlp'}
    metrics = client.get('/metrics').json()
    if 'research' in metrics:
        mean_ap = metrics['research']['decisions']['application_mean_validation_ap']
        assert metrics['selected_model'] == max(mean_ap, key=mean_ap.get)
    else:
        assert metrics['selected_model'] == max(metrics['validation'], key=lambda k: metrics['validation'][k]['average_precision'])
    for model in catalog['models']:
        assert 0 <= model['threshold'] <= 1
        assert 0 <= model['test_metrics']['roc_auc'] <= 1


@pytest.mark.parametrize('model', ['logistic', 'xgboost', 'mlp'])
def test_api_matches_saved_pipeline_and_explanations(client, payload, model):
    payload['model'] = model
    result = client.post('/predict', json=payload)
    assert result.status_code == 200
    result = result.json()
    bundle = joblib.load(BASE_DIR / 'results/models.joblib')
    frame = PredictRequest(**payload).to_frame()
    pipeline = bundle['models'][model]
    expected = pipeline.predict_proba(frame)[0, 1]
    assert result['prob'] == pytest.approx(expected, abs=1e-7)
    assert result['model_id'] == model
    assert result['prediction'] == int(result['prob'] >= result['threshold'])
    assert len(result['factors']) == len(FACTOR_GROUPS)
    by_name = {item['name']: item['value'] for item in result['factors']}
    for name, columns in FACTOR_GROUPS:
        alternative = frame.copy()
        for column in columns:
            alternative[column] = bundle['reference'][column]
        assert by_name[name] == pytest.approx(expected - pipeline.predict_proba(alternative)[0, 1], abs=1e-7)


@pytest.mark.parametrize('model', ['logistic', 'xgboost', 'mlp'])
def test_examples_order_risk(client, payload, model):
    payload.update(model=model, limit=200000, age=42, pay=[-1]*6,
                   bill=[12000,11000,9000,10000,8000,9500], paid=[12000,11000,9000,10000,8000,9500])
    low = client.post('/predict', json=payload).json()['prob']
    payload.update(limit=30000, age=27, pay=[3,3,2,2,2,0],
                   bill=[29500,28000,27000,26000,25000,22000], paid=[0,0,1000,0,500,1500])
    high = client.post('/predict', json=payload).json()['prob']
    assert 0 <= low < high <= 1


def test_threshold_changes_decision_only(client, payload):
    low = client.post('/predict', json={**payload, 'threshold': 0}).json()
    high = client.post('/predict', json={**payload, 'threshold': 1}).json()
    assert low['prediction'] == 1
    assert high['prediction'] == 0
    assert low['prob'] == high['prob']
    assert low['factors'] == high['factors']


@pytest.mark.parametrize('field,value', [
    ('limit', 0), ('limit', -1), ('limit', True), ('limit', '50000'),
    ('age', 17), ('age', 101), ('age', 30.5), ('age', True),
    ('sex', 0), ('sex', 3), ('sex', True), ('edu', 0), ('edu', 5), ('mar', 0),
    ('pay', [0]*5), ('pay', [0]*7), ('pay', [-3]*6), ('pay', [9]*6), ('pay', [0.5]*6),
    ('bill', [0]*5), ('paid', [-1]*6), ('paid', [0]*7),
    ('model', 'unknown'), ('threshold', -0.1), ('threshold', 1.1), ('other', 5),
])
def test_invalid_inputs_rejected(client, payload, field, value):
    result = client.post('/predict', json={**payload, field: value})
    assert result.status_code == 422
    assert result.json()['detail']


@pytest.mark.parametrize('field,value', [('limit', float('nan')), ('limit', float('inf')),
                                         ('bill', [float('inf')]*6), ('paid', [float('nan')]*6),
                                         ('threshold', float('nan'))])
def test_nonfinite_input_returns_validation_error(client, payload, field, value):
    result = client.post('/predict', content=json.dumps({**payload, field: value}), headers={'Content-Type': 'application/json'})
    assert result.status_code == 422
    assert result.json()['detail']


def test_missing_invalid_and_extra_input(client, payload):
    incomplete = {k: v for k,v in payload.items() if k != 'pay'}
    assert client.post('/predict', json=incomplete).status_code == 422
    assert client.post('/predict', content='{invalid', headers={'Content-Type':'application/json'}).status_code == 422
    assert client.post('/predict', json={**payload, 'bill': [-200]*6}).status_code == 200
    assert client.post('/predict', json={**payload, 'pay': [-2,-1,0,1,7,8]}).status_code == 200


@pytest.mark.parametrize('broken', [False, True])
def test_unavailable_artifact_never_falls_back(tmp_path, payload, broken):
    path = tmp_path / 'missing.joblib'
    if broken:
        path.write_text('invalid artifact')
    with TestClient(create_app(path)) as client:
        assert client.get('/health').json()['model_loaded'] is False
        assert client.get('/models').status_code == 503
        result = client.post('/predict', json=payload)
        assert result.status_code == 503
        assert 'prob' not in result.json()
        assert client.get('/').status_code == 200


def test_inference_error_returns_controlled_message(client, payload, monkeypatch):
    selected = client.app.state.bundle['selected_model']
    pipeline = client.app.state.bundle['models'][selected]
    def fail(*args, **kwargs):
        raise ValueError('private internal details')
    monkeypatch.setattr(pipeline, 'predict_proba', fail)
    result = client.post('/predict', json=payload)
    assert result.status_code == 500
    assert 'private internal details' not in result.text


@pytest.mark.parametrize('path,content_type', [('/', 'text/html'), ('/app.js', 'application/javascript'),
                                             ('/style.css', 'text/css'), ('/docs', 'text/html')])
def test_frontend_assets(client, path, content_type):
    response = client.get(path)
    assert response.status_code == 200
    assert content_type in response.headers['content-type']


def test_no_path_traversal(client):
    assert client.get('/data/source.json').status_code == 404
    assert client.get('/results/models.joblib').status_code == 404
    assert client.get('/predict').status_code == 405
