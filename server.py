"""API de inferência e arquivos da interface Risk Bank."""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Annotated, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from threadpoolctl import threadpool_limits

from data import BASE_DIR, FEATURES, clean_features

logger = logging.getLogger(__name__)
ModelKey = Literal['logistic', 'xgboost', 'mlp']
MODEL_NAMES = {'logistic': 'Regressão logística', 'xgboost': 'XGBoost', 'mlp': 'Rede neural MLP'}
EXPLANATION_METHOD = 'Variação da probabilidade em relação à substituição de um grupo pela mediana/moda do treino. Os valores não são aditivos nem causais.'
PayStatus = Annotated[int, Field(strict=True, ge=-2, le=8)]
Money = Annotated[float, Field(allow_inf_nan=False)]
PaidAmount = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, strict=True)
    limit: Annotated[float, Field(gt=0)]
    age: Annotated[int, Field(strict=True, ge=18, le=100)]
    sex: Annotated[int, Field(ge=1, le=2)]
    edu: Annotated[int, Field(ge=1, le=4)]
    mar: Annotated[int, Field(ge=1, le=3)]
    pay: Annotated[list[PayStatus], Field(min_length=6, max_length=6)]
    bill: Annotated[list[Money], Field(min_length=6, max_length=6)]
    paid: Annotated[list[PaidAmount], Field(min_length=6, max_length=6)]
    model: ModelKey | None = None
    threshold: Annotated[float | None, Field(ge=0, le=1)] = None

    def to_frame(self):
        values = [self.limit, self.sex, self.edu, self.mar, self.age, *self.pay, *self.bill, *self.paid]
        return clean_features(pd.DataFrame([values], columns=FEATURES))


class Factor(BaseModel):
    name: str
    value: float


class PredictResponse(BaseModel):
    prob: float
    prediction: int
    threshold: float
    factors: list[Factor]
    model_id: ModelKey
    model_used: str
    explanation_method: str


FACTOR_GROUPS = [
    ('Limite de crédito', ['LIMIT_BAL']),
    ('Idade', ['AGE']), ('Sexo', ['SEX']),
    ('Escolaridade', ['EDUCATION']), ('Estado civil', ['MARRIAGE']),
    ('Histórico de atrasos', FEATURES[5:11]),
    ('Histórico de faturas', FEATURES[11:17]),
    ('Histórico de pagamentos', FEATURES[17:23]),
]


def load_bundle(path):
    # Artefatos joblib devem vir somente do treinamento local ou desta entrega.
    bundle = joblib.load(path)
    if bundle.get('schema_version') != 1 or bundle.get('features') != FEATURES:
        raise ValueError('Artefato com esquema incompatível; execute python train.py.')
    if set(bundle['models']) != set(MODEL_NAMES) or bundle['selected_model'] not in MODEL_NAMES:
        raise ValueError('Artefato incompleto.')
    reference = clean_features(pd.DataFrame([bundle['reference']], columns=FEATURES))
    with threadpool_limits(limits=2):
        for key, model in bundle['models'].items():
            threshold = bundle['report']['validation'][key]['threshold']
            p = float(model.predict_proba(reference)[0, 1])
            if not 0 <= p <= 1 or not 0 <= threshold <= 1:
                raise ValueError('Modelo ou limiar inválido.')
    return bundle


def create_app(model_path: Path | None = None):
    artifact = Path(model_path) if model_path is not None else Path(os.environ.get('RISK_BANK_MODEL', BASE_DIR / 'results/models.joblib'))

    @asynccontextmanager
    async def lifespan(application):
        application.state.bundle = None
        try:
            application.state.bundle = load_bundle(artifact)
        except Exception:
            logger.exception('Não foi possível carregar o modelo. Execute python train.py e reinicie a API.')
        yield

    app = FastAPI(title='Risk Bank API', version='2.0.0', lifespan=lifespan)
    app.state.bundle = None

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_, error):
        # NaN/Infinity na entrada não podem reaparecer em uma resposta JSON.
        detail = [{key: item[key] for key in ('type', 'loc', 'msg')} for item in error.errors()]
        return JSONResponse(status_code=422, content={'detail': detail})

    def get_bundle():
        if app.state.bundle is None:
            raise HTTPException(503, 'Modelo indisponível. Execute python train.py e reinicie o servidor.')
        return app.state.bundle

    @app.get('/health')
    def health():
        bundle = app.state.bundle
        return {'status': 'ready' if bundle else 'not_ready', 'model_loaded': bundle is not None,
                'selected_model': bundle['selected_model'] if bundle else None}

    @app.get('/models')
    def models():
        bundle = get_bundle()
        report = bundle['report']
        return {'selected_model': bundle['selected_model'], 'reference_previously_used': 'research' in report, 'models': [
            {'id': key, 'name': MODEL_NAMES[key], 'threshold': report['validation'][key]['threshold'],
             'calibration_method': report.get('calibration_methods', {}).get(key, 'none'),
             'test_metrics': report['test'][key]} for key in MODEL_NAMES]}

    @app.get('/metrics')
    def metrics():
        return get_bundle()['report']

    @app.post('/predict', response_model=PredictResponse)
    def predict(req: PredictRequest):
        bundle = get_bundle()
        key = req.model or bundle['selected_model']
        model = bundle['models'][key]
        frame = req.to_frame()
        alternatives = [frame]
        for _, columns in FACTOR_GROUPS:
            alternative = frame.copy()
            for col in columns:
                alternative[col] = bundle['reference'][col]
            alternatives.append(alternative)
        try:
            with threadpool_limits(limits=2):
                probabilities = model.predict_proba(pd.concat(alternatives, ignore_index=True))[:, 1]
            if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError('Probabilidades inválidas.')
        except Exception as error:
            logger.exception('Falha na inferência do modelo %s.', key)
            raise HTTPException(500, 'Não foi possível calcular a previsão. Consulte o log do servidor.') from error
        probability = float(probabilities[0])
        threshold = req.threshold if req.threshold is not None else bundle['report']['validation'][key]['threshold']
        factors = [Factor(name=name, value=float(probability - probabilities[i + 1]))
                   for i, (name, _) in enumerate(FACTOR_GROUPS)]
        factors.sort(key=lambda factor: abs(factor.value), reverse=True)
        return PredictResponse(prob=probability, prediction=int(probability >= threshold), threshold=threshold,
                               factors=factors, model_id=key, model_used=MODEL_NAMES[key],
                               explanation_method=EXPLANATION_METHOD)

    @app.get('/')
    def index():
        return FileResponse(BASE_DIR / 'index.html')

    @app.get('/app.js')
    def javascript():
        return FileResponse(BASE_DIR / 'app.js', media_type='application/javascript')

    @app.get('/style.css')
    def stylesheet():
        return FileResponse(BASE_DIR / 'style.css', media_type='text/css')

    return app


app = create_app()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='127.0.0.1', port=8000)
