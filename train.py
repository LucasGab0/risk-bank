"""Treina, compara e salva modelos de classificação de inadimplência."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from data import BASE_DIR, CATEGORICAL, DEFAULT_DATA, FEATURES, NUMERIC, load_dataset, split_dataset

MODEL_NAMES = {"logistic": "Regressão logística", "xgboost": "XGBoost", "mlp": "Rede neural MLP"}


def make_pipeline(kind, seed=42, mlp_epochs=120):
    processor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    models = {
        "logistic": LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
        "xgboost": XGBClassifier(
            n_estimators=350, max_depth=3, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=5.0,
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_jobs=2, random_state=seed,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", alpha=0.01,
            batch_size=256, learning_rate_init=0.001, max_iter=mlp_epochs,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=12, random_state=seed,
        ),
    }
    return Pipeline([("preprocessor", processor), ("classifier", models[kind])])


def choose_threshold(y, probability):
    precision, recall, thresholds = precision_recall_curve(y, probability)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(scores))])


def metrics(y, probability, threshold):
    predicted = np.asarray(probability) >= threshold
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def group_metrics(features, y, probabilities, threshold):
    table = []
    groups = {
        "SEX": features["SEX"], "MARRIAGE": features["MARRIAGE"],
        "AGE_GROUP": pd.cut(features["AGE"], [17, 29, 44, 59, 100], labels=["18–29", "30–44", "45–59", "60–100"]),
    }
    for column, values in groups.items():
        for value in sorted(values.dropna().unique(), key=str):
            mask = np.asarray(values == value)
            predicted = probabilities[mask] >= threshold
            actual = np.asarray(y)[mask]
            negatives = actual == 0
            table.append({
                "attribute": column, "group": str(value), "n": int(mask.sum()),
                "default_rate": float(actual.mean()),
                "recall": float(recall_score(actual, predicted, zero_division=0)),
                "false_positive_rate": float(predicted[negatives].mean()) if negatives.any() else None,
            })
    return table


def atomic_bundle(bundle, path):
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".joblib")
    os.close(fd)
    try:
        joblib.dump(bundle, temp, compress=3)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def train(data_path=DEFAULT_DATA, output=BASE_DIR / "results", seed=42, mlp_epochs=120):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    features, target = load_dataset(Path(data_path))
    train_idx, val_idx, test_idx = split_dataset(features, target, seed)
    xtr, xva, xte = (features.iloc[i] for i in (train_idx, val_idx, test_idx))
    ytr, yva, yte = (target.iloc[i] for i in (train_idx, val_idx, test_idx))
    trained, validation, train_warnings = {}, {}, {}
    for key, name in MODEL_NAMES.items():
        print(f"Treinando {name}...", flush=True)
        pipeline = make_pipeline(key, seed, mlp_epochs)
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=2):
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(xtr, ytr)
        train_warnings[key] = [str(w.message) for w in captured]
        probability = pipeline.predict_proba(xva)[:, 1]
        validation[key] = metrics(yva, probability, choose_threshold(yva, probability))
        trained[key] = pipeline
        print(f"  AP validação: {validation[key]['average_precision']:.4f}", flush=True)
    # A escolha termina antes de consultar os rótulos do teste.
    selected = max(validation, key=lambda key: validation[key]["average_precision"])
    print(f"Modelo selecionado pela validação: {MODEL_NAMES[selected]}", flush=True)
    reference = features.iloc[train_idx].median().to_dict()
    for col in CATEGORICAL:
        reference[col] = float(xtr[col].mode().iloc[0])
    test, probabilities, fairness = {}, {}, {}
    for key, pipeline in trained.items():
        probabilities[key] = pipeline.predict_proba(xte)[:, 1]
        threshold = validation[key]["threshold"]
        test[key] = metrics(yte, probabilities[key], threshold)
        fairness[key] = group_metrics(xte, yte, probabilities[key], threshold)
    dummy = DummyClassifier(strategy="prior").fit(xtr, ytr)
    dummy_probability = dummy.predict_proba(xte)[:, 1]
    baseline = metrics(yte, dummy_probability, 0.5)
    versions = {package: importlib.metadata.version(package) for package in
                ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib"]}
    versions["python"] = platform.python_version()
    data_sha = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    report = {
        "schema_version": 1, "selected_model": selected, "selection_metric": "validation.average_precision",
        "seed": seed, "dataset": {"file": Path(data_path).name, "sha256": data_sha,
            "rows": len(features), "features": FEATURES, "default_rate": float(target.mean()),
            "repeated_feature_profiles": int(features.duplicated().sum())},
        "split": {name: {"rows": len(idx), "default_rate": float(target.iloc[idx].mean())}
                  for name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]},
        "split_method": "Perfis idênticos agrupados; grupos estratificados pelo maior alvo; proporção aproximada 70/15/15.",
        "validation": validation, "test": test, "baseline": baseline,
        "group_metrics_test": fairness, "training_warnings": train_warnings, "versions": versions,
        "parameters": {key: {name: ("NaN" if isinstance(value, float) and not np.isfinite(value) else value)
                             for name, value in pipeline.named_steps["classifier"].get_params().items()}
                       for key, pipeline in trained.items()},
        "notes": ["Pré-processamento ajustado somente no treino.",
                  "Limiar maximiza F1 na validação; modelo maximiza average precision na validação.",
                  "MLP usa parada antecipada com subdivisão interna do treino.",
                  "Probabilidades não passaram por calibração adicional; Brier e log loss são reportados.",
                  "Métricas por grupo são descritivas e não comprovam ausência de viés."],
    }
    bundle = {"schema_version": 1, "features": FEATURES, "selected_model": selected,
              "models": trained, "reference": reference, "report": report}
    atomic_bundle(bundle, output / "models.joblib")
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    rows = [{"model": MODEL_NAMES[key], "selected": key == selected,
             **{k: v for k, v in test[key].items() if k != "confusion_matrix"}} for key in trained]
    pd.DataFrame(rows).to_csv(output / "resultados.csv", index=False)
    pd.DataFrame([{ "model": key, **row} for key in fairness for row in fairness[key]]).to_csv(output / "metricas_por_grupo.csv", index=False)
    np.savez_compressed(output / "split_indices.npz", train=train_idx, validation=val_idx, test=test_idx)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for key, probability in probabilities.items():
        RocCurveDisplay.from_predictions(yte, probability, name=MODEL_NAMES[key], ax=axes[0])
        PrecisionRecallDisplay.from_predictions(yte, probability, name=MODEL_NAMES[key], ax=axes[1])
    axes[1].axhline(float(yte.mean()), color="gray", linestyle="--", label="Prevalência")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "curvas_avaliacao.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key in zip(axes, trained):
        ConfusionMatrixDisplay(np.array(test[key]["confusion_matrix"]), display_labels=["Adimplente", "Inadimplente"]).plot(ax=ax, colorbar=False)
        ax.set_title(MODEL_NAMES[key])
    fig.tight_layout()
    fig.savefig(output / "matrizes_confusao.png", dpi=150)
    plt.close(fig)
    mlp = trained["mlp"].named_steps["classifier"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(mlp.loss_curve_)
    axes[0].set(title="MLP: perda no treino", xlabel="Época", ylabel="Perda")
    axes[1].plot(mlp.validation_scores_)
    axes[1].set(title="MLP: acurácia na validação interna", xlabel="Época", ylabel="Acurácia")
    fig.tight_layout()
    fig.savefig(output / "curvas_mlp.png", dpi=150)
    plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", "--csv", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlp-epochs", type=int, default=120)
    args = parser.parse_args(argv)
    if args.mlp_epochs < 1:
        parser.error("--mlp-epochs deve ser positivo")
    try:
        train(args.data, args.output, args.seed, args.mlp_epochs)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Erro: {error}\n")


if __name__ == "__main__":
    main()
