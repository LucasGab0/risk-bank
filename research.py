"""Experimentos P1–P4, calibração e avaliação com cinco sementes."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t as student_t
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
import torch

from calibration import CalibratedCreditModel, apply_calibrator, fit_calibrator
from data import BASE_DIR, CATEGORICAL, DEFAULT_DATA, FEATURES, NUMERIC, load_dataset, split_dataset
from neural_experiments import configurations, probabilities, train_network
from train import MODEL_NAMES, atomic_bundle, choose_threshold, group_metrics, make_pipeline, metrics

SEEDS = [7, 21, 42, 84, 123]
SENSITIVE = ["SEX", "AGE", "MARRIAGE"]
BASELINES = {"logistic": "Regressão logística", "xgboost": "XGBoost",
             "mlp": "MLP scikit-learn", "knn_l1": "kNN L1 (k=25)",
             "knn_l2": "kNN L2 (k=25)", "svm": "SVM linear + sigmoide",
             "random_forest": "Random forest"}
SCORE_KEYS = ["average_precision", "roc_auc", "f1", "recall", "precision", "brier_score", "log_loss"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def development_split(features, target, reference_test, seed):
    available = np.setdiff1d(np.arange(len(features)), reference_test)
    local = features.iloc[available].reset_index(drop=True)
    groups = local.groupby(FEATURES, sort=False, dropna=False).ngroup()
    labels = pd.DataFrame({"group": groups, "y": target.iloc[available].to_numpy()}).groupby("group")["y"].max()
    fit, rest = train_test_split(labels.index, test_size=0.35, stratify=labels, random_state=seed)
    stop, rest = train_test_split(rest, test_size=5 / 7, stratify=labels.loc[rest], random_state=seed + 1)
    cal, val = train_test_split(rest, test_size=0.60, stratify=labels.loc[rest], random_state=seed + 2)
    return {name: available[np.flatnonzero(groups.isin(part))]
            for name, part in [("fit", fit), ("stop", stop), ("calibration", cal), ("validation", val)]}


def processor(scale=True, one_hot=True, without_sensitive=False):
    numeric = [col for col in NUMERIC if not without_sensitive or col not in SENSITIVE]
    categorical = [col for col in CATEGORICAL if not without_sensitive or col not in SENSITIVE]
    return ColumnTransformer([
        ("numeric", StandardScaler() if scale else "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False) if one_hot else "passthrough", categorical),
    ])


def baseline_pipeline(key, seed, without_sensitive=False):
    if key in MODEL_NAMES:
        model = make_pipeline(key, seed).named_steps["classifier"]
    elif key.startswith("knn"):
        model = KNeighborsClassifier(n_neighbors=25, p=1 if key.endswith("l1") else 2, n_jobs=2)
    elif key == "svm":
        model = LinearSVC(C=1.0, dual=False, max_iter=5000, random_state=seed)
    elif key == "random_forest":
        model = RandomForestClassifier(n_estimators=250, max_depth=10, min_samples_leaf=10, n_jobs=2, random_state=seed)
    else:
        raise ValueError(key)
    return Pipeline([("preprocessor", processor(without_sensitive=without_sensitive)), ("classifier", model)])


def baseline_fit(key, seed, features, target, split, without_sensitive=False):
    pipeline = baseline_pipeline(key, seed, without_sensitive)
    with warnings.catch_warnings(record=True) as observed, threadpool_limits(limits=2):
        warnings.simplefilter("always")
        pipeline.fit(features.iloc[split["fit"]], target.iloc[split["fit"]])
    cal_x, val_x = (features.iloc[split[name]] for name in ["calibration", "validation"])
    if key == "svm":
        # A SVM usa a partição de calibração para produzir probabilidades comparáveis.
        calibrator = fit_calibrator("sigmoid", expit(pipeline.decision_function(cal_x)), target.iloc[split["calibration"]])
        cal_p = apply_calibrator("sigmoid", calibrator, expit(pipeline.decision_function(cal_x)))
        val_p = apply_calibrator("sigmoid", calibrator, expit(pipeline.decision_function(val_x)))
    else:
        with threadpool_limits(limits=2):
            cal_p, val_p = (pipeline.predict_proba(values)[:, 1] for values in [cal_x, val_x])
    return pipeline, cal_p, val_p, [str(item.message) for item in observed]


def summarize(frame, group_columns):
    rows = []
    for keys, group in frame.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["runs"] = len(group)
        for metric in SCORE_KEYS:
            values = group[metric].to_numpy(dtype=float)
            mean, sd = float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0
            radius = float(student_t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values))) if len(values) > 1 else 0.0
            row.update({f"{metric}_mean": mean, f"{metric}_sd": sd,
                        f"{metric}_ci_low": max(0.0, mean - radius), f"{metric}_ci_high": mean + radius})
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_intervals(features, target, probability, threshold, repetitions=1000, seed=2026):
    groups = features.groupby(FEATURES, sort=False, dropna=False).ngroup().to_numpy()
    count = int(groups.max()) + 1
    rng = np.random.default_rng(seed)
    y = np.asarray(target)
    p = np.asarray(probability)
    order = np.argsort(-p, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(y) - 1]
    values = {metric: [] for metric in ["average_precision", "roc_auc", "f1", "recall", "brier_score"]}
    for _ in range(repetitions):
        group_weights = np.bincount(rng.integers(0, count, size=count), minlength=count)
        weights = group_weights[groups]
        positive, negative = weights * y, weights * (1 - y)
        tp, fp = np.cumsum(positive[order])[ends], np.cumsum(negative[order])[ends]
        if tp[-1] == 0 or fp[-1] == 0:
            continue
        recall = tp / tp[-1]
        precision = tp / np.maximum(tp + fp, 1)
        values["average_precision"].append(float(np.dot(np.diff(np.r_[0, recall]), precision)))
        values["roc_auc"].append(float(np.trapezoid(np.r_[0, recall], np.r_[0, fp / fp[-1]])))
        predicted = p >= threshold
        true_positive = positive[predicted].sum()
        false_positive = negative[predicted].sum()
        false_negative = positive[~predicted].sum()
        values["recall"].append(float(true_positive / positive.sum()))
        values["f1"].append(float(2 * true_positive / max(1, 2 * true_positive + false_positive + false_negative)))
        values["brier_score"].append(float(np.average((p - y) ** 2, weights=weights)))
    return {metric: {"low": float(np.quantile(samples, 0.025)), "high": float(np.quantile(samples, 0.975))}
            for metric, samples in values.items()}


def expected_calibration_error(target, probability, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    assignment = np.minimum(np.searchsorted(edges, probability, side="right") - 1, bins - 1)
    return float(sum(np.mean(mask) * abs(np.asarray(target)[mask].mean() - np.asarray(probability)[mask].mean())
                     for index in range(bins) if (mask := assignment == index).any()))


def run(data_path, output, seeds, epochs, resume=False, integrate=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "runs"
    cache.mkdir(exist_ok=True)
    features, target = load_dataset(data_path)
    _, _, holdout = split_dataset(features, target, 42)
    configs = configurations()
    protocol = {
        "version": 1, "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "seeds": seeds, "max_epochs": epochs, "patience": 10, "reference_seed": 42,
        "holdout": {"rows": len(holdout), "sha256_indices": hashlib.sha256(holdout.tobytes()).hexdigest(),
                    "status": "Conjunto já consultado no primeiro desenvolvimento; resultados complementares, sem alegação de teste inédito."},
        "development_fractions": {"fit": 0.65, "stop": 0.10, "calibration": 0.10, "validation": 0.15},
        "neural_configurations": [asdict(config) for config in configs],
        "baselines": BASELINES, "sensitive_ablation": {"models": list(MODEL_NAMES), "excluded": SENSITIVE},
        "model_selection": "Maior AP média de validação em cinco sementes; uso na aplicação limitado aos três modelos já integrados.",
        "calibration_selection": "Menor Brier médio na validação; ajuste do calibrador somente na partição de calibração.",
        "threshold_selection": "F1 máximo na validação para cada modelo e semente; não usa o conjunto de referência.",
        "uncertainty": "Média, desvio e intervalo t descritivo entre sementes; partições se sobrepõem. Bootstrap por perfil na referência, 1000 reamostragens.",
        "training": "Adam, lotes 512, gradiente limitado a norma 5, parada antecipada por BCE não ponderada em stop; biases iniciais zero.",
    }
    protocol_path = output / "protocol.json"
    if resume and protocol_path.exists():
        if json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError("O protocolo mudou; use outra pasta de saída.")
    else:
        if list(cache.glob("*.npz")):
            raise ValueError("Resultados existentes; use --resume ou outra pasta.")
        write_json(protocol_path, protocol)
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    torch.set_num_threads(2)
    records, calibration_records, curve_records, warnings_log = [], [], [], []
    names = {**BASELINES, **{config.key: config.label for config in configs},
             **{f"{key}_no_sensitive": f"{name}, sem sexo/idade/estado civil" for key, name in MODEL_NAMES.items()}}
    jobs = [(key, None) for key in BASELINES] + [(config.key, config) for config in configs] + [(key + "_no_sensitive", None) for key in MODEL_NAMES]
    for seed in seeds:
        split = development_split(features, target, holdout, seed)
        np.savez_compressed(output / f"split_seed_{seed}.npz", **split, reference=holdout)
        for key, config in jobs:
            start = time.monotonic()
            saved = cache / f"{key}_{seed}.npz"
            meta_path = cache / f"{key}_{seed}.json"
            history = []
            if resume and saved.exists() and meta_path.exists():
                arrays = np.load(saved)
                cal_p, val_p = arrays["calibration"], arrays["validation"]
                meta = json.loads(meta_path.read_text())
                history = meta.get("curve", [])
            else:
                print(f"[{seed}] {key}", flush=True)
                if config is not None:
                    prep = processor(config.scale, config.one_hot)
                    xfit = prep.fit_transform(features.iloc[split["fit"]])
                    xstop, xcal, xval = (prep.transform(features.iloc[split[name]]) for name in ["stop", "calibration", "validation"])
                    network, history, meta = train_network(xfit, target.iloc[split["fit"]], xstop, target.iloc[split["stop"]], config, seed, epochs)
                    cal_p, val_p = probabilities(network, xcal), probabilities(network, xval)
                else:
                    no_sensitive = key.endswith("_no_sensitive")
                    original = key.removesuffix("_no_sensitive")
                    _, cal_p, val_p, observed = baseline_fit(original, seed, features, target, split, no_sensitive)
                    meta = {"warnings": observed}
                meta["seconds"] = time.monotonic() - start
                meta["curve"] = history
                meta["protocol_sha256"] = protocol_hash
                np.savez_compressed(saved, calibration=cal_p, validation=val_p)
                write_json(meta_path, meta)
            if meta["protocol_sha256"] != protocol_hash:
                raise ValueError("Cache incompatível com o protocolo.")
            warnings_log.extend({"seed": seed, "model": key, "warning": msg} for msg in meta.get("warnings", []))
            threshold = choose_threshold(target.iloc[split["validation"]], val_p)
            score = metrics(target.iloc[split["validation"]], val_p, threshold)
            score.pop("confusion_matrix")
            records.append({"seed": seed, "model": key, "label": names[key],
                            "question": config.question if config else ("sensitive" if key.endswith("_no_sensitive") else "P1"),
                            "seconds": meta["seconds"], "best_epoch": meta.get("best_epoch"),
                            "epochs_run": meta.get("epochs_run"), **score})
            curve_records.extend({"seed": seed, "model": key, **entry} for entry in history)
            if key in MODEL_NAMES:
                for method in ["none", "sigmoid", "isotonic"]:
                    calibrator = fit_calibrator(method, cal_p, target.iloc[split["calibration"]])
                    calibrated = apply_calibrator(method, calibrator, val_p)
                    threshold = choose_threshold(target.iloc[split["validation"]], calibrated)
                    score = metrics(target.iloc[split["validation"]], calibrated, threshold)
                    score.pop("confusion_matrix")
                    calibration_records.append({"seed": seed, "model": key, "method": method,
                                                "ece": expected_calibration_error(target.iloc[split["validation"]], calibrated), **score})
            pd.DataFrame(records).to_csv(output / "experiments.csv", index=False)
            print(f"  AP={records[-1]['average_precision']:.4f}; {meta['seconds']:.1f}s", flush=True)
    experiments = pd.DataFrame(records)
    summary = summarize(experiments, ["model", "label", "question"])
    summary.to_csv(output / "summary.csv", index=False)
    calibration_frame = pd.DataFrame(calibration_records)
    calibration_frame.to_csv(output / "calibration.csv", index=False)
    calibration_summary = summarize(calibration_frame, ["model", "method"])
    calibration_summary.to_csv(output / "calibration_summary.csv", index=False)
    pd.DataFrame(curve_records).to_csv(output / "learning_curves.csv", index=False)
    convergence = []
    curves_frame = pd.DataFrame(curve_records)
    for record in records:
        if record.get('best_epoch') is None:
            continue
        point = curves_frame[(curves_frame.model == record['model']) & (curves_frame.seed == record['seed']) & (curves_frame.epoch == record['best_epoch'])].iloc[0]
        convergence.append({'model':record['model'], 'seed':record['seed'], 'best_epoch':record['best_epoch'],
                            'epochs_run':record['epochs_run'], 'train_loss_best':point.train_loss,
                            'stop_loss_best':point.stop_loss, 'gap':point.stop_loss-point.train_loss})
    pd.DataFrame(convergence).to_csv(output / 'convergence.csv', index=False)
    write_json(output / "training_warnings.json", warnings_log)
    best_nn = summary[summary["model"].str.startswith("nn_")].sort_values("average_precision_mean", ascending=False).iloc[0]["model"]
    eligible = summary[summary["model"].isin(MODEL_NAMES)]
    selected = eligible.sort_values("average_precision_mean", ascending=False).iloc[0]["model"]
    calibration_methods = {key: calibration_summary[calibration_summary["model"] == key].sort_values("brier_score_mean").iloc[0]["method"] for key in MODEL_NAMES}
    decisions = {"protocol_sha256": protocol_hash, "selected_application_model": selected,
                 "best_neural_configuration": best_nn, "calibration_methods": calibration_methods,
                 "application_mean_validation_ap": {row['model']: float(row['average_precision_mean']) for _, row in eligible.iterrows()},
                 "mean_validation_brier": {key: {row['method']:float(row['brier_score_mean']) for _,row in calibration_summary[calibration_summary.model == key].iterrows()} for key in MODEL_NAMES},
                 "selection_data": "Apenas partições de validação no desenvolvimento; nenhuma métrica de referência nesta decisão."}
    write_json(output / "selection_locked.json", decisions)
    # A avaliação de referência só começa depois de a seleção ter sido salva.
    final_evaluation(features, target, holdout, output, decisions, configs, epochs, integrate)
    paired = []
    indexed = experiments.set_index(["model", "seed"])
    for key in names:
        if key == "nn_base":
            continue
        for metric in ["average_precision", "roc_auc", "f1", "recall"]:
            differences = np.array([indexed.loc[(key, seed), metric] - indexed.loc[("nn_base", seed), metric] for seed in seeds])
            mean, sd = float(differences.mean()), float(differences.std(ddof=1))
            radius = float(student_t.ppf(0.975, len(seeds)-1) * sd / np.sqrt(len(seeds)))
            paired.append({"model": key, "baseline": "nn_base", "metric": metric, "mean_delta": mean,
                           "sd_delta": sd, "ci_low": mean-radius, "ci_high": mean+radius})
    pd.DataFrame(paired).to_csv(output / "paired_differences.csv", index=False)
    plot_research(output, data_path)
    print("Pesquisa concluída; escolhas:", json.dumps(decisions, ensure_ascii=False), flush=True)


def final_evaluation(features, target, holdout, output, decisions, configs, epochs, integrate=False):
    seed = 42
    split = development_split(features, target, holdout, seed)
    xref, yref = features.iloc[holdout], target.iloc[holdout]
    final_models, reference, validation, calibration_test, bootstrap, fairness, cost_rows = {}, {}, {}, [], {}, {}, []
    for key in MODEL_NAMES:
        print(f"Avaliação complementar de referência: {key}", flush=True)
        pipeline, cal_p, val_p, observed = baseline_fit(key, seed, features, target, split)
        cached = np.load(output / "runs" / f"{key}_{seed}.npz")
        np.testing.assert_allclose(val_p, cached["validation"], rtol=0, atol=1e-10)
        for method in ["none", "sigmoid", "isotonic"]:
            model = CalibratedCreditModel(pipeline, method).fit(features.iloc[split["calibration"]], target.iloc[split["calibration"]])
            pval = model.predict_proba(features.iloc[split["validation"]])[:, 1]
            pref = model.predict_proba(xref)[:, 1]
            threshold = choose_threshold(target.iloc[split["validation"]], pval)
            score = metrics(yref, pref, threshold)
            calibration_test.append({"model": key, "method": method, "selected": method == decisions["calibration_methods"][key],
                                     "ece": expected_calibration_error(yref, pref), **{k:v for k,v in score.items() if k != "confusion_matrix"}})
            if method == decisions["calibration_methods"][key]:
                final_models[key] = model
                validation[key] = metrics(target.iloc[split["validation"]], pval, threshold)
                reference[key] = score
                bootstrap[key] = bootstrap_intervals(xref, yref, pref, threshold)
                fairness[key] = group_metrics(xref, yref, pref, threshold)
                np.savez_compressed(output / f"reference_predictions_{key}.npz", indices=holdout, probability=pref,
                                    validation_indices=split["validation"], validation_probability=pval)
                for ratio in [1, 2, 5, 10]:
                    candidates = np.unique(np.r_[0, pval, 1])
                    losses = [ratio * np.sum((target.iloc[split["validation"]].to_numpy() == 1) & (pval < t)) + np.sum((target.iloc[split["validation"]].to_numpy() == 0) & (pval >= t)) for t in candidates]
                    optimal = float(candidates[int(np.argmin(losses))])
                    predicted = pref >= optimal
                    fn = int(np.sum((np.asarray(yref) == 1) & ~predicted))
                    fp = int(np.sum((np.asarray(yref) == 0) & predicted))
                    cost_rows.append({"model": key, "false_negative_cost": ratio, "false_positive_cost": 1,
                                      "threshold_validation": optimal, "reference_fn": fn, "reference_fp": fp,
                                      "cost_per_1000": (ratio * fn + fp) / len(yref) * 1000})
    pd.DataFrame(calibration_test).to_csv(output / "calibration_reference.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(output / "cost_scenarios.csv", index=False)
    for key in ["knn_l1", "knn_l2", "svm", "random_forest"]:
        print(f"Referência complementar: {key}", flush=True)
        pipeline, cal_p, val_p, _ = baseline_fit(key, seed, features, target, split)
        if key == "svm":
            calibrator = fit_calibrator("sigmoid", expit(pipeline.decision_function(features.iloc[split["calibration"]])), target.iloc[split["calibration"]])
            pref = apply_calibrator("sigmoid", calibrator, expit(pipeline.decision_function(xref)))
        else:
            pref = pipeline.predict_proba(xref)[:, 1]
        threshold = choose_threshold(target.iloc[split["validation"]], val_p)
        reference[key] = metrics(yref, pref, threshold)
        bootstrap[key] = bootstrap_intervals(xref, yref, pref, threshold)
        np.savez_compressed(output / f"reference_predictions_{key}.npz", indices=holdout, probability=pref,
                            validation_indices=split["validation"], validation_probability=val_p)
    nn_config = next(config for config in configs if config.key == decisions["best_neural_configuration"])
    prep = processor(nn_config.scale, nn_config.one_hot)
    xfit = prep.fit_transform(features.iloc[split["fit"]])
    network, curve, info = train_network(xfit, target.iloc[split["fit"]], prep.transform(features.iloc[split["stop"]]), target.iloc[split["stop"]], nn_config, seed, epochs)
    val_p = probabilities(network, prep.transform(features.iloc[split["validation"]]))
    np.testing.assert_allclose(val_p, np.load(output / "runs" / f"{nn_config.key}_42.npz")["validation"], rtol=0, atol=1e-10)
    pref = probabilities(network, prep.transform(xref))
    threshold = choose_threshold(target.iloc[split["validation"]], val_p)
    reference["best_torch_mlp"] = metrics(yref, pref, threshold)
    bootstrap["best_torch_mlp"] = bootstrap_intervals(xref, yref, pref, threshold)
    torch.save(network.state_dict(), output / "best_neural_weights.pt")
    joblib.dump(prep, output / "best_neural_preprocessor.joblib", compress=3)
    write_json(output / "best_neural_config.json", info)
    np.savez_compressed(output / "reference_predictions_best_torch_mlp.npz", indices=holdout, probability=pref,
                        validation_indices=split["validation"], validation_probability=val_p)
    final = {"reference_status": "Conjunto já consultado anteriormente; avaliação complementar, sem validação externa.",
             "reference_metrics": reference, "bootstrap_95": bootstrap, "group_metrics": fairness,
             "decisions": decisions, "calibration_rows": len(split["calibration"]),
             "splits_seed_42": {name: {"rows": len(idx), "default_rate": float(target.iloc[idx].mean())} for name,idx in split.items()},
             "versions": {package: importlib.metadata.version(package) for package in ["torch", "scikit-learn", "numpy", "pandas", "xgboost", "scipy"]},
             "python": platform.python_version()}
    write_json(output / "final_evaluation.json", final)
    # O pacote de aplicação mantém as mesmas três opções e passa a usar a calibração escolhida.
    old = joblib.load(BASE_DIR / "results/models.joblib")
    report = dict(old["report"])
    report.update({"selected_model": decisions["selected_application_model"], "selection_metric": "research.mean_validation.average_precision",
                   "validation": validation, "test": {key: reference[key] for key in MODEL_NAMES},
                   "calibration_methods": decisions["calibration_methods"], "research": final,
                   "group_metrics_test": fairness,
                   "split": {**final["splits_seed_42"], "test": {"rows":len(holdout), "default_rate":float(yref.mean())}},
                   "split_method": "Referência original fixa (semente 42); desenvolvimento dividido por perfil em fit/stop/calibration/validation, frações 65/10/10/15%.",
                   "notes": ["Referência já consultada na etapa anterior; não é teste inédito.",
                             "Seleção de modelo por AP média em cinco sementes; calibrador por Brier médio na validação.",
                             "Pipeline base ajustado em fit; calibrador ajustado somente em calibration; limiar escolhido em validation.",
                             "Resultados sensíveis à divisão e à base de Taiwan de 2005; sem validação externa."]})
    reference_profile = features.iloc[split["fit"]].median().to_dict()
    reference_profile["AGE"] = float(features.iloc[split["fit"]]["AGE"].mode().iloc[0])
    for col in CATEGORICAL:
        reference_profile[col] = float(features.iloc[split["fit"]][col].mode().iloc[0])
    bundle = {"schema_version": 1, "features": FEATURES, "selected_model": decisions["selected_application_model"],
              "models": final_models, "reference": reference_profile, "report": report}
    atomic_bundle(bundle, output / "application_models.joblib")
    write_json(output / "application_metrics.json", report)
    np.savez_compressed(output / "application_split_indices.npz", train=split["fit"], validation=split["validation"],
                        early_stop=split["stop"], calibration=split["calibration"], test=holdout)
    if not integrate:
        return
    initial = BASE_DIR / "results/initial_baseline"
    if not initial.exists():
        initial.mkdir()
        for name in ["models.joblib", "metrics.json", "resultados.csv", "split_indices.npz", "metricas_por_grupo.csv", "curvas_avaliacao.png", "matrizes_confusao.png", "curvas_mlp.png"]:
            source = BASE_DIR / "results" / name
            if source.exists():
                shutil.copy2(source, initial / name)
    atomic_bundle(bundle, BASE_DIR / "results/models.joblib")
    write_json(BASE_DIR / "results/metrics.json", report)
    np.savez_compressed(BASE_DIR / "results/split_indices.npz", train=split["fit"], validation=split["validation"],
                        early_stop=split["stop"], calibration=split["calibration"], test=holdout)
    pd.DataFrame([{"model": MODEL_NAMES[key], "selected": key == bundle["selected_model"],
                   **{k:v for k,v in reference[key].items() if k != "confusion_matrix"}} for key in MODEL_NAMES]).to_csv(BASE_DIR / "results/resultados.csv", index=False)
    pd.DataFrame([{"model": key, **row} for key, rows in fairness.items() for row in rows]).to_csv(BASE_DIR / "results/metricas_por_grupo.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    for key, model in final_models.items():
        predicted = model.predict_proba(xref)[:,1]
        RocCurveDisplay.from_predictions(yref,predicted,name=MODEL_NAMES[key],ax=axes[0])
        PrecisionRecallDisplay.from_predictions(yref,predicted,name=MODEL_NAMES[key],ax=axes[1])
    fig.suptitle('Avaliação complementar: referência já consultada')
    fig.tight_layout()
    fig.savefig(BASE_DIR / 'results/curvas_avaliacao.png',dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key in zip(axes,MODEL_NAMES):
        ConfusionMatrixDisplay(np.array(reference[key]['confusion_matrix']),display_labels=['Adimplente','Inadimplente']).plot(ax=ax,colorbar=False)
        ax.set_title(MODEL_NAMES[key])
    fig.tight_layout()
    fig.savefig(BASE_DIR / 'results/matrizes_confusao.png',dpi=150)
    plt.close(fig)
    mlp=final_models['mlp'].named_steps['classifier']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(mlp.loss_curve_)
    axes[0].set(title='MLP da aplicação: perda em fit',xlabel='Época')
    axes[1].plot(mlp.validation_scores_)
    axes[1].set(title='Validação interna da MLP',xlabel='Época')
    fig.tight_layout()
    fig.savefig(BASE_DIR / 'results/curvas_mlp.png',dpi=150)
    plt.close(fig)


def plot_research(output, data_path=DEFAULT_DATA):
    summary = pd.read_csv(output / "summary.csv")
    for name, questions in [("models", ["P1", "base"]), ("architecture", ["P2", "base"]),
                             ("regularization", ["P3", "base"]), ("preprocessing", ["P4", "base"])]:
        data = summary[summary["question"].isin(questions)].sort_values("average_precision_mean")
        fig, ax = plt.subplots(figsize=(10, max(4, len(data) * 0.48)))
        ax.barh(data["label"], data["average_precision_mean"], xerr=data["average_precision_sd"], color="#286373", capsize=3)
        ax.set(xlabel="AP de validação: média e desvio entre cinco sementes", xlim=(0, 0.75))
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=160)
        plt.close(fig)
    cal = pd.read_csv(output / "calibration_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key in zip(axes, MODEL_NAMES):
        data = cal[cal["model"] == key]
        ax.bar(data["method"], data["brier_score_mean"], yerr=data["brier_score_sd"], color="#286373", capsize=3)
        ax.set(title=MODEL_NAMES[key], ylabel="Brier na validação (menor é melhor)")
    fig.tight_layout()
    fig.savefig(output / "calibration_comparison.png", dpi=160)
    plt.close(fig)
    features, target = load_dataset(data_path)
    final = json.loads((output / "final_evaluation.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key in zip(axes, MODEL_NAMES):
        arrays = np.load(output / f"reference_predictions_{key}.npz")
        fraction, predicted = calibration_curve(target.iloc[arrays["indices"]], arrays["probability"], n_bins=10, strategy="quantile")
        ax.plot([0,1], [0,1], "--", color="gray")
        ax.plot(predicted, fraction, "o-", color="#286373")
        ax.set(title=MODEL_NAMES[key], xlabel="Probabilidade média", ylabel="Fração de inadimplentes", xlim=(0,1), ylim=(0,1))
    fig.tight_layout()
    fig.savefig(output / "reliability.png", dpi=160)
    plt.close(fig)
    curves = pd.read_csv(output / "learning_curves.csv")
    selected = ["nn_base", "nn_regularized", "nn_weighted", "nn_unscaled"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, key in zip(axes.ravel(), selected):
        part = curves[(curves["model"] == key) & (curves["seed"] == 42)]
        ax.plot(part["epoch"], part["train_loss"], label="Ajuste")
        ax.plot(part["epoch"], part["stop_loss"], label="Parada antecipada")
        ax.set(title=key, xlabel="Época", ylabel="BCE não ponderada")
        if key == "nn_unscaled":
            ax.set_yscale("symlog", linthresh=1)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "results/research")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--integrate", action="store_true", help="Atualiza também os três modelos usados pelo aplicativo")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2 or 42 not in args.seeds or args.epochs < 1:
        parser.error("Informe pelo menos duas sementes distintas, incluindo 42, e épocas positivas.")
    run(args.data, args.output, args.seeds, args.epochs, args.resume, args.integrate)


if __name__ == "__main__":
    main()
