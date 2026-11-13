"""Esquema compartilhado entre treinamento e inferência."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    *[f"BILL_AMT{i}" for i in range(1, 7)],
    *[f"PAY_AMT{i}" for i in range(1, 7)],
]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
TARGET = "default.payment.next.month"
PAY_COLUMNS = FEATURES[5:11]
DEFAULT_DATA = BASE_DIR / "data" / "default_credit.csv.gz"


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Atributos ausentes: {', '.join(sorted(missing))}")
    if frame.columns.duplicated().any():
        raise ValueError("Há nomes de colunas duplicados.")
    data = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="raise").copy()
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Os atributos devem ser números finitos, sem valores ausentes.")
    data["EDUCATION"] = data["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    data["MARRIAGE"] = data["MARRIAGE"].replace({0: 3})
    for col, allowed in [("SEX", [1, 2]), ("EDUCATION", [1, 2, 3, 4]), ("MARRIAGE", [1, 2, 3])]:
        if not data[col].isin(allowed).all():
            raise ValueError(f"Categoria inválida em {col}.")
    integer_cols = [*CATEGORICAL, "AGE", *PAY_COLUMNS]
    if (data[integer_cols] % 1 != 0).any().any():
        raise ValueError("Idade, categorias e situação de pagamento devem ser inteiros.")
    if (data["LIMIT_BAL"] <= 0).any() or not data["AGE"].between(18, 100).all():
        raise ValueError("Limite deve ser positivo e idade deve estar entre 18 e 100.")
    if not data[PAY_COLUMNS].isin(range(-2, 9)).all().all():
        raise ValueError("Situação de pagamento deve estar entre -2 e 8.")
    if (data[[f"PAY_AMT{i}" for i in range(1, 7)]] < 0).any().any():
        raise ValueError("Pagamentos não podem ser negativos.")
    return data


def load_dataset(path: Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {path}. Informe --data CSV/XLS/XLSX.")
    if path.suffix.lower() in {".xls", ".xlsx"}:
        frame = pd.read_excel(path)
        if "LIMIT_BAL" not in frame.columns and "X1" not in frame.columns:
            frame = pd.read_excel(path, header=1)
    else:
        frame = pd.read_csv(path)
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.columns.duplicated().any():
        raise ValueError("Há nomes de colunas duplicados.")
    aliases = {f"X{i}": col for i, col in enumerate(FEATURES, 1)}
    aliases.update({"Y": TARGET, "default payment next month": TARGET,
                    "default_payment_next_month": TARGET, "PAY_1": "PAY_0"})
    frame = frame.rename(columns=aliases)
    if TARGET not in frame:
        raise ValueError(f"Coluna alvo ausente: {TARGET}.")
    target = pd.to_numeric(frame[TARGET], errors="raise")
    if not target.isin([0, 1]).all() or target.nunique() != 2:
        raise ValueError("O alvo deve conter as duas classes: 0 e 1.")
    if target.value_counts().min() < 10:
        raise ValueError("São necessários pelo menos dez registros por classe.")
    return clean_features(frame).reset_index(drop=True), target.astype(int).reset_index(drop=True)


def split_dataset(features, target, seed=42):
    # Perfis idênticos ficam na mesma partição, inclusive se seus alvos divergirem.
    groups = features.groupby(FEATURES, sort=False, dropna=False).ngroup()
    grouped = pd.DataFrame({"group": groups, "y": target}).groupby("group")["y"].max()
    train_groups, remaining = train_test_split(
        grouped.index, test_size=0.30, stratify=grouped, random_state=seed,
    )
    val_groups, test_groups = train_test_split(
        remaining, test_size=0.50, stratify=grouped.loc[remaining], random_state=seed,
    )
    return tuple(np.flatnonzero(groups.isin(g)) for g in (train_groups, val_groups, test_groups))
