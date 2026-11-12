# ============================================================
# P4 + MLP PROFUNDO  -  VERSÃO OFFLINE (sem downloads)
#
# Só precisa de: numpy, pandas e matplotlib.
# NÃO usa PyTorch, scikit-learn nem internet: o MLP (com batch norm,
# dropout, L2 e otimizador Adam) é implementado em numpy.
#
# P4: "Pré-processamento (normalização, codificação das categorias) e
#      inicialização dos pesos mudam a convergência e o resultado?"
#
#   Experimento A - Pré-processamento (fatorial 2x2):
#       normalização (sem / com)  x  categorias (ordinal / one-hot)
#   Experimento B - Inicialização dos pesos:
#       padrão (estilo PyTorch), Xavier, He, Normal(0, 0.01), zeros
#
# DATASET:
#   - Se houver um arquivo do dataset na mesma pasta do script
#     (.csv, .xls ou .xlsx com "credit" ou "default" no nome), ele é usado.
#   - Também dá para passar:  --csv caminho/do/arquivo
#   - Se não achar nada, gera DADOS SINTÉTICOS no mesmo formato só para
#     o código rodar. Resultados com dados sintéticos NÃO valem para o
#     relatório: use o dataset real.
#
# Uso:
#   python p4_mlp_offline.py
#   python p4_mlp_offline.py --csv default_of_credit_card_clients.xls
#   python p4_mlp_offline.py --epochs 5 --seeds 42      (teste rápido)
# ============================================================

import argparse
import copy
import glob
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # só salva as figuras, não abre janelas
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUT_DIR = "results_p4"


# ============================================================
# 1. DADOS
# ============================================================

def gerar_dataset_sintetico(n=30000, seed=0):
    """Dados falsos no formato do dataset da UCI (23 variáveis, ~22% inadimplentes)."""
    rng = np.random.default_rng(seed)

    limite = np.clip(rng.lognormal(11.6, 0.9, n) // 10000 * 10000, 10000, 1_000_000)
    sexo = rng.choice([1, 2], n, p=[0.4, 0.6])
    educ = rng.choice([0, 1, 2, 3, 4, 5, 6], n, p=[0.001, 0.35, 0.47, 0.16, 0.01, 0.007, 0.002])
    estado = rng.choice([0, 1, 2, 3], n, p=[0.002, 0.455, 0.53, 0.013])
    idade = np.clip(rng.normal(35, 9, n), 21, 75).round()

    risco = rng.normal(0, 1, n)
    df = {"LIMIT_BAL": limite, "SEX": sexo, "EDUCATION": educ, "MARRIAGE": estado, "AGE": idade}

    pays = []
    for k in range(6):
        p = np.clip(np.round(0.9 * risco + rng.normal(0, 0.8, n) - 0.25), -2, 8)
        pays.append(p)
        df["PAY_" + str(k)] = p
    for k in range(6):
        df["BILL_AMT" + str(k + 1)] = limite * rng.beta(2, 3, n) * (1 + 0.2 * np.maximum(pays[k], 0))
    for k in range(6):
        df["PAY_AMT" + str(k + 1)] = df["BILL_AMT" + str(k + 1)] * rng.beta(1, 5, n)

    score = 0.9 * np.maximum(pays[0], 0) + 0.3 * np.mean(pays[1:], axis=0) \
        - 0.25 * np.log(limite / 1e5) + rng.normal(0, 1.3, n)
    lo, hi = -20, 20
    for _ in range(50):  # ajusta o intercepto para ~22,1% de inadimplentes
        b = (lo + hi) / 2
        if (1 / (1 + np.exp(-(score + b)))).mean() > 0.221:
            hi = b
        else:
            lo = b
    y = (rng.random(n) < 1 / (1 + np.exp(-(score + b)))).astype(int)
    return pd.DataFrame(df), pd.Series(y)


def localizar_dataset(caminho):
    if caminho:
        return caminho
    pasta = os.path.dirname(os.path.abspath(__file__))
    for arq in sorted(glob.glob(os.path.join(pasta, "*")) + glob.glob("*")):
        nome = os.path.basename(arq).lower()
        if nome.endswith((".csv", ".xls", ".xlsx")) and ("credit" in nome or "default" in nome):
            return arq
    return None


def carregar_dataset(caminho, forcar_sintetico=False):
    arq = None if forcar_sintetico else localizar_dataset(caminho)

    if arq is None:
        print("\n" + "!" * 70)
        print("DATASET NÃO ENCONTRADO -> usando dados SINTÉTICOS (só para testar o código).")
        print("Coloque o arquivo da UCI na pasta do script ou use --csv para usar o real.")
        print("!" * 70)
        X, y = gerar_dataset_sintetico()
        return X, y, True

    print(f"\nLendo dataset: {arq}")
    if arq.lower().endswith(".csv"):
        df = pd.read_csv(arq)
    else:
        try:
            df = pd.read_excel(arq)
            if any(str(c).startswith("Unnamed") for c in df.columns):
                df = pd.read_excel(arq, header=1)  # o .xls da UCI tem uma linha de títulos extra
        except ImportError:
            raise SystemExit(
                "Para ler .xls é preciso o pacote xlrd (pip install xlrd). "
                "Alternativa sem instalar nada: abra no Excel e salve como .csv."
            )

    alvo = df.columns[-1]  # a variável-alvo é a última coluna
    y = df[alvo].astype(int)
    X = df.drop(columns=[alvo])
    if "ID" in X.columns:
        X = X.drop(columns=["ID"])
    return X.reset_index(drop=True), y.reset_index(drop=True), False


def identificar_colunas(X):
    """Categóricas = sexo, escolaridade, estado civil (SEX/EDUCATION/MARRIAGE ou X2/X3/X4)."""
    for nomes in (("SEX", "EDUCATION", "MARRIAGE"), ("X2", "X3", "X4")):
        if all(n in X.columns for n in nomes):
            cat = list(nomes)
            break
    else:
        cat = list(X.columns[1:4])
    num = [c for c in X.columns if c not in cat]
    return cat, num


def limpar_categorias(X, cat):
    """EDUCATION 0,5,6 -> 4 (outros); MARRIAGE 0 -> 3 (outros). Regra fixa, sem vazamento."""
    X = X.copy()
    X[cat[1]] = X[cat[1]].replace({0: 4, 5: 4, 6: 4})
    X[cat[2]] = X[cat[2]].replace({0: 3})
    return X


def dividir(X, y, seed):
    """Divisão estratificada 70% treino / 15% validação / 15% teste."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for classe in (0, 1):
        idx = rng.permutation(np.where(y.to_numpy() == classe)[0])
        n = len(idx)
        a, b = int(round(0.70 * n)), int(round(0.85 * n))
        tr.append(idx[:a]); va.append(idx[a:b]); te.append(idx[b:])
    tr, va, te = (np.sort(np.concatenate(p)) for p in (tr, va, te))
    return (X.iloc[tr], X.iloc[va], X.iloc[te], y.iloc[tr], y.iloc[va], y.iloc[te])


# ============================================================
# 2. PRÉ-PROCESSAMENTO (fatorial normalização x codificação)
# ============================================================

CONFIGS_PREPROC = {
    # nome: (normalizar?, one-hot?)
    "sem norm + ordinal": (False, False),
    "com norm + ordinal": (True, False),
    "sem norm + one-hot": (False, True),
    "com norm + one-hot": (True, True),
}


def preprocessar(Xtr, Xva, Xte, cat, num, normalizar, one_hot):
    """
    Tudo é ajustado SOMENTE no treino (média/desvio e lista de categorias).
      ordinal : categorias continuam como números (1, 2, 3...)
      one-hot : cada categoria vira colunas 0/1 (categoria nova -> tudo zero)
      norm    : (x - média) / desvio  nas colunas numéricas
                (no modo ordinal, também nas categóricas)
    """
    colunas_num = num if one_hot else num + cat
    A = [d[colunas_num].to_numpy(dtype=np.float64) for d in (Xtr, Xva, Xte)]

    if normalizar:
        mu, sd = A[0].mean(0), A[0].std(0)
        sd[sd == 0] = 1.0
        A = [(m - mu) / sd for m in A]

    if one_hot:
        extras = [[], [], []]
        for c in cat:
            categorias = np.unique(Xtr[c].to_numpy())
            for i, d in enumerate((Xtr, Xva, Xte)):
                extras[i].append((d[c].to_numpy()[:, None] == categorias[None, :]).astype(np.float64))
        A = [np.hstack([m] + e) for m, e in zip(A, extras)]
    return tuple(A)


# ============================================================
# 3. MÉTRICAS (implementadas em numpy)
# ============================================================

def auc_roc(y, p):
    y = np.asarray(y)
    r = pd.Series(p).rank().to_numpy()  # postos (empates -> média)
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def recall_f1(y, pred):
    y = np.asarray(y)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return rec, f1


def limiar_f1_validacao(y, p):
    """Limiar que maximiza o F1 na VALIDAÇÃO (nunca no teste)."""
    y = np.asarray(y)
    ordem = np.argsort(-p)
    ps, ys = p[ordem], y[ordem]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / (tp + fp)
    rec = tp / max(ys.sum(), 1)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(ps[int(np.argmax(f1))])


def metricas_convergencia(h_tr, h_va):
    if np.all(np.isnan(h_va)):
        return dict(melhor_epoca=np.nan, epocas_ate_2pct=np.nan, min_loss_val=np.nan, gap_final=np.nan)
    melhor = int(np.nanargmin(h_va))
    minimo = float(np.nanmin(h_va))
    ate_2pct = int(np.where(h_va <= minimo * 1.02)[0][0]) + 1
    ultimo = int(np.where(~np.isnan(h_va))[0][-1])
    return dict(
        melhor_epoca=melhor + 1,                       # época do menor val. loss
        epocas_ate_2pct=ate_2pct,                      # 1ª época com val. loss <= 1,02 x mínimo
        min_loss_val=minimo,
        gap_final=float(h_va[ultimo] - h_tr[ultimo]),  # validação - treino
    )


# ============================================================
# 4. MLP PROFUNDO EM NUMPY
#    [Linear -> BatchNorm -> ReLU -> Dropout] x 3  ->  Linear(1) + sigmoid
#    L2 (weight decay) dentro do Adam, como no PyTorch.
# ============================================================

INICIALIZACOES = ["padrao", "xavier", "he", "normal_0.01", "zeros"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def bce_com_logits(z, y):
    """Entropia cruzada binária estável numericamente."""
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def init_pesos(rng, fan_in, fan_out, modo):
    if modo == "padrao":        # igual ao nn.Linear do PyTorch: U(-1/sqrt(fan_in), +1/sqrt(fan_in))
        lim = 1 / np.sqrt(fan_in)
        return rng.uniform(-lim, lim, (fan_in, fan_out))
    if modo == "xavier":        # Xavier/Glorot uniforme
        lim = np.sqrt(6 / (fan_in + fan_out))
        return rng.uniform(-lim, lim, (fan_in, fan_out))
    if modo == "he":            # He normal (para ReLU)
        return rng.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out))
    if modo == "normal_0.01":
        return rng.normal(0, 0.01, (fan_in, fan_out))
    if modo == "zeros":
        return np.zeros((fan_in, fan_out))
    raise ValueError("inicialização desconhecida: " + modo)


class MLPProfundo:
    def __init__(self, entrada, camadas=(128, 64, 32), dropout=0.3, init="padrao", seed=0):
        self.rng = np.random.default_rng(seed)
        self.n_camadas = len(camadas)
        self.dropout = dropout
        tam = [entrada, *camadas]
        P, self.rm, self.rv = {}, [], []

        for i in range(self.n_camadas):
            fi, fo = tam[i], tam[i + 1]
            P[f"W{i}"] = init_pesos(self.rng, fi, fo, init)
            if init == "padrao":
                lim = 1 / np.sqrt(fi)
                P[f"b{i}"] = self.rng.uniform(-lim, lim, fo)
            else:
                P[f"b{i}"] = np.zeros(fo)
            P[f"g{i}"] = np.ones(fo)     # gamma do batch norm
            P[f"be{i}"] = np.zeros(fo)   # beta do batch norm
            self.rm.append(np.zeros(fo))
            self.rv.append(np.ones(fo))

        fi = tam[-1]
        P["Wo"] = init_pesos(self.rng, fi, 1, init)
        P["bo"] = self.rng.uniform(-1 / np.sqrt(fi), 1 / np.sqrt(fi), 1) if init == "padrao" else np.zeros(1)
        self.P = P

    # ---------------- forward ----------------
    def forward(self, x, treino):
        P, a, cache = self.P, x, []
        for i in range(self.n_camadas):
            z = a @ P[f"W{i}"] + P[f"b{i}"]
            if treino:
                mu, var = z.mean(0), z.var(0)
                n = len(z)
                self.rm[i] = 0.9 * self.rm[i] + 0.1 * mu
                self.rv[i] = 0.9 * self.rv[i] + 0.1 * var * n / max(n - 1, 1)
            else:
                mu, var = self.rm[i], self.rv[i]
            inv = 1.0 / np.sqrt(var + 1e-5)
            xh = (z - mu) * inv
            y = P[f"g{i}"] * xh + P[f"be{i}"]
            h = np.maximum(y, 0)
            mascara = None
            if treino and self.dropout > 0:
                mascara = (self.rng.random(h.shape) >= self.dropout) / (1 - self.dropout)
                h = h * mascara
            cache.append((a, xh, inv, y > 0, mascara))
            a = h
        logits = (a @ P["Wo"] + P["bo"]).ravel()
        return logits, cache, a

    # ---------------- backward ----------------
    def backward(self, dz, cache, a_ult):
        P, G = self.P, {}
        G["Wo"] = a_ult.T @ dz[:, None]
        G["bo"] = np.array([dz.sum()])
        da = dz[:, None] @ P["Wo"].T

        for i in reversed(range(self.n_camadas)):
            a_in, xh, inv, relu_mask, drop = cache[i]
            if drop is not None:
                da = da * drop
            dy = da * relu_mask
            G[f"g{i}"] = (dy * xh).sum(0)
            G[f"be{i}"] = dy.sum(0)
            dxh = dy * P[f"g{i}"]
            n = len(dy)
            dzi = (inv / n) * (n * dxh - dxh.sum(0) - xh * (dxh * xh).sum(0))
            G[f"W{i}"] = a_in.T @ dzi
            G[f"b{i}"] = dzi.sum(0)
            da = dzi @ P[f"W{i}"].T
        return G

    def prever_probs(self, X):
        return sigmoid(self.forward(X, treino=False)[0])

    def estado(self):
        return copy.deepcopy((self.P, self.rm, self.rv))

    def carregar(self, est):
        self.P, self.rm, self.rv = copy.deepcopy(est)


# ============================================================
# 5. TREINO (Adam com L2)
# ============================================================

def treinar(Xtr, ytr, Xva, yva, seed, epocas, init="padrao",
            lr=1e-3, weight_decay=1e-4, batch_size=256, dropout=0.3):
    """
    Treina por `epocas` épocas fixas (para comparar as curvas) e guarda os
    pesos da época com menor perda de validação. Se a perda divergir
    (NaN/inf), o treino para e o resto da curva fica NaN.
    """
    modelo = MLPProfundo(Xtr.shape[1], dropout=dropout, init=init, seed=seed)
    rng = np.random.default_rng(seed + 1000)

    m = {k: np.zeros_like(v) for k, v in modelo.P.items()}
    v = {k: np.zeros_like(p) for k, p in modelo.P.items()}
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0

    h_tr = np.full(epocas, np.nan)
    h_va = np.full(epocas, np.nan)
    melhor, melhor_est = np.inf, None
    n = len(Xtr)

    for ep in range(epocas):
        ordem = rng.permutation(n)
        soma, cont = 0.0, 0
        for ini in range(0, n, batch_size):
            idx = ordem[ini:ini + batch_size]
            if len(idx) < 2:
                continue
            xb, yb = Xtr[idx], ytr[idx]
            logits, cache, a_ult = modelo.forward(xb, treino=True)
            soma += bce_com_logits(logits, yb) * len(idx)
            cont += len(idx)

            dz = (sigmoid(logits) - yb) / len(idx)
            G = modelo.backward(dz, cache, a_ult)

            t += 1
            for k in modelo.P:
                g = G[k] + weight_decay * modelo.P[k]       # L2 / weight decay
                m[k] = b1 * m[k] + (1 - b1) * g
                v[k] = b2 * v[k] + (1 - b2) * g * g
                mh, vh = m[k] / (1 - b1 ** t), v[k] / (1 - b2 ** t)
                modelo.P[k] = modelo.P[k] - lr * mh / (np.sqrt(vh) + eps)

        loss_tr = soma / max(cont, 1)
        loss_va = bce_com_logits(modelo.forward(Xva, treino=False)[0], yva)
        if not (np.isfinite(loss_tr) and np.isfinite(loss_va)):
            break
        h_tr[ep], h_va[ep] = loss_tr, loss_va
        if loss_va < melhor:
            melhor, melhor_est = loss_va, modelo.estado()

    if melhor_est is not None:
        modelo.carregar(melhor_est)
    return modelo, h_tr, h_va, melhor_est is not None


def avaliar_teste(modelo, Xva, yva, Xte, yte):
    lim = limiar_f1_validacao(yva, modelo.prever_probs(Xva))
    p = modelo.prever_probs(Xte)
    r05, f05 = recall_f1(yte, (p >= 0.5).astype(int))
    rl, fl = recall_f1(yte, (p >= lim).astype(int))
    return dict(auc_teste=auc_roc(yte, p), recall_05=r05, f1_05=f05,
                limiar_val=lim, recall_limiar=rl, f1_limiar=fl)


# ============================================================
# 6. EXPERIMENTOS
# ============================================================

CAMPOS_TESTE = ["auc_teste", "recall_05", "f1_05", "limiar_val", "recall_limiar", "f1_limiar"]


def rodar_experimento(nome_exp, configs, dados, epocas, seeds):
    """configs: lista de (rotulo, funcao(seed)->(Xtr,Xva,Xte), init)."""
    linhas, curvas = [], {}
    for rotulo, obter, init in configs:
        curvas[rotulo] = {"treino": [], "val": []}
        print(f"\n[{nome_exp}] {rotulo}")
        for seed in seeds:
            Xtr, Xva, Xte = obter(seed)
            _, _, _, ytr, yva, yte = dados[seed]
            ytr_a = ytr.to_numpy(dtype=np.float64)
            yva_a = yva.to_numpy(dtype=np.float64)

            modelo, h_tr, h_va, ok = treinar(Xtr, ytr_a, Xva, yva_a, seed=seed, epocas=epocas, init=init)
            curvas[rotulo]["treino"].append(h_tr)
            curvas[rotulo]["val"].append(h_va)

            linha = {"experimento": nome_exp, "config": rotulo, "seed": seed}
            linha.update(metricas_convergencia(h_tr, h_va))
            linha.update(avaliar_teste(modelo, Xva, yva_a, Xte, yte.to_numpy()) if ok
                         else {c: np.nan for c in CAMPOS_TESTE})
            linhas.append(linha)
            print(f"   seed={seed}  melhor época={linha['melhor_epoca']}  "
                  f"val loss={linha['min_loss_val']:.4f}  AUC={linha['auc_teste']:.4f}  "
                  f"recall@0.5={linha['recall_05']:.3f}")
    return pd.DataFrame(linhas), curvas


def resumir(df):
    cols = ["auc_teste", "recall_05", "f1_05", "recall_limiar", "f1_limiar",
            "min_loss_val", "melhor_epoca", "epocas_ate_2pct", "gap_final"]
    med = df.groupby("config", sort=False)[cols].mean()
    dp = df.groupby("config", sort=False)[cols].std()
    res = med.copy().astype(object)
    for c in cols:
        res[c] = [f"{a:.4f} ± {b:.4f}" if pd.notna(a) else "diverge (NaN)"
                  for a, b in zip(med[c], dp[c].fillna(0))]
    return res


# ============================================================
# 7. GRÁFICOS
# ============================================================

def plotar_curvas(curvas, titulo, arquivo):
    fig, eixos = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for rotulo, d in curvas.items():
        for ax, chave, nome in ((eixos[0], "treino", "Treino"), (eixos[1], "val", "Validação")):
            mat = np.vstack(d[chave])
            media, desvio = np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)
            x = np.arange(1, len(media) + 1)
            ax.plot(x, media, label=rotulo)
            ax.fill_between(x, media - desvio, media + desvio, alpha=0.15)
            ax.set_title(f"{nome} - perda (BCE)")
            ax.set_xlabel("Época")
    eixos[0].set_ylabel("Loss")
    eixos[0].legend(fontsize=8)
    fig.suptitle(titulo)
    plt.tight_layout()
    plt.savefig(arquivo, dpi=150)
    plt.close()


def plotar_auc(df, titulo, arquivo):
    r = df.groupby("config", sort=False)["auc_teste"].agg(["mean", "std"]).fillna(0)
    plt.figure(figsize=(9, 5))
    plt.bar(range(len(r)), r["mean"], yerr=r["std"], capsize=4)
    plt.xticks(range(len(r)), r.index, rotation=20, ha="right")
    plt.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    plt.ylim(0.4, max(0.85, float(r["mean"].max()) + 0.05))
    plt.ylabel("AUC-ROC (teste)")
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(arquivo, dpi=150)
    plt.close()


# ============================================================
# 8. MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="caminho do dataset (csv/xls/xlsx)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--sintetico", action="store_true", help="força o uso de dados sintéticos")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("P4 - PRÉ-PROCESSAMENTO E INICIALIZAÇÃO | MLP PROFUNDO (offline)")
    print("=" * 70)
    print(f"épocas: {args.epochs} | sementes: {args.seeds}")

    X, y, sintetico = carregar_dataset(args.csv, args.sintetico)
    cat, num = identificar_colunas(X)
    X = limpar_categorias(X, cat)
    print(f"Clientes: {len(X)} | variáveis: {X.shape[1]} | inadimplentes: {y.mean() * 100:.2f}%")
    print(f"Categóricas: {cat}")

    dados = {s: dividir(X, y, s) for s in args.seeds}
    cache = {}

    def obter(nome_cfg):
        normalizar, one_hot = CONFIGS_PREPROC[nome_cfg]

        def f(seed):
            if (nome_cfg, seed) not in cache:
                Xtr, Xva, Xte, *_ = dados[seed]
                cache[(nome_cfg, seed)] = preprocessar(Xtr, Xva, Xte, cat, num, normalizar, one_hot)
            return cache[(nome_cfg, seed)]
        return f

    # ---------- Experimento A ----------
    print("\n" + "=" * 70)
    print("EXPERIMENTO A - Normalização x codificação das categorias")
    print("=" * 70)
    df_a, curvas_a = rodar_experimento(
        "A_preproc", [(n, obter(n), "padrao") for n in CONFIGS_PREPROC], dados, args.epochs, args.seeds)
    resumo_a = resumir(df_a)
    print("\nResumo A (média ± desvio entre sementes):")
    print(resumo_a.to_string())

    melhor_prep = df_a.groupby("config", sort=False)["auc_teste"].mean().idxmax()
    print(f"\nMelhor pré-processamento (AUC média): {melhor_prep}")

    # ---------- Experimento B ----------
    print("\n" + "=" * 70)
    print(f"EXPERIMENTO B - Inicialização dos pesos (pré-proc.: {melhor_prep})")
    print("=" * 70)
    df_b, curvas_b = rodar_experimento(
        "B_init", [(f"init: {i}", obter(melhor_prep), i) for i in INICIALIZACOES],
        dados, args.epochs, args.seeds)
    resumo_b = resumir(df_b)
    print("\nResumo B (média ± desvio entre sementes):")
    print(resumo_b.to_string())

    # ---------- salvar ----------
    pd.concat([df_a, df_b]).to_csv(f"{OUT_DIR}/p4_resultados_por_seed.csv", index=False)
    resumo_a.to_csv(f"{OUT_DIR}/p4_resumo_preprocessamento.csv")
    resumo_b.to_csv(f"{OUT_DIR}/p4_resumo_inicializacao.csv")
    plotar_curvas(curvas_a, "P4-A: perda por época por pré-processamento (MLP profundo)",
                  f"{OUT_DIR}/p4_curvas_preprocessamento.png")
    plotar_curvas(curvas_b, f"P4-B: perda por época por inicialização ({melhor_prep})",
                  f"{OUT_DIR}/p4_curvas_inicializacao.png")
    plotar_auc(df_a, "P4-A: AUC-ROC no teste por pré-processamento",
               f"{OUT_DIR}/p4_auc_preprocessamento.png")
    plotar_auc(df_b, "P4-B: AUC-ROC no teste por inicialização",
               f"{OUT_DIR}/p4_auc_inicializacao.png")

    print("\n" + "=" * 70)
    print(f"Arquivos salvos em ./{OUT_DIR}/")
    if sintetico:
        print("ATENÇÃO: estes resultados vieram de dados SINTÉTICOS (não use no relatório).")
    print("=" * 70)


if __name__ == "__main__":
    main()
