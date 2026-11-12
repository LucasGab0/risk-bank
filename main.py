# ============================================================
# PREVISÃO DE INADIMPLÊNCIA DE CARTÃO DE CRÉDITO
# Projeto de Inteligência Artificial
#
# Modelos:
#   - kNN L1
#   - kNN L2
#   - Logistic Regression
#   - Linear SVM
#   - MLP superficial
#   - MLP profundo regularizado
#
# Experimentos:
#   - Funções de ativação
#   - Dropout
#   - L2
#   - Batch Normalization
#   - Diferentes seeds
#
# Dataset:
#   Default of Credit Card Clients - UCI
# ============================================================

import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

SEED = 42

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("PROJETO - PREVISÃO DE INADIMPLÊNCIA")
print("=" * 70)

print(f"Seed: {SEED}")
print(f"Dispositivo utilizado: {DEVICE}")
print()


# ============================================================
# 2. REPRODUTIBILIDADE
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# 3. CARREGAMENTO DO DATASET
# ============================================================

def carregar_dataset():

    print("Carregando dataset da UCI...")

    try:

        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=350)

        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()

        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        y = pd.Series(y).astype(int)

        print("Dataset carregado com sucesso.")

        return X, y

    except Exception as erro:

        print()
        print("Não foi possível baixar o dataset automaticamente.")
        print("Erro:", erro)
        print()

        caminho = input(
            "Digite o caminho para o arquivo CSV/XLS/XLSX: "
        )

        if caminho.endswith(".csv"):

            df = pd.read_csv(caminho)

        else:

            df = pd.read_excel(caminho)

        # Nome padrão do dataset da UCI
        target = "default.payment.next.month"

        if target not in df.columns:

            # Tenta encontrar automaticamente
            possiveis = [
                "default.payment.next.month",
                "default payment next month",
                "default_payment_next_month",
            ]

            encontrado = None

            for coluna in possiveis:

                if coluna in df.columns:
                    encontrado = coluna
                    break

            if encontrado is None:

                raise ValueError(
                    "Não foi possível encontrar a variável alvo."
                )

            target = encontrado

        y = df[target].astype(int)

        X = df.drop(columns=[target])

        return X, y


X, y = carregar_dataset()


# ============================================================
# 4. INSPEÇÃO DOS DADOS
# ============================================================

print()
print("=" * 70)
print("INSPEÇÃO DO DATASET")
print("=" * 70)

print(f"Número de clientes: {len(X)}")
print(f"Número de variáveis: {X.shape[1]}")

print()

print("Distribuição da variável alvo:")

print(y.value_counts())

print()

print("Proporção de inadimplentes:")

print(f"{y.mean() * 100:.2f}%")

print()

print("Valores ausentes:")

print(X.isnull().sum().sum())


# ============================================================
# 5. REMOVER ID
# ============================================================

if "ID" in X.columns:

    X = X.drop(columns=["ID"])

    print()
    print("Coluna ID removida.")


# ============================================================
# 6. IDENTIFICAÇÃO DAS VARIÁVEIS
# ============================================================

colunas_categoricas = [
    coluna
    for coluna in [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
    ]
    if coluna in X.columns
]

colunas_numericas = [
    coluna
    for coluna in X.columns
    if coluna not in colunas_categoricas
]


print()
print("Variáveis categóricas:")

print(colunas_categoricas)

print()

print("Variáveis numéricas:")

print(len(colunas_numericas))


# ============================================================
# 7. DIVISÃO DOS DADOS
#
# 70% TREINO
# 15% VALIDAÇÃO
# 15% TESTE
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=TEST_SIZE + VALIDATION_SIZE,
    stratify=y,
    random_state=SEED,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=SEED,
)


print()
print("=" * 70)
print("DIVISÃO DOS DADOS")
print("=" * 70)

print(f"Treinamento: {len(X_train)}")
print(f"Validação:   {len(X_val)}")
print(f"Teste:       {len(X_test)}")


# ============================================================
# 8. PRÉ-PROCESSAMENTO
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numericas",
            StandardScaler(),
            colunas_numericas,
        ),
        (
            "categoricas",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            colunas_categoricas,
        ),
    ]
)


# IMPORTANTE:
# O scaler é ajustado SOMENTE no conjunto de treinamento.

X_train_processed = preprocessor.fit_transform(X_train)

X_val_processed = preprocessor.transform(X_val)

X_test_processed = preprocessor.transform(X_test)


X_train_processed = np.asarray(
    X_train_processed,
    dtype=np.float32,
)

X_val_processed = np.asarray(
    X_val_processed,
    dtype=np.float32,
)

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float32,
)


print()
print("Dimensão após pré-processamento:")

print("Treino:", X_train_processed.shape)
print("Validação:", X_val_processed.shape)
print("Teste:", X_test_processed.shape)


# ============================================================
# 9. FUNÇÃO DE AVALIAÇÃO
# ============================================================

def avaliar_modelo(
    nome,
    modelo,
    X_teste,
    y_teste,
):

    previsoes = modelo.predict(X_teste)

    if hasattr(modelo, "predict_proba"):

        probabilidades = modelo.predict_proba(
            X_teste
        )[:, 1]

    else:

        probabilidades = modelo.decision_function(
            X_teste
        )

    accuracy = accuracy_score(
        y_teste,
        previsoes,
    )

    precision = precision_score(
        y_teste,
        previsoes,
        zero_division=0,
    )

    recall = recall_score(
        y_teste,
        previsoes,
        zero_division=0,
    )

    f1 = f1_score(
        y_teste,
        previsoes,
        zero_division=0,
    )

    auc = roc_auc_score(
        y_teste,
        probabilidades,
    )

    print()
    print("-" * 70)
    print(nome)
    print("-" * 70)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")

    print()
    print("Matriz de confusão:")

    print(
        confusion_matrix(
            y_teste,
            previsoes,
        )
    )

    return {
        "Modelo": nome,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC_ROC": auc,
    }


# ============================================================
# 10. MODELO KNN - DISTÂNCIA L1
# ============================================================

print()
print("=" * 70)
print("TREINANDO KNN L1")
print("=" * 70)

knn_l1 = KNeighborsClassifier(
    n_neighbors=15,
    p=1,
    weights="distance",
)

knn_l1.fit(
    X_train_processed,
    y_train,
)

resultado_knn_l1 = avaliar_modelo(
    "kNN L1",
    knn_l1,
    X_test_processed,
    y_test,
)


# ============================================================
# 11. MODELO KNN - DISTÂNCIA L2
# ============================================================

print()
print("=" * 70)
print("TREINANDO KNN L2")
print("=" * 70)

knn_l2 = KNeighborsClassifier(
    n_neighbors=15,
    p=2,
    weights="distance",
)

knn_l2.fit(
    X_train_processed,
    y_train,
)

resultado_knn_l2 = avaliar_modelo(
    "kNN L2",
    knn_l2,
    X_test_processed,
    y_test,
)


# ============================================================
# 12. CLASSIFICADOR LINEAR
#
# Logistic Regression = cross entropy / softmax
# ============================================================

print()
print("=" * 70)
print("TREINANDO CLASSIFICADOR LINEAR")
print("=" * 70)

modelo_linear = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=SEED,
)

modelo_linear.fit(
    X_train_processed,
    y_train,
)

resultado_linear = avaliar_modelo(
    "Logistic Regression",
    modelo_linear,
    X_test_processed,
    y_test,
)


# ============================================================
# 13. LINEAR SVM
#
# Hinge Loss
# ============================================================

print()
print("=" * 70)
print("TREINANDO LINEAR SVM")
print("=" * 70)

modelo_svm = LinearSVC(
    C=1.0,
    class_weight="balanced",
    max_iter=5000,
    random_state=SEED,
)

modelo_svm.fit(
    X_train_processed,
    y_train,
)

resultado_svm = avaliar_modelo(
    "Linear SVM",
    modelo_svm,
    X_test_processed,
    y_test,
)


# ============================================================
# 14. MODELO MLP
# ============================================================

class MLP(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_layers=(64,),
        activation="relu",
        dropout=0.0,
        batch_norm=False,
    ):

        super().__init__()

        layers = []

        tamanho_entrada = input_size

        if activation == "relu":

            funcao_ativacao = nn.ReLU

        elif activation == "tanh":

            funcao_ativacao = nn.Tanh

        elif activation == "sigmoid":

            funcao_ativacao = nn.Sigmoid

        else:

            raise ValueError(
                "Ativação inválida."
            )

        for tamanho_oculta in hidden_layers:

            layers.append(
                nn.Linear(
                    tamanho_entrada,
                    tamanho_oculta,
                )
            )

            if batch_norm:

                layers.append(
                    nn.BatchNorm1d(
                        tamanho_oculta
                    )
                )

            layers.append(
                funcao_ativacao()
            )

            if dropout > 0:

                layers.append(
                    nn.Dropout(dropout)
                )

            tamanho_entrada = tamanho_oculta

        layers.append(
            nn.Linear(
                tamanho_entrada,
                1,
            )
        )

        self.rede = nn.Sequential(*layers)

    def forward(self, x):

        return self.rede(x)


# ============================================================
# 15. TREINAMENTO DA MLP
# ============================================================

def treinar_mlp(
    modelo,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=50,
    learning_rate=0.001,
    weight_decay=0.0,
):

    modelo = modelo.to(DEVICE)

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32,
    ).reshape(-1, 1)

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32,
    )

    y_val_tensor = torch.tensor(
        y_val.values,
        dtype=torch.float32,
    ).reshape(-1, 1)

    dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # ========================================================
    # BALANCEAMENTO
    # ========================================================

    quantidade_positivos = y_train.sum()

    quantidade_negativos = len(y_train) - quantidade_positivos

    peso_positivo = (
        quantidade_negativos /
        quantidade_positivos
    )

    pos_weight = torch.tensor(
        [peso_positivo],
        dtype=torch.float32,
    ).to(DEVICE)

    # BCEWithLogitsLoss =
    # Binary Cross Entropy + estabilidade numérica

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        modelo.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    historico_treino = []
    historico_validacao = []

    melhor_loss = float("inf")

    paciencia = 8
    contador_paciencia = 0

    melhor_estado = None

    # ========================================================
    # LOOP DE TREINAMENTO
    # ========================================================

    for epoch in range(epochs):

        modelo.train()

        losses = []

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(DEVICE)

            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            saida = modelo(
                batch_X
            )

            loss = criterion(
                saida,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            losses.append(
                loss.item()
            )

        loss_treino = np.mean(
            losses
        )

        # ====================================================
        # VALIDAÇÃO
        # ====================================================

        modelo.eval()

        with torch.no_grad():

            saida_val = modelo(
                X_val_tensor.to(DEVICE)
            )

            loss_val = criterion(
                saida_val,
                y_val_tensor.to(DEVICE),
            ).item()

        historico_treino.append(
            loss_treino
        )

        historico_validacao.append(
            loss_val
        )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if loss_val < melhor_loss:

            melhor_loss = loss_val

            melhor_estado = {
                chave: valor.clone()
                for chave, valor
                in modelo.state_dict().items()
            }

            contador_paciencia = 0

        else:

            contador_paciencia += 1

        if contador_paciencia >= paciencia:

            print(
                f"Early stopping na época {epoch + 1}"
            )

            break

        if (
            epoch + 1 == 1
            or (epoch + 1) % 10 == 0
        ):

            print(
                f"Época {epoch + 1:03d} | "
                f"Treino: {loss_treino:.4f} | "
                f"Validação: {loss_val:.4f}"
            )

    # Recuperar melhor modelo

    if melhor_estado is not None:

        modelo.load_state_dict(
            melhor_estado
        )

    return (
        modelo,
        historico_treino,
        historico_validacao,
    )


# ============================================================
# 16. PREDIÇÃO DA MLP
# ============================================================

def avaliar_mlp(
    nome,
    modelo,
    X_test,
    y_test,
):

    modelo.eval()

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
    ).to(DEVICE)

    with torch.no_grad():

        logits = modelo(
            X_tensor
        )

        probabilidades = torch.sigmoid(
            logits
        ).cpu().numpy().flatten()

    previsoes = (
        probabilidades >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        previsoes,
    )

    precision = precision_score(
        y_test,
        previsoes,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        previsoes,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        previsoes,
        zero_division=0,
    )

    auc = roc_auc_score(
        y_test,
        probabilidades,
    )

    print()
    print("-" * 70)
    print(nome)
    print("-" * 70)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")

    print()
    print("Matriz de confusão:")

    print(
        confusion_matrix(
            y_test,
            previsoes,
        )
    )

    return {
        "Modelo": nome,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC_ROC": auc,
    }


# ============================================================
# 17. MLP SUPERFICIAL
# ============================================================

print()
print("=" * 70)
print("MLP SUPERFICIAL")
print("=" * 70)

mlp_superficial = MLP(
    input_size=X_train_processed.shape[1],
    hidden_layers=(64,),
    activation="relu",
)


(
    mlp_superficial,
    historico_superficial_treino,
    historico_superficial_val,
) = treinar_mlp(
    mlp_superficial,
    X_train_processed,
    y_train,
    X_val_processed,
    y_val,
)


resultado_mlp_superficial = avaliar_mlp(
    "MLP Superficial",
    mlp_superficial,
    X_test_processed,
    y_test,
)


# ============================================================
# 18. MLP PROFUNDA
#
# L2 + Dropout + Batch Normalization
# ============================================================

print()
print("=" * 70)
print("MLP PROFUNDA REGULARIZADA")
print("=" * 70)

mlp_profunda = MLP(
    input_size=X_train_processed.shape[1],

    hidden_layers=(
        128,
        64,
        32,
    ),

    activation="relu",

    dropout=0.30,

    batch_norm=True,
)


(
    mlp_profunda,
    historico_profundo_treino,
    historico_profundo_val,
) = treinar_mlp(
    mlp_profunda,
    X_train_processed,
    y_train,
    X_val_processed,
    y_val,

    weight_decay=0.0001,
)


resultado_mlp_profunda = avaliar_mlp(
    "MLP Profunda Regularizada",
    mlp_profunda,
    X_test_processed,
    y_test,
)


# ============================================================
# 19. COMPARAÇÃO FINAL
# ============================================================

resultados = [

    resultado_knn_l1,

    resultado_knn_l2,

    resultado_linear,

    resultado_svm,

    resultado_mlp_superficial,

    resultado_mlp_profunda,

]


df_resultados = pd.DataFrame(
    resultados
)


df_resultados = df_resultados.sort_values(
    by="F1",
    ascending=False,
)


print()
print()
print("=" * 70)
print("RESULTADOS FINAIS")
print("=" * 70)

print(
    df_resultados.to_string(
        index=False
    )
)


# ============================================================
# 20. SALVAR RESULTADOS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


df_resultados.to_csv(
    "results/resultados.csv",
    index=False,
)


# ============================================================
# 21. MATRIZ DE CONFUSÃO
# ============================================================

nomes = [
    "kNN L1",
    "kNN L2",
    "Logistic Regression",
    "Linear SVM",
    "MLP Superficial",
    "MLP Profunda Regularizada",
]


modelos = [
    knn_l1,
    knn_l2,
    modelo_linear,
    modelo_svm,
]


fig, axes = plt.subplots(
    2,
    3,
    figsize=(14, 8),
)


# Modelos sklearn

for ax, nome, modelo in zip(
    axes.flat[:4],
    nomes[:4],
    modelos,
):

    previsoes = modelo.predict(
        X_test_processed
    )

    matriz = confusion_matrix(
        y_test,
        previsoes,
    )

    ax.imshow(matriz)

    ax.set_title(nome)

    ax.set_xlabel(
        "Predito"
    )

    ax.set_ylabel(
        "Real"
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                matriz[i, j],
                ha="center",
                va="center",
            )


# ============================================================
# MLPs
# ============================================================

mlps = [
    (
        "MLP Superficial",
        mlp_superficial,
    ),
    (
        "MLP Profunda",
        mlp_profunda,
    ),
]


for ax, (nome, modelo) in zip(
    axes.flat[4:],
    mlps,
):

    modelo.eval()

    X_tensor = torch.tensor(
        X_test_processed,
        dtype=torch.float32,
    ).to(DEVICE)

    with torch.no_grad():

        probabilidades = torch.sigmoid(
            modelo(X_tensor)
        ).cpu().numpy().flatten()

    previsoes = (
        probabilidades >= 0.5
    ).astype(int)

    matriz = confusion_matrix(
        y_test,
        previsoes,
    )

    ax.imshow(matriz)

    ax.set_title(nome)

    ax.set_xlabel(
        "Predito"
    )

    ax.set_ylabel(
        "Real"
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                matriz[i, j],
                ha="center",
                va="center",
            )


plt.tight_layout()

plt.savefig(
    "results/matrizes_confusao.png",
    dpi=150,
)

plt.close()


# ============================================================
# 22. CURVAS DE TREINAMENTO
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    historico_superficial_treino,
    label="MLP superficial - treino",
)

plt.plot(
    historico_superficial_val,
    linestyle="--",
    label="MLP superficial - validação",
)


plt.plot(
    historico_profundo_treino,
    label="MLP profunda - treino",
)

plt.plot(
    historico_profundo_val,
    linestyle="--",
    label="MLP profunda - validação",
)


plt.xlabel(
    "Época"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Curvas de treinamento e validação"
)

plt.legend()

plt.tight_layout()


plt.savefig(
    "results/curvas_treinamento.png",
    dpi=150,
)

plt.close()


# ============================================================
# 23. RELATÓRIO FINAL
# ============================================================

print()
print("=" * 70)
print("PROJETO FINALIZADO")
print("=" * 70)

print()
print("Arquivos gerados:")

print(
    "results/resultados.csv"
)

print(
    "results/matrizes_confusao.png"
)

print(
    "results/curvas_treinamento.png"
)

print()
print("Melhor modelo segundo F1:")

print(
    df_resultados.iloc[0]["Modelo"]
)

print(
    f"F1 = {df_resultados.iloc[0]['F1']:.4f}"
)

print(
    f"AUC = {df_resultados.iloc[0]['AUC_ROC']:.4f}"
)

print()
print("Experimento concluído.")