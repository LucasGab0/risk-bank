import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib
import torch
import torch.nn as nn

# Diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent

# Definição da arquitetura MLP para inferência
class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_layers=(128, 64, 32),
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = True,
    ):
        super().__init__()
        layers = []
        tamanho_entrada = input_size
        act_cls = nn.ReLU if activation == "relu" else (nn.Tanh if activation == "tanh" else nn.Sigmoid)

        for tamanho_oculta in hidden_layers:
            layers.append(nn.Linear(tamanho_entrada, tamanho_oculta))
            if batch_norm:
                layers.append(nn.BatchNorm1d(tamanho_oculta))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            tamanho_entrada = tamanho_oculta

        layers.append(nn.Linear(tamanho_entrada, 1))
        self.rede = nn.Sequential(*layers)

    def forward(self, x):
        return self.rede(x)


app = FastAPI(
    title="Risk Bank API",
    description="Backend para previsão de risco de inadimplência de crédito",
    version="1.0.0",
)

# Configuração de CORS para permitir acesso local ou via rede
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Esquemas de dados
class Factor(BaseModel):
    name: str
    value: float

class PredictRequest(BaseModel):
    limit: float = Field(..., description="Limite de crédito (NT$)")
    age: int = Field(..., description="Idade do cliente")
    sex: int = Field(..., description="1 = Masculino, 2 = Feminino")
    edu: int = Field(..., description="1 = Pós-graduação, 2 = Universidade, 3 = Ensino Médio, 4 = Outro")
    mar: int = Field(..., description="1 = Casado, 2 = Solteiro, 3 = Outro")
    pay: List[int] = Field(..., min_length=6, max_length=6, description="Status de pagamento de Set a Abr (6 meses)")
    bill: List[float] = Field(..., min_length=6, max_length=6, description="Valores das faturas de Set a Abr (6 meses)")
    paid: List[float] = Field(..., min_length=6, max_length=6, description="Valores pagos de Set a Abr (6 meses)")

class PredictResponse(BaseModel):
    prob: float
    factors: List[Factor]
    model_used: str

# Carregamento do modelo e pré-processador
model_instance = None
preprocessor_instance = None
device = torch.device("cpu")

def carregar_modelo():
    global model_instance, preprocessor_instance
    model_path = BASE_DIR / "results" / "mlp_profunda.pt"
    prep_path = BASE_DIR / "results" / "preprocessor.joblib"

    if model_path.exists() and prep_path.exists():
        try:
            preprocessor_instance = joblib.load(prep_path)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            input_size = checkpoint.get("input_size", 33)
            hidden_layers = checkpoint.get("hidden_layers", (128, 64, 32))
            dropout = checkpoint.get("dropout", 0.30)
            batch_norm = checkpoint.get("batch_norm", True)
            activation = checkpoint.get("activation", "relu")
            
            mlp = MLP(
                input_size=input_size,
                hidden_layers=hidden_layers,
                activation=activation,
                dropout=dropout,
                batch_norm=batch_norm,
            )
            mlp.load_state_dict(checkpoint["state_dict"])
            mlp.eval()
            model_instance = mlp
            print("[INFO] Modelo PyTorch e Pré-processador carregados com sucesso de results/")
        except Exception as e:
            print(f"[AVISO] Erro ao carregar modelo treinado: {e}. Usando heurística.")
            model_instance = None
            preprocessor_instance = None
    else:
        print("[INFO] Modelo treinado ainda não encontrado em results/. Usando heurística calibrada.")

# Tenta carregar na inicialização do script
carregar_modelo()


@app.on_event("startup")
def startup_event():
    carregar_modelo()


def calcular_fatores(x: PredictRequest) -> List[Factor]:
    """Calcula os fatores de explicabilidade que mais pesam na decisão."""
    w = [0.55, 0.25, 0.15, 0.1, 0.08, 0.05]
    c = lambda v: max(-1, min(6, v))
    
    # 1. Atrasos nos pagamentos
    fat_atrasos = 0.9 * sum(w[i] * c(x.pay[i]) for i in range(6))
    
    # 2. Utilização do limite
    util = min(1.5, max(0.0, x.bill[0] / x.limit)) if x.limit > 0 else 1.0
    fat_util = 0.9 * (util - 0.4)
    
    # 3. Proporção paga da fatura anterior
    ratio = min(1.0, x.paid[0] / x.bill[1]) if x.bill[1] > 0 else 1.0
    fat_pago = -0.9 * (ratio - 0.15)
    
    # 4. Limite de crédito
    fat_limite = -0.25 * np.log10(max(x.limit, 1000.0) / 50000.0)
    
    factors = [
        Factor(name="Atrasos no pagamento", value=round(float(fat_atrasos), 3)),
        Factor(name="Fatura atual vs. limite", value=round(float(fat_util), 3)),
        Factor(name="Parcela paga da fatura anterior", value=round(float(fat_pago), 3)),
        Factor(name="Limite de crédito", value=round(float(fat_limite), 3)),
    ]
    return factors


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Endpoint para calcular a probabilidade de inadimplência de crédito.
    Recebe os dados do cliente e histórico de 6 meses.
    """
    try:
        factors = calcular_fatores(req)
        
        # Se o modelo PyTorch treinado estiver disponível, executa inferência real
        if model_instance is not None and preprocessor_instance is not None:
            # Montar DataFrame com os 23 atributos na ordem correta
            data = {
                "LIMIT_BAL": [req.limit],
                "SEX": [req.sex],
                "EDUCATION": [req.edu],
                "MARRIAGE": [req.mar],
                "AGE": [req.age],
                "PAY_0": [req.pay[0]],
                "PAY_2": [req.pay[1]],
                "PAY_3": [req.pay[2]],
                "PAY_4": [req.pay[3]],
                "PAY_5": [req.pay[4]],
                "PAY_6": [req.pay[5]],
                "BILL_AMT1": [req.bill[0]],
                "BILL_AMT2": [req.bill[1]],
                "BILL_AMT3": [req.bill[2]],
                "BILL_AMT4": [req.bill[3]],
                "BILL_AMT5": [req.bill[4]],
                "BILL_AMT6": [req.bill[5]],
                "PAY_AMT1": [req.paid[0]],
                "PAY_AMT2": [req.paid[1]],
                "PAY_AMT3": [req.paid[2]],
                "PAY_AMT4": [req.paid[3]],
                "PAY_AMT5": [req.paid[4]],
                "PAY_AMT6": [req.paid[5]],
            }
            df = pd.DataFrame(data)
            X_proc = preprocessor_instance.transform(df)
            X_proc = np.asarray(X_proc, dtype=np.float32)
            X_tensor = torch.tensor(X_proc, dtype=torch.float32).to(device)

            with torch.no_grad():
                logits = model_instance(X_tensor)
                prob = float(torch.sigmoid(logits).cpu().numpy().flatten()[0])

            model_name = "MLP Profunda Regularizada (PyTorch)"
        else:
            # Heurística calibrada com a sigmoide
            score = -1.45 + sum(f.value for f in factors)
            prob = float(1.0 / (1.0 + np.exp(-score)))
            model_name = "Heurística Calibrada (Modo Demonstração)"

        prob = max(0.01, min(0.99, prob))
        return PredictResponse(
            prob=round(prob, 4),
            factors=factors,
            model_used=model_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento da predição: {str(e)}")


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": model_instance is not None,
    }


# Rotas para servir o frontend
@app.get("/")
def get_index():
    return FileResponse(BASE_DIR / "index.html")

@app.get("/style.css")
def get_css():
    return FileResponse(BASE_DIR / "style.css", media_type="text/css")

@app.get("/app.js")
def get_js():
    return FileResponse(BASE_DIR / "app.js", media_type="application/javascript")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("Iniciando Risk Bank Server...")
    print("Acesse localmente em:  http://localhost:8000")
    print("Acesse em outro PC/celular (LAN): http://192.168.1.29:8000")
    print("Documentação da API:   http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
