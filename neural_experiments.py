"""Redes neurais em CPU para os experimentos controlados da pesquisa."""
from dataclasses import asdict, dataclass
from math import ceil

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class NeuralConfig:
    key: str
    question: str
    label: str
    hidden: tuple = (64, 32)
    activation: str = "relu"
    l2: float = 0.0
    dropout: float = 0.0
    batch_norm: bool = False
    balance: str = "none"
    scale: bool = True
    one_hot: bool = True
    initialization: str = "default"
    learning_rate: float = 0.001
    batch_size: int = 512


def configurations():
    specs = [
        ("nn_base", "base", "MLP 64/32, ReLU", {}),
        ("nn_shallow_32", "P2", "Uma camada: 32", {"hidden": (32,)}),
        ("nn_shallow_128", "P2", "Uma camada: 128", {"hidden": (128,)}),
        ("nn_deep", "P2", "Três camadas: 128/64/32", {"hidden": (128, 64, 32)}),
        ("nn_tanh", "P2", "Ativação tanh", {"activation": "tanh"}),
        ("nn_sigmoid", "P2", "Ativação sigmoide", {"activation": "sigmoid"}),
        ("nn_l2", "P3", "Somente L2", {"l2": 0.001}),
        ("nn_dropout", "P3", "Somente dropout", {"dropout": 0.3}),
        ("nn_batchnorm", "P3", "Somente batch normalization", {"batch_norm": True}),
        ("nn_regularized", "P3", "L2 + dropout + batch normalization", {"l2": 0.001, "dropout": 0.3, "batch_norm": True}),
        ("nn_weighted", "P3", "Peso de classe", {"balance": "weight"}),
        ("nn_oversampling", "P3", "Sobreamostragem aleatória", {"balance": "oversample"}),
        ("nn_regularized_weighted", "P3", "Regularização conjunta + peso", {"l2": 0.001, "dropout": 0.3, "batch_norm": True, "balance": "weight"}),
        ("nn_unscaled", "P4", "Sem normalização", {"scale": False}),
        ("nn_ordinal", "P4", "Sem one-hot", {"one_hot": False}),
        ("nn_unscaled_ordinal", "P4", "Sem normalização e sem one-hot", {"scale": False, "one_hot": False}),
        ("nn_xavier", "P4", "Inicialização Xavier", {"initialization": "xavier"}),
        ("nn_kaiming", "P4", "Inicialização Kaiming", {"initialization": "kaiming"}),
    ]
    return [NeuralConfig(key, question, label, **kwargs) for key, question, label, kwargs in specs]


class CreditNetwork(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        activation = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[config.activation]
        layers, width = [], input_size
        for size in config.hidden:
            layers.append(nn.Linear(width, size))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(activation())
            if config.dropout:
                layers.append(nn.Dropout(config.dropout))
            width = size
        layers.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)
                if config.initialization == "xavier":
                    gain = nn.init.calculate_gain(config.activation)
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                elif config.initialization == "kaiming":
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def forward(self, data):
        return self.layers(data).ravel()


def train_network(xfit, yfit, xstop, ystop, config, seed, max_epochs=80, patience=10):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    network = CreditNetwork(xfit.shape[1], config)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate, weight_decay=config.l2)
    xt = torch.as_tensor(np.asarray(xfit, dtype=np.float32))
    yt = torch.as_tensor(np.asarray(yfit, dtype=np.float32))
    xs = torch.as_tensor(np.asarray(xstop, dtype=np.float32))
    ys = torch.as_tensor(np.asarray(ystop, dtype=np.float32))
    positives = np.flatnonzero(np.asarray(yfit) == 1)
    negatives = np.flatnonzero(np.asarray(yfit) == 0)
    pos_weight = torch.tensor(len(negatives) / len(positives)) if config.balance == "weight" else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    evaluation_loss = nn.BCEWithLogitsLoss()
    indices = np.arange(len(yfit))
    if config.balance == "oversample":
        indices = np.r_[negatives, rng.choice(positives, len(negatives), replace=True)]
    best_loss, best_state, best_epoch, stale, curve = float("inf"), None, 0, 0, []
    for epoch in range(1, max_epochs + 1):
        network.train()
        # Divisão equilibrada evita o último lote unitário em batch normalization.
        batches = np.array_split(rng.permutation(indices), max(1, ceil(len(indices) / config.batch_size)))
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(network(xt[batch]), yt[batch])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Perda não finita: {config.key}, semente {seed}.")
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
        network.eval()
        with torch.no_grad():
            training_loss = float(evaluation_loss(network(xt), yt))
            stopping_loss = float(evaluation_loss(network(xs), ys))
        curve.append({"epoch": epoch, "train_loss": training_loss, "stop_loss": stopping_loss})
        if stopping_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = stopping_loss, epoch, 0
            best_state = {name: tensor.detach().clone() for name, tensor in network.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise FloatingPointError("Nenhum estado válido da rede foi encontrado.")
    network.load_state_dict(best_state)
    network.eval()
    return network, curve, {"best_epoch": best_epoch, "epochs_run": epoch,
                            "stopping_loss": best_loss, "fit_rows_after_sampling": len(indices),
                            "config": asdict(config)}


def probabilities(network, values):
    network.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(np.asarray(values, dtype=np.float32))
        result = torch.sigmoid(network(tensor)).numpy().astype(float)
    if not np.isfinite(result).all():
        raise FloatingPointError("A rede produziu probabilidades não finitas.")
    return result
