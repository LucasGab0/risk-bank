# Risk Bank

Aplicação e pesquisa sobre previsão de inadimplência de cartão de crédito. A entrega inclui os dados oficiais, modelos treinados, experimentos P1–P4, calibração, código, testes e PDFs atualizados.

Projeto atribuído no repositório a Athos Andre, Lucas Brasileiro, Guilherme Uchoa, Matheus Albuquerque e Dhigor Padilha. Os documentos recebidos foram preservados em `docs/originais/`.

## Abrir a aplicação

Use Python 3.11 ou superior. O ambiente validado é Python 3.12 no Linux. Extraia o ZIP e abra um terminal na pasta que contém `main.py`.

```bash
python -m venv .venv
```

Ative o ambiente:

```bash
# Linux/macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Instale e execute:

```bash
python -m pip install -r requirements.txt
python main.py --serve
```

Abra **http://127.0.0.1:8000**. Os modelos já estão treinados; não é necessário instalar PyTorch nem repetir a pesquisa para usar a aplicação. Após a instalação, a aplicação e a pesquisa com a base incluída funcionam sem internet. Abra a interface pelo servidor, não diretamente pelo `index.html`.

A tela oferece regressão logística, XGBoost e MLP scikit-learn, histórico da sessão e análise de sensibilidade calculada pelo modelo. Não há previsão por regra manual caso o artefato falhe.

## Pesquisa completa executada

Foram realizados **140 ajustes**, correspondentes a **28 configurações em cinco sementes: 7, 21, 42, 84 e 123**.

| Experimento | Comparações |
|---|---|
| P1: modelos | Regressão logística, SVM linear calibrada, kNN L1/L2, random forest, XGBoost e MLP |
| P2: redes | Largura 32/128, duas ou três camadas, ReLU, tanh e sigmoide |
| P3: regularização | L2, dropout, batch normalization, combinações, peso de classe e sobreamostragem |
| P4: preparação | Com/sem normalização, com/sem one-hot, inicialização padrão, Xavier e Kaiming |
| Calibração | Probabilidades originais, sigmoide e isotônica nos três modelos da aplicação |
| Sensibilidade a atributos | Os três modelos com e sem sexo, idade e estado civil |
| Incerteza e custo | Dispersão entre sementes, intervalos bootstrap por perfil e cenários ilustrativos de custo |

As 45 avaliações de calibração reaproveitam predições dos classificadores; não são 45 novos ajustes dos modelos base. As redes PyTorch foram treinadas em CPU de forma determinística, com até 80 épocas e parada antecipada.

O conjunto de referência é o mesmo já consultado na primeira etapa: **não é apresentado como teste inédito**. Ele é mantido fora de todos os novos ajustes, calibração, seleção e escolha de limiares. O desenvolvimento é dividido por perfil em `fit`, `stop`, `calibration` e `validation`; perfis idênticos não atravessam partições. O scaler e o encoder são ajustados apenas em `fit`.

## Resultados atuais

Métricas abaixo: **média ± desvio da AP de validação**, antes do ajuste adicional de calibração.

| Modelo | AP média ± desvio |
|---|---:|
| Regressão logística | 0,495 ± 0,012 |
| SVM linear | 0,492 ± 0,012 |
| kNN L1 | 0,483 ± 0,010 |
| kNN L2 | 0,491 ± 0,010 |
| Random forest | 0,543 ± 0,016 |
| XGBoost | 0,543 ± 0,009 |
| MLP scikit-learn | 0,538 ± 0,011 |
| Melhor rede PyTorch: 64/32, tanh | 0,543 ± 0,011 |

Os melhores candidatos ficaram próximos. O XGBoost foi selecionado entre **os três modelos integrados à aplicação**, pela maior AP média de validação; não foi declarado vencedor de todas as configurações. A rede tanh liderou as 18 redes PyTorch e está salva como artefato de pesquisa separado.

A calibração usa o menor Brier médio de validação. A aplicação recebeu **isotônica na regressão logística**, **nenhum ajuste adicional no XGBoost** e **sigmoide na MLP scikit-learn**. Reduzir Brier não garante melhorar log loss, AP e recall simultaneamente.

Na avaliação complementar da referência, o XGBoost final teve AUC-ROC **0,768**, AP **0,529**, F1 **0,525** e recall **0,512**, com limiar validado de aproximadamente **0,299**. Foram identificados 509 dos 995 inadimplentes; 486 não foram detectados. Não houve seleção usando esses resultados.

Os limiares maximizam F1 na validação da semente 42. As métricas exibidas na tela usam esse limiar exato. O controle visual arredonda para 0,1 ponto percentual; alterar o controle muda a classificação, sem recalcular as métricas da referência.

## Reproduzir os experimentos e PDFs

Instale as dependências opcionais de pesquisa. O arquivo usa o índice oficial de CPU do PyTorch; o pacote sem sufixo CPU é usado no macOS.

```bash
python -m pip install -r requirements-research.txt
python main.py
python build_report.py
```

`python main.py` executa a pesquisa completa com integração e retomada. Os resultados já incluídos são reconhecidos pelo protocolo e reaproveitados; os modelos finais são reproduzidos e as equivalências com as predições registradas são verificadas. Não altere os arquivos de cache para combinar protocolos diferentes.

Comando explícito equivalente:

```bash
python research.py --integrate --resume
```

Para um novo experimento em outra pasta:

```bash
python research.py --output results/outro_estudo --seeds 7 21 42 84 123 --epochs 80
```

Esse comando produz um artefato `application_models.joblib` nessa pasta. Sem `--integrate`, preserva os modelos ativos. O gerador de PDFs lê a pesquisa padrão em `results/research/`. Para um arquivo próprio, use `--data caminho/arquivo.csv`; são aceitos CSV, CSV gzip, XLS e XLSX compatíveis com o esquema UCI.

`train.py` preserva a comparação inicial rápida, de uma semente e sem calibração adicional. Use uma pasta separada para não substituir os modelos finais da aplicação:

```bash
python train.py --output results/baseline_rapido
```

A rede de pesquisa usa PyTorch; a MLP servida pela API usa scikit-learn. Nenhum objeto PyTorch é necessário ao servidor. Os pesos da melhor rede experimental, seu pré-processador e sua configuração estão em `results/research/best_neural_*`.

## Documentos

- `docs/Previsão de Inadimplência de Crédito com Redes Neurais.pdf`: relatório completo, 12 páginas.
- `docs/proposta_pesquisa_inadimplencia.pdf`: síntese atualizada dos experimentos, 4 páginas.
- `docs/apresentacao_resultados.pdf`: apresentação de resultados, 16 slides.
- `PESQUISA.md`: fonte textual com os resultados e as figuras.
- `RELATORIO.md`: diagnóstico da entrega e registro das pendências concluídas.
- `docs/originais/`: PDFs recebidos, preservados com seus hashes.

O relatório responde P1–P4 com os experimentos executados, incluindo resultados que não melhoraram e limites das conclusões. Os intervalos entre sementes são descritivos porque as partições se sobrepõem. Os intervalos bootstrap da referência são condicionais ao modelo ajustado, não à transferência para outra população.

## Testes

```bash
python -m pip install -r requirements-research.txt
python -m pytest -q
```

A validação completa está registrada em `results/pytest.txt`. Os testes de pesquisa e de PDFs são opcionais quando PyTorch e pypdf não estão instalados; nesse caso o pytest informa os módulos ignorados. Para somente os testes da aplicação, instale `requirements-dev.txt`.

Navegador:

```bash
python -m playwright install chromium
python tests/browser_smoke.py
# Ou com Chromium já instalado:
python tests/browser_smoke.py --chromium /usr/bin/chromium
```

O script inicia e encerra seu próprio servidor. Aceita `--screenshots pasta` e `--report arquivo.json`. Os cenários e as probabilidades verificadas estão em `results/browser_tests.json`.

`requirements-lock.txt` registra o ambiente inicial de aplicação/testes; `requirements-research-lock.txt` registra o ambiente completo da pesquisa, ambos com versões fixadas. Carregue apenas artefatos joblib confiáveis. Ao mudar as versões do scikit-learn ou XGBoost, treine novamente.

## API

- `GET /health`: estado de carregamento.
- `GET /models`: modelos, limiares, calibração e métricas.
- `GET /metrics`: relatório do modelo carregado e escolhas da pesquisa.
- `POST /predict`: previsão e sensibilidade.
- `GET /docs`: documentação interativa.

```json
{
  "limit": 50000, "age": 35, "sex": 1, "edu": 2, "mar": 2,
  "pay": [0, 0, 0, 0, 0, 0],
  "bill": [20000, 20000, 20000, 20000, 20000, 20000],
  "paid": [2000, 2000, 2000, 2000, 2000, 2000],
  "model": "xgboost", "threshold": 0.299
}
```

`model` e `threshold` são opcionais. Os arrays seguem **setembro, agosto, julho, junho, maio, abril de 2005**. Os valores são em **NT$**, não em reais. Sexo: 1 masculino, 2 feminino. Escolaridade: 1 pós-graduação, 2 universidade, 3 ensino médio, 4 outro. Estado civil: 1 casado, 2 solteiro, 3 outro. Situação de pagamento: -2 sem consumo, -1 pago em dia, 0 rotativo, 1–8 meses de atraso.

`prob` é a probabilidade estimada. `prediction` é 1 quando ela atinge o limiar. Os fatores comparam a previsão atual com a previsão após substituir um grupo pela mediana/moda de `fit`. `value` é uma diferença de probabilidades; a tela a mostra em pontos percentuais. Não são valores SHAP, efeitos causais nem uma decomposição aditiva. A substituição pode produzir combinações pouco comuns entre variáveis correlacionadas.

Artefato ausente ou corrompido: HTTP 503, sem resultado substituto. Alterar dados ou modelo limpa a previsão anterior. O histórico pertence apenas à sessão da página. Reinicie o servidor após gerar um novo artefato. A variável `RISK_BANK_MODEL` permite apontar para outro arquivo de modelos compatível.

## Arquivos da pesquisa

`protocol.json`: desenho registrado; `experiments.csv`: 140 ajustes; `summary.csv`: médias e desvios; `paired_differences.csv`: diferenças contra a rede de controle; `learning_curves.csv` e `convergence.csv`: perdas e melhores épocas; `calibration.csv`: 45 avaliações; `selection_locked.json`: escolhas por validação; `final_evaluation.json`: referência e intervalos; `cost_scenarios.csv`: custos ilustrativos; `split_seed_*.npz`: índices; `runs/`: predições de calibração e validação por execução. Tudo fica em `results/research/`.

Os resultados da primeira etapa foram preservados em `results/initial_baseline/`. Os arquivos na raiz de `results/` correspondem aos modelos finais calibrados.

## Fonte e limitações

Yeh, I-Cheng; Lien, Che-hui. *Default of Credit Card Clients*. UCI Machine Learning Repository, 2009. DOI: https://doi.org/10.24432/C55S3H. Base sob licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Origem, conversão e hashes em `data/source.json`. Fontes tipográficas DejaVu incluídas com licença em `assets/fonts/`.

Os dados são de um banco de Taiwan em 2005. A análise de grupos e a remoção de atributos sensíveis não demonstram ausência de discriminação. Custos de erros são cenários hipotéticos. Não houve validação externa recente ou temporal; esses resultados não estabelecem desempenho em crédito brasileiro atual.
