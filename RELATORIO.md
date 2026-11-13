# Entrega final — Risk Bank

O fluxo funcional e a ampliação acadêmica solicitada foram executados. O pacote reúne aplicação, base oficial, modelos treinados e calibrados, experimentos, resultados brutos, testes e PDFs atualizados.

## Pendências acadêmicas concluídas

| Pendência anterior | Trabalho executado |
|---|---|
| Repetir com outras sementes | Cinco sementes: 7, 21, 42, 84 e 123; 28 configurações por semente, total de 140 ajustes |
| Comparar arquiteturas e ativações | Redes com uma, duas e três camadas; larguras 32/128; ReLU, tanh e sigmoide |
| Comparar regularização | L2, dropout e batch normalization, individualmente e em combinação |
| Avaliar balanceamento | Peso de classe e sobreamostragem aleatória, aplicados somente em fit |
| Avaliar preparação e inicialização | Normalização e one-hot ligados/desligados; distribuição padrão, Xavier e Kaiming |
| Comparar baselines | Regressão logística, SVM linear, kNN L1/L2, random forest, XGBoost e MLP |
| Avaliar calibração | 45 avaliações de probabilidades originais, sigmoide e isotônica nos três modelos da aplicação |
| Relatar incerteza | Média, desvio, intervalos descritivos e diferenças pareadas entre sementes; 1.000 bootstraps por perfil na referência |
| Aprofundar análise de atributos | Três ablações de sexo/idade/estado civil e métricas por grupo |
| Avaliar custos | Cenários ilustrativos FN/FP de 1/1, 2/1, 5/1 e 10/1, com limiares escolhidos na validação |
| Atualizar PDFs | Relatório de 12 páginas, síntese de 4 páginas e apresentação de 16 slides; originais preservados |

## Resultados principais

A melhor rede PyTorch foi a MLP 64/32 com tanh: AP média de validação 0,543 ± 0,011. Random forest e XGBoost tiveram resultados próximos. Entre os três modelos já integrados à aplicação, o XGBoost liderou a AP média e permaneceu como padrão. Não foi afirmado que ele superou todos os candidatos.

A calibração foi escolhida pelo Brier médio de validação: isotônica na regressão logística, nenhuma alteração adicional no XGBoost e sigmoide na MLP scikit-learn. Brier e log loss não melhoraram sempre juntos; essa diferença está documentada.

Na referência já consultada, o XGBoost final teve AP 0,529, AUC-ROC 0,768, F1 0,525 e recall 0,512. Foram detectados 509 dos 995 inadimplentes; 486 não foram detectados. Os resultados completos e os intervalos estão nos CSVs e JSONs da pesquisa.

## O que foi corrigido no código

- Estrutura duplicada removida da entrega ativa; documentos originais preservados.
- Frontend, API, dados, pré-processamento e modelos conectados.
- Previsões por fórmulas manuais eliminadas, inclusive em caso de falha do modelo.
- Fatores manuais substituídos por sensibilidade calculada pelo modelo efetivamente usado.
- Entradas inválidas rejeitadas, incluindo listas de tamanho incorreto, categorias inválidas, booleanos, NaN e infinito.
- Perfis idênticos mantidos na mesma partição, sem atravessar desenvolvimento e referência.
- Dados sintéticos automáticos removidos do fluxo; a base oficial está incluída e identificada por hash.
- Alterações de dados/modelo limpam a previsão anterior e cancelam requisições pendentes.
- Textos de demonstração e comentários genéricos revisados; atribuições e referências preservadas.
- Modelos calibrados, métricas, gráficos e documentação atualizados de forma consistente.

## Validação

A validação usa ambiente virtual, Python 3.12, Linux e bibliotecas com versões registradas. Os testes cobrem a aplicação, a separação de dados, calibração sem reajuste do modelo base, reprodução determinística das redes e consistência dos documentos. Os resultados estão em `results/pytest.txt` e `results/browser_tests.json`.

As predições reproduzidas dos modelos finais foram comparadas numericamente com as registradas. O conjunto de referência original permanece o mesmo em todas as cinco sementes. O protocolo e a seleção estão registrados, sem uso de resultados da referência para refazer as escolhas.

## Limitações que permanecem

O conjunto de referência já havia sido consultado na primeira etapa. A ampliação não o transforma em teste inédito; os resultados são apresentados como avaliação complementar. As partições das cinco sementes se sobrepõem, portanto seus intervalos são descritivos. Bootstrap na referência é condicional ao modelo treinado, sem cobrir toda a incerteza de seleção ou transferência de população.

A pesquisa usa uma grade limitada, não uma busca exaustiva. Dados externos recentes e custos reais de uma instituição não estão disponíveis no material enviado. Consequentemente, não há validação de crédito brasileiro atual, desempenho operacional ou ausência de discriminação. Essas limitações estão explicitadas nos PDFs, sem substituir experimentos por resultados inventados.

## Leitura e execução

Comece por `README.md`. O relatório acadêmico completo está em `PESQUISA.md` e em `docs/Previsão de Inadimplência de Crédito com Redes Neurais.pdf`.

A aplicação já contém os modelos treinados:

```bash
python -m pip install -r requirements.txt
python main.py --serve
```

Abra `http://127.0.0.1:8000`. Para reproduzir a pesquisa, instale `requirements-research.txt`, execute `python main.py` e depois `python build_report.py`.
