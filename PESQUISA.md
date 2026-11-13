# Previsão de Inadimplência de Crédito com Redes Neurais

Relatório dos experimentos executados. Documentos anteriores preservados em docs/originais.

## Resumo

Este estudo compara classificadores de inadimplência de cartão de crédito e investiga o efeito de arquitetura, ativação, regularização, balanceamento e preparação dos dados em redes neurais. Foram executados 140 ajustes: 28 configurações em cinco sementes, usando os 30.000 registros oficiais do dataset Default of Credit Card Clients.

Entre as redes PyTorch avaliadas, Ativação tanh obteve a maior AP média de validação (0,543 ± 0,011). O XGBoost alcançou 0,543 ± 0,009 e permaneceu como padrão entre os três modelos integrados à aplicação. As diferenças pequenas entre os melhores candidatos não demonstram superioridade geral de um algoritmo.

A calibração selecionada pelo Brier médio de validação foi: regressão logística, isotonic; XGBoost, none; MLP scikit-learn, sigmoid. Os resultados de referência são complementares: o conjunto já havia sido consultado na etapa anterior e não é apresentado como teste inédito.

Palavras-chave: inadimplência; classificação binária; redes neurais; XGBoost; calibração de probabilidades.

## 1. Problema e perguntas de pesquisa

O objetivo é estimar se um cliente deixará de pagar no mês seguinte a partir do limite, perfil e histórico de seis meses. Como cerca de 22,1% dos registros pertencem à classe inadimplente, acurácia isolada é insuficiente.

P1: uma MLP supera kNN e classificadores lineares? P2: como largura, profundidade e ativação afetam o desempenho e a convergência? P3: quais efeitos são observados com L2, dropout, batch normalization e balanceamento? P4: como normalização, one-hot e inicialização mudam os resultados?

A regressão logística é o modelo de regressão apropriado ao alvo binário. XGBoost e random forest são referências adicionais para dados tabulares. kNN é um método supervisionado baseado em instâncias; sua etapa de ajuste armazena os exemplos, e não aprende uma parametrização como a rede neural.

## 2. Dados, limpeza e partições

A base reúne 30.000 clientes de um banco de Taiwan em 2005, com 6.636 inadimplentes, 23 atributos explicativos e um alvo binário. O ID é excluído dos modelos. A cópia incluída preserva a planilha oficial; origem e hashes estão em data/source.json. Licença: CC BY 4.0.

As categorias educação 0/5/6 são reunidas em 4; estado civil 0 é reunido em 3. Não há imputação automática nem geração de dados sintéticos. Números ausentes ou infinitos, categorias inválidas e pagamentos negativos são rejeitados. Faturas negativas são aceitas porque podem representar crédito.

Há 56 registros que repetem um perfil de atributos. Perfis idênticos são agrupados, inclusive quando os alvos divergem. Grupos são estratificados pelo maior alvo e nunca atravessam partições.

A referência de 4.504 registros é a mesma do primeiro desenvolvimento (semente 42). Nas novas execuções, ela é excluída de todas as partições de desenvolvimento. O restante é dividido em ajuste (65%), parada antecipada (10%), calibração (10%) e validação (15%). São frações do desenvolvimento, não da base total. A tabela mostra a realização da semente 42.

| Partição | Registros | Taxa inadimplente |
| --- | --- | --- |
| fit | 16564 | 0,221 |
| stop | 2552 | 0,221 |
| calibration | 2549 | 0,221 |
| validation | 3831 | 0,221 |
| Referência já consultada | 4504 | 0,221 |

## 3. Protocolo e controle de vazamento

As sementes são 7, 21, 42, 84 e 123. Cada uma muda as partições de desenvolvimento e a aleatoriedade do treinamento. As 28 configurações usam as mesmas partições em cada semente, permitindo comparações pareadas. São cinco repetições de holdout, e não validação cruzada de cinco folds.

O scaler e o encoder são ajustados somente na partição fit. As redes PyTorch usam stop para parada antecipada. A MLP scikit-learn usa sua subdivisão interna de 15% de fit; portanto seu número efetivo de exemplos para gradiente difere do das redes PyTorch. Os calibradores usam somente calibration. A escolha dos métodos e dos limiares usa somente validation.

O protocolo foi registrado em results/research/protocol.json antes dos ajustes. Modelo da aplicação: maior AP média de validação entre regressão logística, XGBoost e MLP scikit-learn. Melhor configuração neural: maior AP média de validação entre as 18 redes PyTorch. Calibrador: menor Brier médio de validação entre none, sigmoid e isotonic. Limiar: F1 máximo na validação de cada execução.

As escolhas foram registradas antes da avaliação complementar de referência. O fato de a referência já ter sido consultada anteriormente impede chamá-la de teste inédito. A seleção de muitas configurações também pode produzir otimismo nas métricas de validação. Não há validação externa ou temporal.

## 4. Métricas e incerteza

A métrica principal de ordenação é average precision (AP), calculada pela soma dos incrementos de recall ponderados pela precisão. Ela não é idêntica à área trapezoidal da curva PR. Também são apresentados AUC-ROC, precisão, recall, F1, matriz de confusão, Brier score e log loss.

AP e AUC medem ordenação; Brier e log loss avaliam probabilidades. F1 e recall dependem do limiar. Os limiares são escolhidos na validação, antes de aplicar a regra à referência. A API usa o limiar exato, enquanto o controle visual arredonda para 0,1 ponto percentual.

Médias e desvios entre cinco sementes descrevem a variabilidade do procedimento. Os intervalos t de 95% e as diferenças pareadas estão nos CSVs; como as partições se sobrepõem, esses intervalos são descritivos e não sustentam testes de significância com observações independentes.

Na referência da semente 42, intervalos percentis de 95% usam 1.000 reamostragens de perfis de atributos, mantendo juntas as linhas de um mesmo perfil. São condicionais ao modelo ajustado: não incluem toda a incerteza de treinamento, seleção ou transferência para outra população.

## 5. P1 — comparação de modelos

Os baselines usam regressão logística com C=1; SVM linear com C=1 e calibração sigmoide na partição de calibração; kNN com k=25 nas distâncias L1 e L2; random forest com 250 árvores, profundidade máxima 10 e mínimo de 10 exemplos por folha; e XGBoost com 350 árvores, profundidade 3 e learning rate 0,04. Os parâmetros completos constam do código e dos arquivos JSON.

A melhor MLP observada alcançou AP média de 0,543, frente a 0,495 da regressão, 0,483 do kNN L1 e 0,491 do kNN L2. Nesse protocolo, as redes superaram esses baselines pela AP média. Isso não implica que redes sejam sempre superiores.

Random forest (0,543), XGBoost (0,543) e a melhor MLP (0,543) ficaram próximos. O XGBoost é o padrão da aplicação por liderar o subconjunto dos três modelos nela integrados; não foi declarado vencedor de todos os 28 candidatos.

| Modelo | AP média ± DP | ROC média | F1 médio | Recall médio |
| --- | --- | --- | --- | --- |
| Regressão logística | 0,495 ± 0,012 | 0,724 | 0,511 | 0,510 |
| SVM linear + sigmoide | 0,492 ± 0,012 | 0,720 | 0,511 | 0,497 |
| kNN L1 (k=25) | 0,483 ± 0,010 | 0,739 | 0,495 | 0,520 |
| kNN L2 (k=25) | 0,491 ± 0,010 | 0,742 | 0,507 | 0,537 |
| Random forest | 0,543 ± 0,016 | 0,773 | 0,541 | 0,562 |
| XGBoost | 0,543 ± 0,009 | 0,774 | 0,544 | 0,544 |
| MLP scikit-learn | 0,538 ± 0,011 | 0,768 | 0,536 | 0,572 |
| MLP 64/32, ReLU | 0,534 ± 0,013 | 0,769 | 0,535 | 0,557 |
| Ativação tanh | 0,543 ± 0,011 | 0,770 | 0,535 | 0,563 |

![5. P1 — comparação de modelos](results/research/models.png)

## 6. P2 — arquitetura, ativação e convergência

A configuração de controle tem camadas 64/32 e ReLU. Foram comparadas uma camada com 32 ou 128 neurônios, três camadas 128/64/32, tanh e sigmoide mantendo a arquitetura 64/32 nas mudanças de ativação. As demais condições do controle são preservadas.

As redes usam Adam, learning rate 0,001, lotes de até 512 exemplos e até 80 épocas. A parada antecipada tem paciência 10 e melhoria mínima 0,0001 na BCE não ponderada de stop. O melhor estado é restaurado. Todos os vieses iniciais são zero; o gradiente é limitado à norma 5.

Tanh foi a ativação com maior AP média entre as redes testadas (0,543). Aumentar a profundidade para três camadas não elevou a AP média sobre a rede 64/32 ReLU nesse desenho. A sigmoide exigiu em média 67,4 épocas até seu melhor estado, contra 25,6 do controle.

O gap informado é BCE de stop menos BCE de fit, medido no melhor estado; valores positivos indicam perda maior fora do ajuste. Ele não é, isoladamente, uma prova de overfitting nem resume toda a dinâmica das curvas.

| Configuração | AP média ± DP | Melhor época média | Gap BCE médio |
| --- | --- | --- | --- |
| MLP 64/32, ReLU | 0,534 ± 0,013 | 25,6 | 0,015 |
| Uma camada: 32 | 0,524 ± 0,013 | 63,2 | 0,011 |
| Uma camada: 128 | 0,527 ± 0,012 | 33,4 | 0,013 |
| Três camadas: 128/64/32 | 0,532 ± 0,014 | 14,0 | 0,015 |
| Ativação tanh | 0,543 ± 0,011 | 43,2 | 0,011 |
| Ativação sigmoide | 0,525 ± 0,021 | 67,4 | 0,005 |

![6. P2 — arquitetura, ativação e convergência](results/research/architecture.png)

## 7. P3 — regularização e balanceamento

L2 é implementado por weight_decay=0,001 no Adam; dropout usa probabilidade 0,30; batch normalization é aplicada após as camadas lineares ocultas. Há ablações individuais e uma combinação das três técnicas. O peso positivo na BCE é a razão entre negativos e positivos de fit. A sobreamostragem aleatória iguala as classes usando apenas exemplos de fit.

Dropout isolado teve AP média 0,541, frente a 0,534 do controle. A combinação de L2, dropout e batch normalization alcançou 0,535. Portanto adicionar mais regularização não trouxe uma melhoria monotônica.

O recall médio foi 0,557 no controle, 0,556 com peso de classe e 0,571 com sobreamostragem. Como o limiar foi otimizado separadamente em cada validação, esses valores refletem o efeito conjunto do treinamento e da decisão, e não apenas dos pesos.

A sobreamostragem aumenta o número de exemplos e de atualizações por época; o custo computacional por época não é o mesmo do controle. Pesos e amostragem podem distorcer probabilidades sem calibração. Os tempos e os tamanhos de ajuste após amostragem estão registrados por execução.

| Configuração | AP média ± DP | Recall médio | Gap BCE médio |
| --- | --- | --- | --- |
| MLP 64/32, ReLU | 0,534 ± 0,013 | 0,557 | 0,015 |
| Somente L2 | 0,532 ± 0,013 | 0,564 | 0,011 |
| Somente dropout | 0,541 ± 0,017 | 0,567 | 0,012 |
| Somente batch normalization | 0,531 ± 0,018 | 0,558 | 0,017 |
| L2 + dropout + batch normalization | 0,535 ± 0,015 | 0,570 | 0,013 |
| Peso de classe | 0,529 ± 0,011 | 0,556 | 0,013 |
| Sobreamostragem aleatória | 0,527 ± 0,012 | 0,571 | 0,025 |
| Regularização conjunta + peso | 0,530 ± 0,012 | 0,543 | 0,013 |

![7. P3 — regularização e balanceamento](results/research/regularization.png)

## 8. P4 — pré-processamento e inicialização

A normalização padroniza os atributos numéricos usando a média e o desvio de fit. One-hot é aplicado a sexo, escolaridade e estado civil. Sem one-hot, os códigos dessas categorias entram como valores numéricos. Sem normalização, os valores monetários permanecem em sua escala original.

A remoção da normalização reduziu a AP média de 0,534 para 0,281. Mantendo a normalização, a versão sem one-hot atingiu 0,533; a diferença para o controle foi pequena nesse conjunto de configurações.

As inicializações comparadas são a distribuição padrão das camadas lineares do PyTorch, Xavier uniforme e Kaiming normal. Os vieses permanecem zero em todos os casos. Xavier e Kaiming foram avaliadas com a rede ReLU de controle; seus resultados não justificam declarar uma inicialização universalmente melhor.

A comparação mantém a mesma taxa de aprendizado e o mesmo limite de épocas. A queda sem normalização descreve esse procedimento de otimização e não prova que qualquer rede sem normalização seja incapaz de aprender. As curvas da versão não normalizada usam escala symlog para permitir visualizar perdas muito maiores.

| Configuração | AP média ± DP | ROC média | Melhor época média |
| --- | --- | --- | --- |
| MLP 64/32, ReLU | 0,534 ± 0,013 | 0,769 | 25,6 |
| Sem normalização | 0,281 ± 0,020 | 0,618 | 21,0 |
| Sem one-hot | 0,533 ± 0,015 | 0,768 | 29,8 |
| Sem normalização e sem one-hot | 0,288 ± 0,015 | 0,628 | 18,6 |
| Inicialização Xavier | 0,532 ± 0,015 | 0,764 | 23,0 |
| Inicialização Kaiming | 0,529 ± 0,008 | 0,765 | 24,0 |

![8. P4 — pré-processamento e inicialização](results/research/preprocessing.png)

## 9. Curvas de aprendizagem

A figura mostra quatro configurações na semente 42: controle, regularização conjunta, peso de classe e ausência de normalização. As perdas são BCE não ponderada calculada em modo de avaliação, de modo que dropout fica desligado e batch normalization usa as estatísticas aprendidas. O objetivo usado durante o gradiente pode ser ponderado, mas a curva relatada usa a mesma definição de perda.

As curvas completas de todas as 90 redes estão em learning_curves.csv; melhores épocas e gaps estão em convergence.csv. Não são curvas calculadas sobre a referência.

![9. Curvas de aprendizagem](results/research/learning_curves.png)

## 10. Calibração de probabilidades

Foram comparados: none, que mantém a probabilidade original; sigmoid, regressão logística ajustada sobre o logit da probabilidade original com C=1.000.000; e isotonic, regressão isotônica monotônica. O calibrador recebe apenas as previsões e os alvos de calibration. Seu desempenho é comparado em validation.

A escolha usa o menor Brier médio nas cinco sementes, e não a aparência da curva ou a referência. Também são relatados log loss e ECE com dez intervalos uniformes. O diagrama de confiabilidade usa dez grupos por quantis; por isso seus agrupamentos diferem dos usados no ECE.

A regressão logística escolheu isotonic, a MLP scikit-learn escolheu sigmoid e o XGBoost manteve none. A isotônica reduziu o Brier médio da regressão, mas aumentou seu log loss médio: os critérios não são equivalentes. Probabilidades isotônicas nos extremos podem receber penalidade alta quando erram.

Os três modelos da aplicação foram atualizados com os métodos escolhidos. A melhor rede PyTorch permanece como artefato experimental separado; a MLP da aplicação continua sendo a implementação scikit-learn, agora calibrada.

| Modelo | Método | Brier médio ± DP | Log loss médio | ECE médio |
| --- | --- | --- | --- | --- |
| Regressão logística | none | 0,146 ± 0,002 | 0,467 | 0,045 |
| Regressão logística | sigmoid | 0,147 ± 0,001 | 0,467 | 0,049 |
| Regressão logística | isotonic (escolhida) | 0,144 ± 0,002 | 0,481 | 0,019 |
| XGBoost | none (escolhida) | 0,136 ± 0,001 | 0,434 | 0,014 |
| XGBoost | sigmoid | 0,137 ± 0,001 | 0,434 | 0,016 |
| XGBoost | isotonic | 0,138 ± 0,002 | 0,448 | 0,023 |
| Rede neural MLP | none | 0,138 ± 0,002 | 0,440 | 0,019 |
| Rede neural MLP | sigmoid (escolhida) | 0,138 ± 0,002 | 0,438 | 0,015 |
| Rede neural MLP | isotonic | 0,139 ± 0,002 | 0,450 | 0,017 |

![10. Calibração de probabilidades](results/research/calibration_comparison.png)

## 11. Confiabilidade na referência já consultada

Os pontos mostram a fração observada de inadimplentes contra a probabilidade média de cada grupo, para os modelos finais da semente 42. A diagonal representa coincidência entre as duas medidas. Oscilações, sobretudo em grupos pequenos e extremos, não devem ser interpretadas como estabilidade fora desta base.

Esses diagramas são descritivos e não foram usados para refazer a escolha dos calibradores. A base de referência já havia sido observada em uma etapa anterior.

![11. Confiabilidade na referência já consultada](results/research/reliability.png)

## 12. Avaliação complementar de referência

Após salvar as escolhas, os modelos foram aplicados aos mesmos 4.504 registros da referência original. A tabela mostra a realização da semente 42. AP, AUC e limiares podem diferir do primeiro desenvolvimento, pois agora parte do desenvolvimento é reservada para parada antecipada e calibração.

Os intervalos abaixo são bootstrap por perfil, condicionais ao modelo ajustado. A avaliação inclui os baselines de P1 e a melhor rede selecionada na validação. Não foi feita nova seleção usando esses resultados.

O XGBoost final teve AP 0,529, AUC-ROC 0,768 e recall 0,512. Sua matriz registra 3.075 adimplentes corretamente identificados, 434 falsos positivos, 486 inadimplentes não detectados e 509 verdadeiros positivos. O desempenho continua deixando uma parcela relevante de inadimplentes sem identificação.

| Modelo | AP | IC 95% AP | ROC | F1 | Recall |
| --- | --- | --- | --- | --- | --- |
| Regressão logística | 0,468 | 0,434–0,502 | 0,712 | 0,492 | 0,452 |
| XGBoost | 0,529 | 0,496–0,566 | 0,768 | 0,525 | 0,512 |
| Rede neural MLP | 0,517 | 0,483–0,552 | 0,756 | 0,511 | 0,578 |
| kNN L1 | 0,479 | 0,444–0,513 | 0,737 | 0,486 | 0,480 |
| kNN L2 | 0,475 | 0,441–0,508 | 0,730 | 0,490 | 0,490 |
| SVM linear | 0,485 | 0,451–0,522 | 0,708 | 0,481 | 0,425 |
| Random forest | 0,534 | 0,500–0,569 | 0,764 | 0,526 | 0,555 |
| MLP PyTorch: tanh | 0,524 | 0,490–0,558 | 0,758 | 0,521 | 0,588 |

## 13. Atributos sensíveis e diferenças por grupo

Foram repetidos os três modelos da aplicação removendo, em conjunto, sexo, idade e estado civil do pré-processamento. As comparações usam as mesmas partições e os mesmos hiperparâmetros; a tabela mostra a AP média de validação antes de calibração adicional.

Uma alteração pequena de AP após excluir atributos sensíveis não comprova justiça. Outras variáveis podem funcionar como proxies. Também não se pode concluir causalidade ou adequação jurídica a partir dessas métricas.

A análise descritiva por sexo, estado civil e faixa etária apresenta tamanho do grupo, prevalência, recall e taxa de falsos positivos na referência. Ela não define um critério normativo de equidade nem inclui uma validação independente de discriminação.

| Modelo | AP completa | AP sem sensíveis | Diferença |
| --- | --- | --- | --- |
| Regressão logística | 0,495 | 0,494 | -0,001 |
| XGBoost | 0,543 | 0,540 | -0,003 |
| Rede neural MLP | 0,538 | 0,519 | -0,019 |

## 13.1. Grupos no modelo padrão

Codificações: SEX 1 masculino, 2 feminino; MARRIAGE 1 casado, 2 solteiro, 3 outro. AGE_GROUP indica faixas de idade. As métricas usam o limiar validado do XGBoost. Grupos pequenos têm incerteza maior; os valores abaixo não recebem interpretação causal.

| Atributo / grupo | n | Prevalência | Recall | Falsos positivos |
| --- | --- | --- | --- | --- |
| SEX / 1 | 1761 | 0,229 | 0,505 | 0,150 |
| SEX / 2 | 2743 | 0,215 | 0,516 | 0,107 |
| MARRIAGE / 1 | 2136 | 0,228 | 0,539 | 0,138 |
| MARRIAGE / 2 | 2322 | 0,214 | 0,485 | 0,111 |
| MARRIAGE / 3 | 46 | 0,261 | 0,500 | 0,088 |
| AGE_GROUP / 18–29 | 1421 | 0,222 | 0,521 | 0,127 |
| AGE_GROUP / 30–44 | 2304 | 0,215 | 0,486 | 0,117 |
| AGE_GROUP / 45–59 | 725 | 0,230 | 0,557 | 0,136 |
| AGE_GROUP / 60–100 | 54 | 0,315 | 0,647 | 0,162 |

## 14. Cenários de custo e limiares

Foi realizada uma análise de sensibilidade ilustrativa: custo de falso positivo fixado em 1 e custo de falso negativo em 1, 2, 5 ou 10. Para cada cenário, o limiar que minimiza o custo é escolhido na validação e aplicado à referência.

Os custos por mil clientes são unidades hipotéticas, não reais, NT$ ou estimativas de perda financeira. O dataset não traz custos de concessão, cobrança e recuperação suficientes para definir uma política comercial. O limiar padrão do aplicativo continua sendo o de F1 validado; estes cenários são uma análise separada.

| Custo FN / FP | Limiar validação | FN referência | FP referência | Custo / mil |
| --- | --- | --- | --- | --- |
| 1 / 1 | 0,396 | 572 | 261 | 184,9 |
| 2 / 1 | 0,299 | 486 | 434 | 312,2 |
| 5 / 1 | 0,192 | 327 | 981 | 580,8 |
| 10 / 1 | 0,083 | 55 | 2610 | 701,6 |

## 15. Integração, testes e reprodutibilidade

A aplicação oferece regressão logística, XGBoost e MLP scikit-learn com os calibradores selecionados. Os fatores são sensibilidade ao substituir grupos pela mediana/moda de fit: não são explicações causais, SHAP nem parcelas aditivas. Indisponibilidade do artefato retorna HTTP 503; não há regra manual substituta.

O código inclui testes de entradas inválidas, calibração sem novo ajuste do modelo base, equivalência entre API e artefato, partições sem vazamento, índices reproduzíveis, treino neural determinístico em CPU e falhas de carregamento. O navegador é verificado em desktop e celular, com troca de modelo, limiar, rede indisponível e cancelamento de resposta desatualizada.

Comandos: python main.py --serve para a aplicação; python research.py --integrate --resume para repetir ou retomar a pesquisa; python build_report.py para reconstruir os PDFs; python -m pytest -q para os testes. Dependências de pesquisa e de geração dos documentos são opcionais ao uso da aplicação e estão separadas das dependências de inferência.

Os artefatos incluem protocolo, resultados por execução, médias e desvios, diferenças pareadas, predições de validação e referência, curvas por época, configurações, pesos da melhor rede e checksums. As versões do ambiente e o hash dos dados permitem verificar a origem dos resultados.

## 16. Limitações e conclusões

P1: as redes testadas superaram kNN e classificadores lineares pela AP média observada; as melhores redes e os modelos de árvores ficaram próximos. P2: tanh teve o melhor resultado entre as ativações estudadas; profundidade maior não assegurou melhora. P3: dropout ajudou nesse desenho, enquanto a combinação de técnicas e o balanceamento não trouxeram ganhos uniformes. P4: a normalização teve efeito muito maior do que retirar one-hot ou trocar a inicialização nas condições testadas.

Calibrar não é necessariamente melhorar todas as métricas. Os métodos finais foram escolhidos pelo Brier de validação, preservando a opção de não ajustar o XGBoost. A escolha por Brier pode divergir de log loss e de métricas de ordenação.

Os cinco resultados por configuração não são cinco amostras independentes da população. A busca é limitada a uma grade pequena e a uma taxa de aprendizado. Não houve busca exaustiva, teste externo recente ou validação temporal. O conjunto de referência já tinha sido consultado e seus intervalos são condicionais aos modelos.

A base de um banco de Taiwan em 2005 não representa automaticamente crédito brasileiro atual. Para implantação operacional seriam necessários dados recentes e autorizados, custos reais, análise de governança e validação de equidade. Essas são limitações de aplicação e generalização, e não resultados demonstrados pelo experimento.

## Referências

YEH, I-Cheng; LIEN, Che-hui. Default of Credit Card Clients. UCI Machine Learning Repository, 2009. DOI: 10.24432/C55S3H. Licença CC BY 4.0.

YEH, I-Cheng; LIEN, Che-hui. The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, v. 36, n. 2, 2009.

CHEN, Tianqi; GUESTRIN, Carlos. XGBoost: A Scalable Tree Boosting System. KDD, 2016. DOI: 10.1145/2939672.2939785.

SRIVASTAVA, Nitish et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, v. 15, 2014.

IOFFE, Sergey; SZEGEDY, Christian. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML, 2015.

GUO, Chuan et al. On Calibration of Modern Neural Networks. ICML, 2017.

KINGMA, Diederik P.; BA, Jimmy. Adam: A Method for Stochastic Optimization. ICLR, 2015.

PASZKE, Adam et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS, 2019.

## Apêndice — versões e leitura dos arquivos

Versões: torch 2.10.0+cpu; scikit-learn 1.8.0; numpy 2.3.5; pandas 2.2.3; xgboost 3.4.1; scipy 1.17.0; Python 3.12.14.

experiments.csv: 140 ajustes e métricas de validação. summary.csv: média, desvio e intervalos descritivos. paired_differences.csv: diferenças contra a rede de controle. calibration.csv: 45 avaliações de métodos de calibração. calibration_reference.csv: avaliação complementar dos nove pares modelo/método. final_evaluation.json: escolhas, métricas e intervalos bootstrap. cost_scenarios.csv: cenários de custo. split_seed_*.npz: índices completos de cada semente.

O protocolo registrado usa máximo de 80 épocas, paciência 10 e sementes 7/21/42/84/123. Os 140 ajustes incluem 7 baselines, 18 configurações PyTorch e 3 ablações de atributos sensíveis por semente. As 45 avaliações de calibração usam predições desses modelos, sem novo treinamento dos classificadores.

Os documentos originais recebidos estão em docs/originais. Este relatório descreve experimentos executados e não reproduz como fatos as hipóteses das propostas anteriores.
