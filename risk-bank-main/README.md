# Risk Bank

Aplicativo web para estimar o risco de um cliente de cartão de crédito deixar de pagar a fatura no mês seguinte.

Projeto da disciplina de Inteligência Artificial (Ciência da computação, UNIMA), desenvolvido por Athos Andre, Lucas Brasileiro, Guilherme Uchoa, Matheus Albuquerque e Dhigor Padilha.

> **Aviso:** este é um estudo acadêmico. Os resultados não devem ser usados para decidir crédito real sem validação adicional.

## Sobre

Instituições financeiras concedem crédito sem saber de antemão se o cliente conseguirá pagar. Quando isso não acontece, ocorre a inadimplência, que gera prejuízo e influencia limites de crédito e políticas de cobrança.

O Risk Bank recebe o histórico de um cliente e devolve a **probabilidade de inadimplência no mês seguinte**, junto com uma classificação (provável bom pagador ou provável inadimplente) e os fatores que mais pesaram no resultado.

## Funcionalidades

- Formulário com o perfil do cliente e o histórico dos últimos seis meses.
- Medidor com a probabilidade de inadimplência e o veredito.
- Lista dos três fatores que mais influenciaram o resultado.
- Gráfico de faturas e pagamentos por mês, atualizado enquanto você digita.
- Controle do limite de classificação, para escolher o quanto o app deve ser rigoroso.
- Histórico das análises recentes da sessão.
- Dois exemplos prontos (bom pagador e risco alto) para testar rapidamente.

## Como usar

O frontend é um único arquivo, sem instalação nem build.

1. Abra o `index.html` no navegador.
2. Preencha o perfil do cliente (limite, idade, sexo, escolaridade e estado civil).
3. Informe, para cada mês de abril a setembro, a situação de pagamento, o valor da fatura e o valor pago.
4. Clique em **Analisar cliente**.
5. Leia o resultado no painel da direita. Ajuste o limite de classificação se quiser um critério mais ou menos rigoroso.

Para ver um exemplo, use os botões "Exemplo: bom pagador" e "Exemplo: risco alto".

### Situação de pagamento

| Valor | Significado |
|---|---|
| Sem consumo | O cliente não usou o cartão no mês |
| Pago em dia | A fatura foi quitada |
| Uso rotativo | Pagou parte da fatura e manteve saldo |
| 1 a 8+ meses | Meses de atraso no pagamento |

### Limite de classificação

O app transforma a probabilidade em uma decisão comparando-a com um limite (padrão de 35%).

- **Limite menor:** encontra mais clientes inadimplentes (recall maior), mas gera mais alarmes falsos.
- **Limite maior:** reduz os alarmes falsos, mas deixa passar mais inadimplentes.

## Como funciona

O app segue o desenho de um problema de classificação binária supervisionada: a partir das 23 variáveis do histórico do cliente, o modelo estima a chance da classe "inadimplente".

### Variáveis de entrada

| Grupo | Descrição | Quantidade |
|---|---|---|
| Limite de crédito | Valor do crédito concedido | 1 |
| Perfil | Sexo, escolaridade, estado civil e idade | 4 |
| Histórico de pagamento | Situação do pagamento de abril a setembro de 2005 | 6 |
| Valores das faturas | Valor da fatura em cada um dos seis meses | 6 |
| Valores pagos | Valor pago em cada um dos seis meses | 6 |

### Modelos

O estudo compara modelos de complexidade crescente:

| Modelo | Papel |
|---|---|
| kNN (distâncias L1 e L2) | Baseline sem aprendizado |
| Classificador linear (softmax e SVM loss) | Baseline linear |
| MLP com 1 camada escondida | Primeira não linearidade |
| MLP profundo com L2, dropout e batch normalization | Modelo de maior capacidade |
| Random forest (opcional) | Referência externa para dados tabulares |

### Avaliação

Como apenas cerca de 22% dos clientes são inadimplentes, um modelo que classificasse todos como bons pagadores já teria cerca de 78% de acurácia sem aprender nada. Por isso a avaliação usa **recall, precisão, F1, AUC-ROC e matriz de confusão**, e a acurácia serve só de referência.

### Estado atual da previsão

A versão atual da interface usa uma **regra simples escrita à mão** no lugar do modelo treinado, e a tela exibe um aviso de "Modo demonstração". A regra considera os atrasos de pagamento (com mais peso nos meses recentes), o tamanho da fatura em relação ao limite, a parcela paga da fatura anterior e o limite de crédito.

O resultado, portanto, serve para demonstrar a interface e não deve ser tomado como previsão do modelo.

## Integração com o modelo

A previsão fica isolada na função `predict(x)` do `index.html`. Para conectar um modelo treinado, basta substituí-la por uma chamada à API.

**Requisição:** `POST /predict`

```json
{
  "limit": 50000, "age": 35, "sex": 1, "edu": 2, "mar": 2,
  "pay":  [0, 0, 0, 0, 0, 0],
  "bill": [20000, 20000, 20000, 20000, 20000, 20000],
  "paid": [2000, 2000, 2000, 2000, 2000, 2000]
}
```

**Resposta:**

```json
{
  "prob": 0.31,
  "factors": [{ "name": "Atrasos no pagamento", "value": 0.4 }]
}
```

- Os arrays `pay`, `bill` e `paid` seguem a ordem do dataset: índice 0 é setembro (mais recente) e índice 5 é abril (mais antigo).
- `prob` é a probabilidade de inadimplência, entre 0 e 1.
- Em `factors`, valores positivos aumentam o risco e negativos reduzem.
- Codificação das categorias: sexo (1 masculino, 2 feminino), escolaridade (1 pós-graduação, 2 universidade, 3 ensino médio, 4 outro) e estado civil (1 casado, 2 solteiro, 3 outro).

Mais detalhes para quem for desenvolver estão no [`CONTEXT.md`](CONTEXT.md).

## Dataset

*Default of Credit Card Clients*, do UCI Machine Learning Repository, criado por I-Cheng Yeh e Che-hui Lien (2009).

- 30.000 clientes de um banco de Taiwan, com dados de abril a setembro de 2005.
- 23 variáveis explicativas, sem valores faltantes segundo a documentação.
- Alvo binário: inadimplente no mês seguinte (1) ou não (0).
- Cerca de 22,1% dos clientes são inadimplentes.
- Licença: Creative Commons Attribution 4.0 (CC BY 4.0).

## Limitações e ética

- Os dados são de 2005 e de um único banco de Taiwan, então os resultados não se generalizam automaticamente para outros países ou épocas.
- O dataset inclui sexo, idade e estado civil, o que traz risco de viés e de discriminação. Por isso o desempenho deve ser analisado por grupo.
- O app é um estudo acadêmico e não substitui análise de crédito real.

## Referências

- ABU-MOSTAFA, Y. S.; MAGDON-ISMAIL, M.; LIN, H.-T. *Learning From Data*. AMLBook, 2012.
- IOFFE, S.; SZEGEDY, C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. In: ICML, 2015.
- SRIVASTAVA, N. et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *Journal of Machine Learning Research*, v. 15, p. 1929-1958, 2014.
- YEH, I.-C.; LIEN, C.-H. The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, v. 36, n. 2, p. 2473-2480, 2009.
- YEH, I. *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository, 2009. https://doi.org/10.24432/C55S3H.
- ZANCHETTIN, C. Revisão ML + NN. Material de aula, Centro de Informática, UFPE.
