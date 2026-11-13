"""Gera o relatório, a síntese e a apresentação a partir dos resultados executados."""
from pathlib import Path
import json
import re
import shutil
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from data import BASE_DIR
from train import MODEL_NAMES

ROOT = BASE_DIR
R = ROOT / 'results/research'
DOCS = ROOT / 'docs'
INK = colors.HexColor('#14333b')
TEAL = colors.HexColor('#286373')
MUTED = colors.HexColor('#52636a')
FONT_DIRS = [ROOT/'assets/fonts', Path('/usr/share/fonts/truetype/dejavu')]


def register_fonts():
    for directory in FONT_DIRS:
        regular, bold = directory/'DejaVuSans.ttf', directory/'DejaVuSans-Bold.ttf'
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont('RiskSans', str(regular)))
            pdfmetrics.registerFont(TTFont('RiskSansBold', str(bold)))
            pdfmetrics.registerFontFamily('RiskSans', normal='RiskSans', bold='RiskSansBold', italic='RiskSans', boldItalic='RiskSansBold')
            return
    raise FileNotFoundError('As fontes DejaVuSans precisam estar em assets/fonts/.')


def number(value, digits=3):
    return f'{float(value):.{digits}f}'.replace('.', ',')


def mean_sd(row, metric):
    return number(row[metric+'_mean']) + ' ± ' + number(row[metric+'_sd'])


def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(value).replace('|','/') for value in row) + ' |' for row in rows])


def styles(slides=False):
    sheet = getSampleStyleSheet()
    size = 13 if slides else 9.2
    sheet.add(ParagraphStyle(name='BodyRisk', fontName='RiskSans', fontSize=size, leading=size*1.5,
                            textColor=INK, spaceAfter=9 if slides else 7))
    sheet.add(ParagraphStyle(name='TitleRisk', fontName='RiskSansBold', fontSize=28 if slides else 27,
                            leading=36, textColor=INK, spaceAfter=20))
    sheet.add(ParagraphStyle(name='HeadingRisk', fontName='RiskSansBold', fontSize=23 if slides else 15,
                            leading=30 if slides else 21, textColor=TEAL, spaceBefore=10, spaceAfter=12))
    sheet.add(ParagraphStyle(name='CellRisk', fontName='RiskSans', fontSize=10 if slides else 7.8,
                            leading=14 if slides else 11, textColor=INK))
    sheet.add(ParagraphStyle(name='CellHeadRisk', parent=sheet['CellRisk'], fontName='RiskSansBold', textColor=colors.white))
    sheet.add(ParagraphStyle(name='CaptionRisk', fontName='RiskSans', fontSize=9 if slides else 7.5,
                            leading=12 if slides else 10.5, textColor=MUTED, spaceAfter=9))
    return sheet


def para(text, style):
    return Paragraph(escape(str(text)).replace('\n','<br/>'), style)


def table(headers, rows, width, sheet, proportions=None):
    if proportions is None:
        proportions = [0.37] + [0.63/(len(headers)-1)]*(len(headers)-1)
    content = [[para(value, sheet['CellHeadRisk']) for value in headers]] + [
        [para(value, sheet['CellRisk']) for value in row] for row in rows]
    result = Table(content, colWidths=[width*p for p in proportions], repeatRows=1, hAlign='LEFT')
    result.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),TEAL), ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f5f5'),colors.white]),
        ('LINEBELOW',(0,0),(-1,0),0.6,TEAL), ('LINEBELOW',(0,1),(-1,-1),0.2,colors.HexColor('#d6e1e3')),
        ('LEFTPADDING',(0,0),(-1,-1),6), ('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),6), ('BOTTOMPADDING',(0,0),(-1,-1),6),
    ]))
    return result


def figure(path, width, max_height=300):
    picture = Image(str(path))
    ratio = min(width / picture.imageWidth, max_height / picture.imageHeight)
    picture.drawWidth, picture.drawHeight = picture.imageWidth*ratio, picture.imageHeight*ratio
    picture.hAlign = 'CENTER'
    return picture


def footer(canvas, document):
    canvas.saveState()
    width, height = document.pagesize
    canvas.setStrokeColor(colors.HexColor('#d6e1e3'))
    canvas.line(40,32,width-40,32)
    canvas.setFont('RiskSans',7)
    canvas.setFillColor(MUTED)
    canvas.drawString(40,21,'Risk Bank · resultados experimentais · base UCI')
    canvas.drawRightString(width-40,21,str(document.page))
    canvas.restoreState()


def load_sections():
    summary = pd.read_csv(R/'summary.csv').set_index('model')
    experiments = pd.read_csv(R/'experiments.csv')
    calibrations = pd.read_csv(R/'calibration_summary.csv')
    raw_calibrations = pd.read_csv(R/'calibration.csv')
    final = json.loads((R/'final_evaluation.json').read_text())
    protocol = json.loads((R/'protocol.json').read_text())
    decisions = final['decisions']
    config = {row['key']:row for row in protocol['neural_configurations']}
    best = decisions['best_neural_configuration']
    selected = decisions['selected_application_model']
    convergence = pd.read_csv(R/'convergence.csv').groupby('model').mean(numeric_only=True)
    sections = []
    def add(title, paragraphs=(), headers=None, rows=None, image=None, caption=None, proportions=None):
        sections.append({'title':title,'paragraphs':list(paragraphs),'headers':headers,'rows':rows,
                         'image':image,'caption':caption,'proportions':proportions})
    add('Resumo',[
        'Este estudo compara classificadores de inadimplência de cartão de crédito e investiga o efeito de arquitetura, ativação, regularização, balanceamento e preparação dos dados em redes neurais. Foram executados 140 ajustes: 28 configurações em cinco sementes, usando os 30.000 registros oficiais do dataset Default of Credit Card Clients.',
        f"Entre as redes PyTorch avaliadas, {summary.loc[best,'label']} obteve a maior AP média de validação ({mean_sd(summary.loc[best],'average_precision')}). O XGBoost alcançou {mean_sd(summary.loc['xgboost'],'average_precision')} e permaneceu como padrão entre os três modelos integrados à aplicação. As diferenças pequenas entre os melhores candidatos não demonstram superioridade geral de um algoritmo.",
        f"A calibração selecionada pelo Brier médio de validação foi: regressão logística, {decisions['calibration_methods']['logistic']}; XGBoost, {decisions['calibration_methods']['xgboost']}; MLP scikit-learn, {decisions['calibration_methods']['mlp']}. Os resultados de referência são complementares: o conjunto já havia sido consultado na etapa anterior e não é apresentado como teste inédito.",
        'Palavras-chave: inadimplência; classificação binária; redes neurais; XGBoost; calibração de probabilidades.'
    ])
    add('1. Problema e perguntas de pesquisa',[
        'O objetivo é estimar se um cliente deixará de pagar no mês seguinte a partir do limite, perfil e histórico de seis meses. Como cerca de 22,1% dos registros pertencem à classe inadimplente, acurácia isolada é insuficiente.',
        'P1: uma MLP supera kNN e classificadores lineares? P2: como largura, profundidade e ativação afetam o desempenho e a convergência? P3: quais efeitos são observados com L2, dropout, batch normalization e balanceamento? P4: como normalização, one-hot e inicialização mudam os resultados?',
        'A regressão logística é o modelo de regressão apropriado ao alvo binário. XGBoost e random forest são referências adicionais para dados tabulares. kNN é um método supervisionado baseado em instâncias; sua etapa de ajuste armazena os exemplos, e não aprende uma parametrização como a rede neural.'
    ])
    split_rows = [[name,str(value['rows']),number(value['default_rate'])] for name,value in final['splits_seed_42'].items()]
    split_rows.append(['Referência já consultada','4504','0,221'])
    add('2. Dados, limpeza e partições',[
        'A base reúne 30.000 clientes de um banco de Taiwan em 2005, com 6.636 inadimplentes, 23 atributos explicativos e um alvo binário. O ID é excluído dos modelos. A cópia incluída preserva a planilha oficial; origem e hashes estão em data/source.json. Licença: CC BY 4.0.',
        'As categorias educação 0/5/6 são reunidas em 4; estado civil 0 é reunido em 3. Não há imputação automática nem geração de dados sintéticos. Números ausentes ou infinitos, categorias inválidas e pagamentos negativos são rejeitados. Faturas negativas são aceitas porque podem representar crédito.',
        'Há 56 registros que repetem um perfil de atributos. Perfis idênticos são agrupados, inclusive quando os alvos divergem. Grupos são estratificados pelo maior alvo e nunca atravessam partições.',
        'A referência de 4.504 registros é a mesma do primeiro desenvolvimento (semente 42). Nas novas execuções, ela é excluída de todas as partições de desenvolvimento. O restante é dividido em ajuste (65%), parada antecipada (10%), calibração (10%) e validação (15%). São frações do desenvolvimento, não da base total. A tabela mostra a realização da semente 42.'
    ],['Partição','Registros','Taxa inadimplente'],split_rows)
    add('3. Protocolo e controle de vazamento',[
        'As sementes são 7, 21, 42, 84 e 123. Cada uma muda as partições de desenvolvimento e a aleatoriedade do treinamento. As 28 configurações usam as mesmas partições em cada semente, permitindo comparações pareadas. São cinco repetições de holdout, e não validação cruzada de cinco folds.',
        'O scaler e o encoder são ajustados somente na partição fit. As redes PyTorch usam stop para parada antecipada. A MLP scikit-learn usa sua subdivisão interna de 15% de fit; portanto seu número efetivo de exemplos para gradiente difere do das redes PyTorch. Os calibradores usam somente calibration. A escolha dos métodos e dos limiares usa somente validation.',
        'O protocolo foi registrado em results/research/protocol.json antes dos ajustes. Modelo da aplicação: maior AP média de validação entre regressão logística, XGBoost e MLP scikit-learn. Melhor configuração neural: maior AP média de validação entre as 18 redes PyTorch. Calibrador: menor Brier médio de validação entre none, sigmoid e isotonic. Limiar: F1 máximo na validação de cada execução.',
        'As escolhas foram registradas antes da avaliação complementar de referência. O fato de a referência já ter sido consultada anteriormente impede chamá-la de teste inédito. A seleção de muitas configurações também pode produzir otimismo nas métricas de validação. Não há validação externa ou temporal.'
    ])
    add('4. Métricas e incerteza',[
        'A métrica principal de ordenação é average precision (AP), calculada pela soma dos incrementos de recall ponderados pela precisão. Ela não é idêntica à área trapezoidal da curva PR. Também são apresentados AUC-ROC, precisão, recall, F1, matriz de confusão, Brier score e log loss.',
        'AP e AUC medem ordenação; Brier e log loss avaliam probabilidades. F1 e recall dependem do limiar. Os limiares são escolhidos na validação, antes de aplicar a regra à referência. A API usa o limiar exato, enquanto o controle visual arredonda para 0,1 ponto percentual.',
        'Médias e desvios entre cinco sementes descrevem a variabilidade do procedimento. Os intervalos t de 95% e as diferenças pareadas estão nos CSVs; como as partições se sobrepõem, esses intervalos são descritivos e não sustentam testes de significância com observações independentes.',
        'Na referência da semente 42, intervalos percentis de 95% usam 1.000 reamostragens de perfis de atributos, mantendo juntas as linhas de um mesmo perfil. São condicionais ao modelo ajustado: não incluem toda a incerteza de treinamento, seleção ou transferência para outra população.'
    ])
    baseline_keys=['logistic','svm','knn_l1','knn_l2','random_forest','xgboost','mlp','nn_base',best]
    p1_rows=[[summary.loc[key,'label'],mean_sd(summary.loc[key],'average_precision'),number(summary.loc[key,'roc_auc_mean']),number(summary.loc[key,'f1_mean']),number(summary.loc[key,'recall_mean'])] for key in baseline_keys]
    add('5. P1 — comparação de modelos',[
        'Os baselines usam regressão logística com C=1; SVM linear com C=1 e calibração sigmoide na partição de calibração; kNN com k=25 nas distâncias L1 e L2; random forest com 250 árvores, profundidade máxima 10 e mínimo de 10 exemplos por folha; e XGBoost com 350 árvores, profundidade 3 e learning rate 0,04. Os parâmetros completos constam do código e dos arquivos JSON.',
        f"A melhor MLP observada alcançou AP média de {number(summary.loc[best,'average_precision_mean'])}, frente a {number(summary.loc['logistic','average_precision_mean'])} da regressão, {number(summary.loc['knn_l1','average_precision_mean'])} do kNN L1 e {number(summary.loc['knn_l2','average_precision_mean'])} do kNN L2. Nesse protocolo, as redes superaram esses baselines pela AP média. Isso não implica que redes sejam sempre superiores.",
        f"Random forest ({number(summary.loc['random_forest','average_precision_mean'])}), XGBoost ({number(summary.loc['xgboost','average_precision_mean'])}) e a melhor MLP ({number(summary.loc[best,'average_precision_mean'])}) ficaram próximos. O XGBoost é o padrão da aplicação por liderar o subconjunto dos três modelos nela integrados; não foi declarado vencedor de todos os 28 candidatos."
    ],['Modelo','AP média ± DP','ROC média','F1 médio','Recall médio'],p1_rows,
        image='models.png',caption='Resultados de validação; barras representam média e desvio entre sementes.',proportions=[.34,.22,.14,.14,.16])
    p2_keys=['nn_base','nn_shallow_32','nn_shallow_128','nn_deep','nn_tanh','nn_sigmoid']
    p2_rows=[[summary.loc[key,'label'],mean_sd(summary.loc[key],'average_precision'),number(convergence.loc[key,'best_epoch'],1),number(convergence.loc[key,'gap'])] for key in p2_keys]
    add('6. P2 — arquitetura, ativação e convergência',[
        'A configuração de controle tem camadas 64/32 e ReLU. Foram comparadas uma camada com 32 ou 128 neurônios, três camadas 128/64/32, tanh e sigmoide mantendo a arquitetura 64/32 nas mudanças de ativação. As demais condições do controle são preservadas.',
        'As redes usam Adam, learning rate 0,001, lotes de até 512 exemplos e até 80 épocas. A parada antecipada tem paciência 10 e melhoria mínima 0,0001 na BCE não ponderada de stop. O melhor estado é restaurado. Todos os vieses iniciais são zero; o gradiente é limitado à norma 5.',
        f"Tanh foi a ativação com maior AP média entre as redes testadas ({number(summary.loc['nn_tanh','average_precision_mean'])}). Aumentar a profundidade para três camadas não elevou a AP média sobre a rede 64/32 ReLU nesse desenho. A sigmoide exigiu em média {number(convergence.loc['nn_sigmoid','best_epoch'],1)} épocas até seu melhor estado, contra {number(convergence.loc['nn_base','best_epoch'],1)} do controle.",
        'O gap informado é BCE de stop menos BCE de fit, medido no melhor estado; valores positivos indicam perda maior fora do ajuste. Ele não é, isoladamente, uma prova de overfitting nem resume toda a dinâmica das curvas.'
    ],['Configuração','AP média ± DP','Melhor época média','Gap BCE médio'],p2_rows,image='architecture.png')
    p3_keys=['nn_base','nn_l2','nn_dropout','nn_batchnorm','nn_regularized','nn_weighted','nn_oversampling','nn_regularized_weighted']
    p3_rows=[[summary.loc[key,'label'],mean_sd(summary.loc[key],'average_precision'),number(summary.loc[key,'recall_mean']),number(convergence.loc[key,'gap'])] for key in p3_keys]
    add('7. P3 — regularização e balanceamento',[
        'L2 é implementado por weight_decay=0,001 no Adam; dropout usa probabilidade 0,30; batch normalization é aplicada após as camadas lineares ocultas. Há ablações individuais e uma combinação das três técnicas. O peso positivo na BCE é a razão entre negativos e positivos de fit. A sobreamostragem aleatória iguala as classes usando apenas exemplos de fit.',
        f"Dropout isolado teve AP média {number(summary.loc['nn_dropout','average_precision_mean'])}, frente a {number(summary.loc['nn_base','average_precision_mean'])} do controle. A combinação de L2, dropout e batch normalization alcançou {number(summary.loc['nn_regularized','average_precision_mean'])}. Portanto adicionar mais regularização não trouxe uma melhoria monotônica.",
        f"O recall médio foi {number(summary.loc['nn_base','recall_mean'])} no controle, {number(summary.loc['nn_weighted','recall_mean'])} com peso de classe e {number(summary.loc['nn_oversampling','recall_mean'])} com sobreamostragem. Como o limiar foi otimizado separadamente em cada validação, esses valores refletem o efeito conjunto do treinamento e da decisão, e não apenas dos pesos.",
        'A sobreamostragem aumenta o número de exemplos e de atualizações por época; o custo computacional por época não é o mesmo do controle. Pesos e amostragem podem distorcer probabilidades sem calibração. Os tempos e os tamanhos de ajuste após amostragem estão registrados por execução.'
    ],['Configuração','AP média ± DP','Recall médio','Gap BCE médio'],p3_rows,image='regularization.png')
    p4_keys=['nn_base','nn_unscaled','nn_ordinal','nn_unscaled_ordinal','nn_xavier','nn_kaiming']
    p4_rows=[[summary.loc[key,'label'],mean_sd(summary.loc[key],'average_precision'),number(summary.loc[key,'roc_auc_mean']),number(convergence.loc[key,'best_epoch'],1)] for key in p4_keys]
    add('8. P4 — pré-processamento e inicialização',[
        'A normalização padroniza os atributos numéricos usando a média e o desvio de fit. One-hot é aplicado a sexo, escolaridade e estado civil. Sem one-hot, os códigos dessas categorias entram como valores numéricos. Sem normalização, os valores monetários permanecem em sua escala original.',
        f"A remoção da normalização reduziu a AP média de {number(summary.loc['nn_base','average_precision_mean'])} para {number(summary.loc['nn_unscaled','average_precision_mean'])}. Mantendo a normalização, a versão sem one-hot atingiu {number(summary.loc['nn_ordinal','average_precision_mean'])}; a diferença para o controle foi pequena nesse conjunto de configurações.",
        'As inicializações comparadas são a distribuição padrão das camadas lineares do PyTorch, Xavier uniforme e Kaiming normal. Os vieses permanecem zero em todos os casos. Xavier e Kaiming foram avaliadas com a rede ReLU de controle; seus resultados não justificam declarar uma inicialização universalmente melhor.',
        'A comparação mantém a mesma taxa de aprendizado e o mesmo limite de épocas. A queda sem normalização descreve esse procedimento de otimização e não prova que qualquer rede sem normalização seja incapaz de aprender. As curvas da versão não normalizada usam escala symlog para permitir visualizar perdas muito maiores.'
    ],['Configuração','AP média ± DP','ROC média','Melhor época média'],p4_rows,image='preprocessing.png')
    add('9. Curvas de aprendizagem',[
        'A figura mostra quatro configurações na semente 42: controle, regularização conjunta, peso de classe e ausência de normalização. As perdas são BCE não ponderada calculada em modo de avaliação, de modo que dropout fica desligado e batch normalization usa as estatísticas aprendidas. O objetivo usado durante o gradiente pode ser ponderado, mas a curva relatada usa a mesma definição de perda.',
        'As curvas completas de todas as 90 redes estão em learning_curves.csv; melhores épocas e gaps estão em convergence.csv. Não são curvas calculadas sobre a referência.'
    ],image='learning_curves.png')
    cal_rows=[]
    for _,row in calibrations.iterrows():
        selected_marker=' (escolhida)' if decisions['calibration_methods'][row['model']] == row['method'] else ''
        mean_ece=raw_calibrations[(raw_calibrations.model==row['model'])&(raw_calibrations.method==row['method'])].ece.mean()
        cal_rows.append([MODEL_NAMES[row['model']],row['method']+selected_marker,mean_sd(row,'brier_score'),number(row['log_loss_mean']),number(mean_ece)])
    add('10. Calibração de probabilidades',[
        'Foram comparados: none, que mantém a probabilidade original; sigmoid, regressão logística ajustada sobre o logit da probabilidade original com C=1.000.000; e isotonic, regressão isotônica monotônica. O calibrador recebe apenas as previsões e os alvos de calibration. Seu desempenho é comparado em validation.',
        'A escolha usa o menor Brier médio nas cinco sementes, e não a aparência da curva ou a referência. Também são relatados log loss e ECE com dez intervalos uniformes. O diagrama de confiabilidade usa dez grupos por quantis; por isso seus agrupamentos diferem dos usados no ECE.',
        'A regressão logística escolheu isotonic, a MLP scikit-learn escolheu sigmoid e o XGBoost manteve none. A isotônica reduziu o Brier médio da regressão, mas aumentou seu log loss médio: os critérios não são equivalentes. Probabilidades isotônicas nos extremos podem receber penalidade alta quando erram.',
        'Os três modelos da aplicação foram atualizados com os métodos escolhidos. A melhor rede PyTorch permanece como artefato experimental separado; a MLP da aplicação continua sendo a implementação scikit-learn, agora calibrada.'
    ],['Modelo','Método','Brier médio ± DP','Log loss médio','ECE médio'],cal_rows,image='calibration_comparison.png',proportions=[.24,.23,.23,.16,.14])
    add('11. Confiabilidade na referência já consultada',[
        'Os pontos mostram a fração observada de inadimplentes contra a probabilidade média de cada grupo, para os modelos finais da semente 42. A diagonal representa coincidência entre as duas medidas. Oscilações, sobretudo em grupos pequenos e extremos, não devem ser interpretadas como estabilidade fora desta base.',
        'Esses diagramas são descritivos e não foram usados para refazer a escolha dos calibradores. A base de referência já havia sido observada em uma etapa anterior.'
    ],image='reliability.png')
    labels={**{key:MODEL_NAMES.get(key,key) for key in final['reference_metrics']},'knn_l1':'kNN L1','knn_l2':'kNN L2','svm':'SVM linear','random_forest':'Random forest','best_torch_mlp':f'MLP PyTorch: {config[best]["activation"]}'}
    ref_rows=[]
    for key,metric in final['reference_metrics'].items():
        interval=final['bootstrap_95'][key]['average_precision']
        ref_rows.append([labels[key],number(metric['average_precision']),f"{number(interval['low'])}–{number(interval['high'])}",number(metric['roc_auc']),number(metric['f1']),number(metric['recall'])])
    add('12. Avaliação complementar de referência',[
        'Após salvar as escolhas, os modelos foram aplicados aos mesmos 4.504 registros da referência original. A tabela mostra a realização da semente 42. AP, AUC e limiares podem diferir do primeiro desenvolvimento, pois agora parte do desenvolvimento é reservada para parada antecipada e calibração.',
        'Os intervalos abaixo são bootstrap por perfil, condicionais ao modelo ajustado. A avaliação inclui os baselines de P1 e a melhor rede selecionada na validação. Não foi feita nova seleção usando esses resultados.',
        f"O XGBoost final teve AP {number(final['reference_metrics']['xgboost']['average_precision'])}, AUC-ROC {number(final['reference_metrics']['xgboost']['roc_auc'])} e recall {number(final['reference_metrics']['xgboost']['recall'])}. Sua matriz registra 3.075 adimplentes corretamente identificados, 434 falsos positivos, 486 inadimplentes não detectados e 509 verdadeiros positivos. O desempenho continua deixando uma parcela relevante de inadimplentes sem identificação."
    ],['Modelo','AP','IC 95% AP','ROC','F1','Recall'],ref_rows,proportions=[.29,.12,.23,.12,.12,.12])
    sensitive_rows=[]
    for key in MODEL_NAMES:
        full,removed=summary.loc[key],summary.loc[key+'_no_sensitive']
        sensitive_rows.append([MODEL_NAMES[key],number(full.average_precision_mean),number(removed.average_precision_mean),number(removed.average_precision_mean-full.average_precision_mean)])
    add('13. Atributos sensíveis e diferenças por grupo',[
        'Foram repetidos os três modelos da aplicação removendo, em conjunto, sexo, idade e estado civil do pré-processamento. As comparações usam as mesmas partições e os mesmos hiperparâmetros; a tabela mostra a AP média de validação antes de calibração adicional.',
        'Uma alteração pequena de AP após excluir atributos sensíveis não comprova justiça. Outras variáveis podem funcionar como proxies. Também não se pode concluir causalidade ou adequação jurídica a partir dessas métricas.',
        'A análise descritiva por sexo, estado civil e faixa etária apresenta tamanho do grupo, prevalência, recall e taxa de falsos positivos na referência. Ela não define um critério normativo de equidade nem inclui uma validação independente de discriminação.'
    ],['Modelo','AP completa','AP sem sensíveis','Diferença'],sensitive_rows)
    groups=final['group_metrics'][selected]
    add('13.1. Grupos no modelo padrão',[
        'Codificações: SEX 1 masculino, 2 feminino; MARRIAGE 1 casado, 2 solteiro, 3 outro. AGE_GROUP indica faixas de idade. As métricas usam o limiar validado do XGBoost. Grupos pequenos têm incerteza maior; os valores abaixo não recebem interpretação causal.'
    ],['Atributo / grupo','n','Prevalência','Recall','Falsos positivos'],[
        [f"{g['attribute']} / {g['group']}",g['n'],number(g['default_rate']),number(g['recall']),number(g['false_positive_rate']) if g['false_positive_rate'] is not None else '—'] for g in groups
    ])
    costs=pd.read_csv(R/'cost_scenarios.csv')
    costs=costs[costs.model==selected]
    add('14. Cenários de custo e limiares',[
        'Foi realizada uma análise de sensibilidade ilustrativa: custo de falso positivo fixado em 1 e custo de falso negativo em 1, 2, 5 ou 10. Para cada cenário, o limiar que minimiza o custo é escolhido na validação e aplicado à referência.',
        'Os custos por mil clientes são unidades hipotéticas, não reais, NT$ ou estimativas de perda financeira. O dataset não traz custos de concessão, cobrança e recuperação suficientes para definir uma política comercial. O limiar padrão do aplicativo continua sendo o de F1 validado; estes cenários são uma análise separada.'
    ],['Custo FN / FP','Limiar validação','FN referência','FP referência','Custo / mil'],[
        [f"{int(row.false_negative_cost)} / 1",number(row.threshold_validation),int(row.reference_fn),int(row.reference_fp),number(row.cost_per_1000,1)] for _,row in costs.iterrows()
    ],proportions=[.19,.22,.18,.18,.23])
    add('15. Integração, testes e reprodutibilidade',[
        'A aplicação oferece regressão logística, XGBoost e MLP scikit-learn com os calibradores selecionados. Os fatores são sensibilidade ao substituir grupos pela mediana/moda de fit: não são explicações causais, SHAP nem parcelas aditivas. Indisponibilidade do artefato retorna HTTP 503; não há regra manual substituta.',
        'O código inclui testes de entradas inválidas, calibração sem novo ajuste do modelo base, equivalência entre API e artefato, partições sem vazamento, índices reproduzíveis, treino neural determinístico em CPU e falhas de carregamento. O navegador é verificado em desktop e celular, com troca de modelo, limiar, rede indisponível e cancelamento de resposta desatualizada.',
        'Comandos: python main.py --serve para a aplicação; python research.py --integrate --resume para repetir ou retomar a pesquisa; python build_report.py para reconstruir os PDFs; python -m pytest -q para os testes. Dependências de pesquisa e de geração dos documentos são opcionais ao uso da aplicação e estão separadas das dependências de inferência.',
        'Os artefatos incluem protocolo, resultados por execução, médias e desvios, diferenças pareadas, predições de validação e referência, curvas por época, configurações, pesos da melhor rede e checksums. As versões do ambiente e o hash dos dados permitem verificar a origem dos resultados.'
    ])
    add('16. Limitações e conclusões',[
        'P1: as redes testadas superaram kNN e classificadores lineares pela AP média observada; as melhores redes e os modelos de árvores ficaram próximos. P2: tanh teve o melhor resultado entre as ativações estudadas; profundidade maior não assegurou melhora. P3: dropout ajudou nesse desenho, enquanto a combinação de técnicas e o balanceamento não trouxeram ganhos uniformes. P4: a normalização teve efeito muito maior do que retirar one-hot ou trocar a inicialização nas condições testadas.',
        'Calibrar não é necessariamente melhorar todas as métricas. Os métodos finais foram escolhidos pelo Brier de validação, preservando a opção de não ajustar o XGBoost. A escolha por Brier pode divergir de log loss e de métricas de ordenação.',
        'Os cinco resultados por configuração não são cinco amostras independentes da população. A busca é limitada a uma grade pequena e a uma taxa de aprendizado. Não houve busca exaustiva, teste externo recente ou validação temporal. O conjunto de referência já tinha sido consultado e seus intervalos são condicionais aos modelos.',
        'A base de um banco de Taiwan em 2005 não representa automaticamente crédito brasileiro atual. Para implantação operacional seriam necessários dados recentes e autorizados, custos reais, análise de governança e validação de equidade. Essas são limitações de aplicação e generalização, e não resultados demonstrados pelo experimento.'
    ])
    references=[
        'YEH, I-Cheng; LIEN, Che-hui. Default of Credit Card Clients. UCI Machine Learning Repository, 2009. DOI: 10.24432/C55S3H. Licença CC BY 4.0.',
        'YEH, I-Cheng; LIEN, Che-hui. The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, v. 36, n. 2, 2009.',
        'CHEN, Tianqi; GUESTRIN, Carlos. XGBoost: A Scalable Tree Boosting System. KDD, 2016. DOI: 10.1145/2939672.2939785.',
        'SRIVASTAVA, Nitish et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, v. 15, 2014.',
        'IOFFE, Sergey; SZEGEDY, Christian. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML, 2015.',
        'GUO, Chuan et al. On Calibration of Modern Neural Networks. ICML, 2017.',
        'KINGMA, Diederik P.; BA, Jimmy. Adam: A Method for Stochastic Optimization. ICLR, 2015.',
        'PASZKE, Adam et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS, 2019.',
    ]
    add('Referências',references)
    add('Apêndice — versões e leitura dos arquivos',[
        'Versões: ' + '; '.join(f'{key} {value}' for key,value in final['versions'].items()) + f"; Python {final['python']}.",
        'experiments.csv: 140 ajustes e métricas de validação. summary.csv: média, desvio e intervalos descritivos. paired_differences.csv: diferenças contra a rede de controle. calibration.csv: 45 avaliações de métodos de calibração. calibration_reference.csv: avaliação complementar dos nove pares modelo/método. final_evaluation.json: escolhas, métricas e intervalos bootstrap. cost_scenarios.csv: cenários de custo. split_seed_*.npz: índices completos de cada semente.',
        'O protocolo registrado usa máximo de 80 épocas, paciência 10 e sementes 7/21/42/84/123. Os 140 ajustes incluem 7 baselines, 18 configurações PyTorch e 3 ablações de atributos sensíveis por semente. As 45 avaliações de calibração usam predições desses modelos, sem novo treinamento dos classificadores.',
        'Os documentos originais recebidos estão em docs/originais. Este relatório descreve experimentos executados e não reproduz como fatos as hipóteses das propostas anteriores.'
    ])
    return sections, summary, final, protocol


def make_document(path, sections, title, subtitle, condensed=False):
    sheet=styles()
    width=A4[0]-88
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=40,bottomMargin=46,
                         title=title,author='Equipe Risk Bank')
    story=[Spacer(1,50),para(title,sheet['TitleRisk']),para(subtitle,sheet['HeadingRisk']),
           para('Estudo executado com dados oficiais da UCI · cinco sementes · código e resultados reproduzíveis',sheet['BodyRisk']),
           Spacer(1,20),para('Autores informados no repositório: Athos Andre, Lucas Brasileiro, Guilherme Uchoa, Matheus Albuquerque e Dhigor Padilha.',sheet['BodyRisk']),
           para('Base: Taiwan, abril a setembro de 2005. O conjunto de referência já foi consultado em etapa anterior.',sheet['CaptionRisk']),PageBreak()]
    for section in sections:
        heading=para(section['title'],sheet['HeadingRisk'])
        paragraphs=section['paragraphs'][:2] if condensed else section['paragraphs']
        if paragraphs:
            story.append(KeepTogether([heading,para(paragraphs[0],sheet['BodyRisk'])]))
            story.extend(para(text,sheet['BodyRisk']) for text in paragraphs[1:])
        else:
            story.append(heading)
        if section['headers']:
            story += [table(section['headers'],section['rows'],width,sheet,section.get('proportions')),Spacer(1,12)]
        if section['image'] and not condensed:
            story.append(figure(R/section['image'],width))
        if section['caption'] and not condensed:
            story.append(para(section['caption'],sheet['CaptionRisk']))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def make_slides(path,sections,summary,final):
    sheet=styles(True)
    pagesize=landscape(A4)
    width=pagesize[0]-88
    story=[]
    chosen=[0,1,2,3,5,6,7,8,9,10,11,12,13,15,17,18]
    for index,section_index in enumerate(chosen):
        section=sections[section_index]
        if index:
            story.append(PageBreak())
        title='Previsão de inadimplência de crédito' if index==0 else section['title']
        story.append(para(title,sheet['HeadingRisk']))
        if section_index in [7,9,11]:
            image=section['image']
            if image:
                story.append(figure(R/image,width,max_height=340))
            story.append(para(section['paragraphs'][0],sheet['CaptionRisk']))
        elif section['headers']:
            # Um quadro por slide, sem repetir os parágrafos extensos do artigo.
            rows=section['rows']
            if len(rows)>9:
                rows=rows[:9]
            story.append(table(section['headers'],rows,width,sheet,section.get('proportions')))
            story.append(Spacer(1,15))
            takeaway={
                2:'Referência fixa: 4.504 registros já consultados. Ajuste, parada, calibração e seleção usam partições distintas.',
                5:'Redes e árvores ficaram próximas. O XGBoost liderou os três modelos disponíveis na aplicação.',
                6:'Mais camadas não garantiram melhor AP. Tanh liderou as redes avaliadas.',
                8:'A normalização teve o maior efeito nesse desenho; não houve ganho uniforme com Xavier ou Kaiming.',
                10:'A escolha usa Brier médio de validação. Calibração não melhora necessariamente todas as métricas.',
                12:'Avaliação complementar, sem alegação de teste inédito. Intervalos condicionais ao modelo ajustado.',
                13:'Excluir atributos sensíveis não comprova ausência de discriminação.'
            }.get(section_index,'Resultados do protocolo de cinco sementes; os detalhes e as limitações estão no relatório.')
            story.append(para(takeaway,sheet['BodyRisk']))
        else:
            paragraphs=section['paragraphs']
            if section_index==0:
                paragraphs=[paragraphs[0],paragraphs[1],'Entrega: aplicação integrada, resultados brutos, relatório completo, síntese e apresentação.']
            elif section_index==3:
                paragraphs=['Cinco sementes: 7, 21, 42, 84 e 123. As mesmas partições são usadas por todos os candidatos em cada repetição.',
                            '140 ajustes: 7 baselines, 18 redes e 3 ablações de atributos sensíveis por semente. Máximo de 80 épocas nas redes, com parada antecipada.',
                            'Modelo: AP média de validação. Calibrador: Brier médio de validação. Limiar: F1 na validação. A referência já consultada não participa dessas escolhas.']
            elif section_index==1:
                paragraphs=[paragraphs[0],paragraphs[1]]
            elif section_index==17:
                paragraphs=[paragraphs[0],paragraphs[1],paragraphs[2]]
            else:
                paragraphs=paragraphs[:3]
            for text in paragraphs:
                story.append(para(text,sheet['BodyRisk']))
    doc=SimpleDocTemplate(str(path),pagesize=pagesize,rightMargin=44,leftMargin=44,topMargin=32,bottomMargin=43,
                         title='Risk Bank — apresentação dos resultados',author='Equipe Risk Bank')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def main():
    register_fonts()
    DOCS.mkdir(exist_ok=True)
    originals=DOCS/'originais'
    originals.mkdir(exist_ok=True)
    names=['Previsão de Inadimplência de Crédito com Redes Neurais.pdf','proposta_pesquisa_inadimplencia.pdf','apresentacao_completa_original.pdf']
    for name in names:
        current=DOCS/name
        if current.exists() and not (originals/name).exists():
            shutil.copy2(current,originals/name)
    sections,summary,final,protocol=load_sections()
    title='Previsão de Inadimplência de Crédito com Redes Neurais'
    make_document(DOCS/names[0],sections,title,'Relatório de resultados e comparação com modelos de referência')
    short=[sections[index] for index in [0,2,3,5,10,12,17]]
    make_document(DOCS/names[1],short,'Síntese da pesquisa de inadimplência','Da proposta aos experimentos executados',condensed=True)
    make_slides(DOCS/'apresentacao_resultados.pdf',sections,summary,final)
    # O terceiro documento da entrega inicial é preservado; sua versão atual é a apresentação de resultados.
    if (DOCS/'apresentacao_completa_original.pdf').exists():
        (DOCS/'apresentacao_completa_original.pdf').unlink()
    text=['# '+title,'','Relatório dos experimentos executados. Documentos anteriores preservados em docs/originais.','']
    for section in sections:
        text += ['## '+section['title'],'',*sum(([paragraph,''] for paragraph in section['paragraphs']),[])]
        if section['headers']:
            text += [markdown_table(section['headers'],section['rows']),'']
        if section['image']:
            text += [f"![{section['title']}](results/research/{section['image']})",'']
    (ROOT/'PESQUISA.md').write_text('\n'.join(text),encoding='utf-8')
    print('PDFs e PESQUISA.md gerados a partir dos resultados reais.')


if __name__=='__main__':
    main()
