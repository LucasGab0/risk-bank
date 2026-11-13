# Base de dados

`default_credit.csv.gz` contém as 30.000 linhas da planilha oficial do dataset **Default of Credit Card Clients**, preservando o ID, os atributos e o alvo original. O ID é excluído do treinamento. A compressão apenas reduz o tamanho da entrega.

Autores: I-Cheng Yeh e Che-hui Lien (2009). Fonte: UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C55S3H. Licença: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

`source.json` registra a URL oficial, o nome e o SHA-256 da planilha original, o SHA-256 do CSV gzip e a conversão realizada. As transformações de categorias usadas no treinamento são feitas em `data.py`, sem modificar este arquivo de dados.
