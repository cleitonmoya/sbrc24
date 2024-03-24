#Artigo SBRC 2024

## Descrição
Este repositório contém o código-fonte e a base de dados utilizados no artigo **Inferindo pontos de mudança em séries temporais com dados
não rotulados: um breve estudo usando dados do NDT**, submetido ao [42º Simpósio Brasileiro de Redes de Computadores e Sistemas Distribuídos (SBRC)](https://sbrc.sbc.org.br/2024/).

## Instruções
- A implementação dos métodos de detecção de *change-point* encontra-se no módulo [changepoint.py](experiment/changepoint.py)
- Para reproduzir o experimento, clone o respositório e execute o *script* [experiment.py](experiment/experiment.py)
	- O experimento gera o *data-frame* `df_results.pkl`.
- Os gráficos são gerados com os *scripts* da pasta [paper_figures](paper_figures/) 

## Requisitos
Os algoritmos foram implementados e testados em **Python 3.9** com os seguintes pacotes:
- Numpy 1.23.5
- Scipy 1.11.4
- Pandas 1.5.2
- Statsmodels 0.14.1
- Matplotlib 3.8.2
- Seaborn 0.13.1


