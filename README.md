# Trabalho-Final
Análise de Dados de Animes e Predição de Scores
Este projeto é uma aplicação web desenvolvida em Flask, que permite o upload de arquivos CSV contendo dados sobre animes. A partir desses dados, a aplicação gera gráficos de análise e treina um modelo de regressão linear para prever os scores dos animes com base em variáveis como rank e popularidade.

Funcionalidades
Upload de Arquivos CSV: O usuário pode enviar um arquivo CSV contendo dados de animes.
Análise Exploratória de Dados: Após o upload do arquivo, a aplicação gera os seguintes gráficos:
Histograma da distribuição de scores
Mapa de calor de correlação entre variáveis
Boxplot de scores
Gráfico de dispersão entre score e rank
Gráfico de densidade dos scores
Modelagem Preditiva: Um modelo de Regressão Linear é treinado para prever os scores dos animes com base no rank e na popularidade. O modelo utiliza a métrica de erro quadrático médio (MSE) para avaliar a precisão das previsões.
Armazenamento de Gráficos: Os gráficos gerados são armazenados na pasta static/images e podem ser visualizados na página de resultados.
Tecnologias Utilizadas
Flask: Framework web para criar a aplicação.
Pandas: Para manipulação de dados CSV.
Seaborn e Matplotlib: Para a criação de gráficos e visualizações.
Scikit-learn: Para implementação do modelo de regressão linear e pré-processamento dos dados.