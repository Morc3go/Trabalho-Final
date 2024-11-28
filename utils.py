import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Modelo de regressão linear
from sklearn.metrics import mean_squared_error  # Para calcular o erro quadrático médio
from sklearn.impute import SimpleImputer  # Para lidar com NaNs


# Função para processar os dados e gerar gráficos
def process_data(data):
    # Normalizar as colunas do DataFrame, removendo espaços e convertendo para minúsculas
    data.columns = data.columns.str.strip().str.lower()

    # Garantir que a coluna 'score', 'rank' e 'popularity' sejam numéricas (convertendo valores não numéricos para NaN)
    data['score'] = pd.to_numeric(data['score'], errors='coerce')
    data['rank'] = pd.to_numeric(data['rank'], errors='coerce')
    data['popularity'] = pd.to_numeric(data['popularity'], errors='coerce')

    # Definir as colunas disponíveis para análise
    available_columns = ['score', 'rank', 'popularity']

    # Remover qualquer linha que tenha NaN nas colunas importantes ('score', 'rank', 'popularity')
    data = data.dropna(subset=available_columns)

    # Verificar se a pasta 'static/images' existe, caso contrário, cria a pasta
    if not os.path.exists('static/images'):
        os.makedirs('static/images')

    # Inicializar os caminhos dos gráficos como None
    hist_path = corr_path = boxplot_path = scatter_path = density_path = None

    # Gerar um histograma da coluna 'score', caso ela exista
    if 'score' in available_columns:
        plt.figure(figsize=(10, 6))  # Definir o tamanho do gráfico
        sns.histplot(data['score'], kde=True, color='skyblue')  # Gerar histograma com KDE
        plt.title('Distribution of Scores')  # Título do gráfico
        plt.xlabel('Score')  # Rótulo do eixo X
        plt.ylabel('Frequency')  # Rótulo do eixo Y
        plt.tight_layout()  # Ajustar layout para evitar sobreposição
        hist_path = 'static/images/histogram.png'  # Caminho onde o gráfico será salvo
        plt.savefig(hist_path, bbox_inches='tight')  # Salvar o gráfico
        plt.close()  # Fechar a figura para liberar memória

    # Gerar uma matriz de correlação das colunas selecionadas
    if len(available_columns) > 1:
        plt.figure(figsize=(10, 8))
        corr = data[available_columns].corr()  # Calcular a correlação entre as colunas
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)  # Criar o gráfico de calor
        plt.title('Correlation Heatmap')  # Título do gráfico
        plt.tight_layout()  # Ajustar layout
        corr_path = 'static/images/correlation.png'  # Caminho para salvar o gráfico
        plt.savefig(corr_path, bbox_inches='tight')  # Salvar o gráfico
        plt.close()

    # Gerar um boxplot da coluna 'score', caso ela exista
    if 'score' in available_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data['score'])  # Criar boxplot da coluna 'score'
        plt.title('Boxplot of Scores')  # Título do gráfico
        plt.tight_layout()  # Ajustar layout
        boxplot_path = 'static/images/boxplot.png'  # Caminho para salvar o gráfico
        plt.savefig(boxplot_path, bbox_inches='tight')  # Salvar o gráfico
        plt.close()

    # Gerar um gráfico de dispersão entre 'score' e 'rank', caso ambas as colunas existam
    if 'score' in available_columns and 'rank' in available_columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data['score'], y=data['rank'], color='green')  # Criar gráfico de dispersão
        plt.title('Scatter Plot of Score vs Rank')  # Título do gráfico
        plt.xlabel('Score')  # Rótulo do eixo X
        plt.ylabel('Rank')  # Rótulo do eixo Y
        plt.tight_layout()  # Ajustar layout
        scatter_path = 'static/images/scatter.png'  # Caminho para salvar o gráfico
        plt.savefig(scatter_path, bbox_inches='tight')  # Salvar o gráfico
        plt.close()

    # Gerar um gráfico de densidade para a coluna 'score', caso ela exista
    if 'score' in available_columns:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data['score'], fill=True, color='purple')  # Criar gráfico de densidade
        plt.title('Density Plot of Scores')  # Título do gráfico
        plt.xlabel('Score')  # Rótulo do eixo X
        plt.ylabel('Density')  # Rótulo do eixo Y
        plt.tight_layout()  # Ajustar layout
        density_path = 'static/images/density_plot.png'  # Caminho para salvar o gráfico
        plt.savefig(density_path, bbox_inches='tight')  # Salvar o gráfico
        plt.close()

    # Retornar os caminhos de todos os gráficos gerados (alguns valores podem ser None)
    return hist_path, corr_path, boxplot_path, scatter_path, None, None, None, None


# Função para treinar um modelo de regressão linear
def train_model(data):
    # Definir as variáveis independentes (X) e a variável dependente (y)
    X = data[['rank', 'popularity']]  # Variáveis independentes
    y = data['score']  # Variável dependente (o que queremos prever)

    # Verificar quantos valores ausentes (NaN) existem na variável dependente 'y'
    print(f"NaN values in 'score': {y.isnull().sum()}")

    # Se existirem valores ausentes, removê-los da variável dependente
    if y.isnull().sum() > 0:
        print(f"Found {y.isnull().sum()} NaN values in the target variable 'score'. Removing them.")
        data = data.dropna(subset=['score'])  # Remover linhas onde 'score' é NaN
        X = data[['rank', 'popularity']]  # Recarregar as variáveis independentes após remoção de NaN em 'score'
        y = data['score']  # Recarregar 'y' após a remoção

    # Verificar novamente a quantidade de valores ausentes
    print(f"NaN values in 'score' after cleaning: {y.isnull().sum()}")

    # Preencher valores ausentes nas variáveis independentes com a média
    imputer = SimpleImputer(strategy='mean')  # Substituir NaN pela média
    X_imputed = imputer.fit_transform(X)  # Aplicar o imputador nas variáveis independentes

    # Dividir os dados em conjuntos de treino (70%) e teste (30%)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

    # Inicializar e treinar o modelo de regressão linear
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Fazer previsões com o modelo treinado
    y_pred = model.predict(X_test)

    # Calcular o erro quadrático médio (MSE) das previsões
    mse = mean_squared_error(y_test, y_pred)

    # Retornar o erro quadrático médio (MSE) e as previsões feitas
    return {
        'mse': mse,  # Retornar o erro quadrático médio
        'predictions': y_pred  # Retornar as previsões feitas pelo modelo
    }
