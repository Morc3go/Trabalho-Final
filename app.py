from flask import Flask, render_template, request, flash, redirect, url_for
import os
import pandas as pd
from werkzeug.utils import secure_filename
from utils import process_data, train_model

# Criando a instância do aplicativo Flask
app = Flask(__name__)

# Definindo uma chave secreta para o uso do mecanismo de flash
app.secret_key = 'your_secret_key'  # Defina uma chave secreta para usar o flash

# Diretório onde os arquivos enviados serão armazenados
UPLOAD_FOLDER = 'datasets/databases'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Definindo o diretório de upload
app.config['ALLOWED_EXTENSIONS'] = {'csv'}  # Permitindo apenas arquivos CSV

# Função para verificar se o arquivo tem uma extensão permitida (CSV)
def allowed_file(filename):
    # Verifica se o nome do arquivo contém um ponto e se a extensão é .csv
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Página inicial (agora será a página de upload)
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # Verifica se o método de requisição é POST (quando o formulário é enviado)
    if request.method == 'POST':
        # Verifica se o arquivo foi enviado
        if 'file' not in request.files:
            flash('No file part')  # Exibe uma mensagem de erro caso não haja arquivo
            return redirect(request.url)  # Redireciona para a página de upload

        file = request.files['file']  # Obtém o arquivo enviado
        # Verifica se o arquivo não foi selecionado
        if file.filename == '':
            flash('No selected file')  # Exibe uma mensagem de erro
            return redirect(request.url)

        # Se o arquivo for válido, faz o processamento
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # Garante que o nome do arquivo seja seguro
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Caminho para salvar o arquivo

            # Verifica se a pasta de upload existe, caso contrário cria-a
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            file.save(file_path)  # Salva o arquivo na pasta especificada
            print(f"File uploaded and saved at: {file_path}")  # Debug: Exibe o caminho onde o arquivo foi salvo

            try:
                # Lê o arquivo CSV para um DataFrame do pandas
                data = pd.read_csv(file_path)

                # Normaliza os nomes das colunas (removendo espaços e convertendo para minúsculas)
                data.columns = data.columns.str.strip().str.lower()

                # Exibe as colunas após a normalização para fins de depuração
                print("Columns after normalization:", data.columns)

                # Verifica se a coluna 'score' existe no conjunto de dados
                if 'score' in data.columns:
                    # Processa os dados e gera os gráficos
                    hist_path, corr_path, boxplot_path, scatter_path, line_path, barplot_path, pie_path, density_path = process_data(data)

                    # Treina um modelo (por exemplo, Regressão Logística) com os dados
                    model_results = train_model(data)

                    # Passa os caminhos dos gráficos e resultados do modelo para a página de resultados
                    return render_template('analyze.html',
                                           hist_path=hist_path,
                                           corr_path=corr_path,
                                           boxplot_path=boxplot_path,
                                           scatter_path=scatter_path,
                                           line_path=line_path,
                                           barplot_path=barplot_path,
                                           pie_path=pie_path,
                                           density_path=density_path,
                                           model_results=model_results)
                else:
                    flash("The 'Score' column was not found in the dataset.")  # Exibe mensagem de erro se a coluna 'score' não for encontrada
                    return redirect(request.url)

            except Exception as e:
                print(f"Error processing the file: {e}")  # Exibe erro se algo der errado ao processar o arquivo
                flash(f"There was an error processing the file: {e}")  # Exibe mensagem de erro
                return redirect(request.url)

        else:
            flash('Invalid file format. Please upload a CSV file.')  # Exibe mensagem de erro se o arquivo não for CSV
            return redirect(request.url)

    return render_template('index.html')  # Exibe o formulário de upload para requisições GET

# Iniciar o servidor Flask
if __name__ == '__main__':
    # Garantir que a pasta de upload exista antes de iniciar o servidor
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    app.run(debug=True)  # Inicia o servidor em modo de depuração (debug mode)
