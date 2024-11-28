import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from models import train_model
from utils import process_data

app = Flask(__name__)

UPLOAD_FOLDER = 'datasets/databases'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        data = pd.read_csv(file_path)

        hist_path, corr_path, boxplot_path = process_data(data)

        return render_template('analyze.html', data=data.to_html(),
                               hist_path=hist_path, corr_path=corr_path, boxplot_path=boxplot_path)

@app.route('/train', methods=['POST'])
def train():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'your_file.csv')
    data = pd.read_csv(file_path)

    model, accuracy = train_model(data)

    return render_template('result.html', accuracy=accuracy)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
