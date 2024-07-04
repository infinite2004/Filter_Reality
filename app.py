from flask import Flask, request, render_template, send_from_directory
import os
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return 'File uploaded successfully'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/start-ar', methods=['POST'])
def start_ar():
    try:
        # Start the AR application
        subprocess.Popen(['python', 'ar_app.py'])
        return 'AR application started successfully'
    except Exception as e:
        return f'Error starting AR application: {e}'

if __name__ == '__main__':
    app.run(debug=True)