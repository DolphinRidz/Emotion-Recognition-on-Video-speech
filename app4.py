from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
import pickle
import librosa
import soundfile
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
model = pickle.load(open('MlpModel3.pkl', 'rb'))
ALLOWED_EXTENSIONS = set(['wav'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

@app.route('/')
def home():
    return render_template('index3.html')


@app.route('/', methods=['POST'])
def upload_predict():
    if 'file' not in request.files:
        flash('No audio file given')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No audio selected yet. please select an audio')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        prediction = ""
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        new_feature = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
        arr = model.predict([new_feature])
        prediction = arr[0]
        flash(filename)
        return render_template('index3.html', filename=filename, prediction= prediction)
    else:
        flash('Allowed audio types are only .wav')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_audio(filename):
    print('display_audio filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)