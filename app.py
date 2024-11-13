from flask import Flask, render_template, request, redirect, send_from_directory, flash, Response
import pickle
import librosa
import soundfile
import numpy as np
from werkzeug.utils import secure_filename
import os
import miniaudio
from scipy.io import wavfile
import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = 'static/uploads/convo1_chunk1.wav'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = pickle.load(open('MlpModel3.pkl', 'rb'))

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
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        print("FORM DATA RECEIVED")
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        print(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'])
        file.save(path)
        return redirect(url_for('index'))
        if file.filename == "":
            return redirect(request.url)
        if file:
            new_feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
            arr = model.predict([new_feature])
            prediction = arr[0]
    return render_template('index.html', prediction_text=prediction)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
