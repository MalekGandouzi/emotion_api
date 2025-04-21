from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import librosa

app = Flask(__name__)
model = load_model('emotion_model.keras')

@app.route('/')
def home():
    return 'API de reconnaissance d’émotions OK 😄'

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    y, sr = librosa.load(file, sr=22050)
    features = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    prediction = model.predict(np.expand_dims(features, axis=0))
    emotion = int(np.argmax(prediction))
    return jsonify({'emotion': emotion})
