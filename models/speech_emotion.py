import tensorflow as tf
import numpy as np
import librosa

class SpeechEmotionDetector:
    def __init__(self, model_path="models/speech_model.h5"):
        self.model = tf.keras.models.load_model(model_path)
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
        
    def extract_features(self, audio_path):
        # Load audio file
        y, sr = librosa.load(audio_path, duration=3, offset=0.5)
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        # Reshape for model input
        return mfcc_scaled.reshape(1, -1)
    
    def predict(self, audio_path):
        features = self.extract_features(audio_path)
        predictions = self.model.predict(features)
        predicted_class = np.argmax(predictions[0])
        return self.emotions[predicted_class] 