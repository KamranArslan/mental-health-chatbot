import tensorflow as tf
import numpy as np
import librosa

class SpeechEmotionDetector:
    def __init__(self, model_path="models/speech_model.h5"):
        # Load pre-trained speech emotion recognition model
        self.model = tf.keras.models.load_model(model_path)
        
        # Define emotion labels (must match model’s output order)
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
        
    def extract_features(self, audio_path):
        # Load audio file (3 seconds duration starting from 0.5s)
        y, sr = librosa.load(audio_path, duration=3, offset=0.5)
        
        # Generate Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Resize or pad spectrogram to 128x128
        if mel_db.shape[1] < 128:
            mel_db = np.pad(mel_db, ((0, 0), (0, 128 - mel_db.shape[1])), mode='constant')
        else:
            mel_db = mel_db[:, :128]
        
        # Normalize spectrogram values
        mel_db = mel_db / 255.0
        
        # Reshape to match CNN input shape (batch_size, height, width, channels)
        return mel_db.reshape(1, 128, 128, 1)
    
    def predict(self, audio_path):
        # Extract features from audio
        features = self.extract_features(audio_path)
        
        # Predict emotion probabilities
        predictions = self.model.predict(features)
        
        # Get index of highest probability
        predicted_class = np.argmax(predictions[0])
        
        # Return corresponding emotion label
        return self.emotions[predicted_class]
