import tensorflow as tf
import numpy as np
import librosa
import os

class SpeechEmotionDetector:
    def __init__(self, model_path="models/speech_model.h5"):
        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        print(f"Loading model from {model_path} ...")
        try:
            # Handle .h5 or SavedModel formats automatically
            if model_path.endswith(".h5"):
                # Add custom_objects={} if you have custom layers
                self.model = tf.keras.models.load_model(model_path, compile=False, custom_objects={})
            else:
                self.model = tf.keras.models.load_model(model_path)

            print("✅ Model loaded successfully!")
            print(self.model.summary())
        except Exception as e:
            print("❌ Failed to load model:", str(e))
            print("⚠️ TIP: If the model uses custom layers, add them to 'custom_objects'.")
            raise e

        # Define emotion labels (ensure correct order matching model output)
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

    def extract_features(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio (3 sec segment starting after 0.5s offset)
        y, sr = librosa.load(audio_path, duration=3, offset=0.5)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        print(f"Extracted Mel Spectrogram Shape: {mel_db.shape}")

        # Resize or pad to 128x128
        if mel_db.shape[1] < 128:
            mel_db = np.pad(mel_db, ((0, 0), (0, 128 - mel_db.shape[1])), mode='constant')
        else:
            mel_db = mel_db[:, :128]

        # Normalize and reshape
        mel_db = mel_db / 255.0
        return mel_db.reshape(1, 128, 128, 1)

    def predict(self, audio_path):
        features = self.extract_features(audio_path)
        predictions = self.model.predict(features)

        predicted_class = np.argmax(predictions[0])
        predicted_emotion = self.emotions[predicted_class]

        print(f"Predicted Emotion: {predicted_emotion}")
        return predicted_emotion

