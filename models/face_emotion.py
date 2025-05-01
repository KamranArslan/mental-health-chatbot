import tensorflow as tf
import numpy as np
import cv2

class FaceEmotionDetector:
    def __init__(self, model_path="models/face_model.h5"):
        self.model = tf.keras.models.load_model(model_path)
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def preprocess_image(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
            
        # Get the first face
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        
        # Resize to model input size (adjust size based on your model's requirements)
        face = cv2.resize(face, (48, 48))
        
        # Normalize
        face = face / 255.0
        
        # Reshape for model input
        return face.reshape(1, 48, 48, 1)
    
    def predict(self, image):
        processed_image = self.preprocess_image(image)
        if processed_image is None:
            return "neutral"  # Default emotion if no face detected
            
        predictions = self.model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        return self.emotions[predicted_class]
