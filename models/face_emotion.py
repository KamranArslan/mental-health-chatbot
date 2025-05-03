import tensorflow as tf
import numpy as np
import cv2

class FaceEmotionDetector:
    def __init__(self, model_path="models/face_model.h5"):
        # Load pre-trained facial emotion recognition model
        self.model = tf.keras.models.load_model(model_path)
        
        # Define emotion labels (order must match model's output)
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
        
        # Load OpenCV face detection model (Haar cascade)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def preprocess_image(self, image):
        # Convert image to grayscale (required for face detection)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # If no faces are detected, return None
        if len(faces) == 0:
            print("No face detected!")
            return None  # Or return 'neutral' directly if preferred
            
        # Use the first detected face (x, y, width, height)
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        
        # Resize face to match model's input size (48x48 for most models)
        face = cv2.resize(face, (48, 48))
        
        # Normalize pixel values
        face = face / 255.0
        
        # Reshape to match model's expected input shape: (batch_size, height, width, channels)
        return face.reshape(1, 48, 48, 1)
    
    def predict(self, image):
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        
        # If no face detected, return 'neutral' emotion by default
        if processed_image is None:
            return "neutral"
        
        # Predict emotion probabilities
        predictions = self.model.predict(processed_image)
        
        # Get index of the highest probability
        predicted_class = np.argmax(predictions[0])
        
        # Return the corresponding emotion label
        return self.emotions[predicted_class]
