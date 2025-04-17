from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import base64
import io

from models.text_emotion import TextEmotionDetector
from models.speech_emotion import SpeechEmotionDetector
from models.face_emotion import FaceEmotionDetector
from models.emotion_fusion import EmotionFusion
from models.prompt_generator import PromptGenerator
from models.langchain_client import LangChainClient

app = Flask(__name__)

# Initialize models
text_detector = TextEmotionDetector()
speech_detector = SpeechEmotionDetector()
face_detector = FaceEmotionDetector()
emotion_fusion = EmotionFusion()
prompt_generator = PromptGenerator()
llm_client = LangChainClient()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    text_input = data.get('text', '')
    
    # Process text emotion
    text_emotion = None
    if text_input:
        text_emotion = text_detector.predict(text_input)
    
    # Process audio if provided
    speech_emotion = None
    if 'audio' in data and data['audio']:
        try:
            # Decode base64 audio
            audio_data = base64.b64decode(data['audio'].split(',')[1])
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_data)
                speech_emotion = speech_detector.predict(tmp_file.name)
                os.unlink(tmp_file.name)
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
    
    # Process image if provided
    face_emotion = None
    if 'image' in data and data['image']:
        try:
            # Decode base64 image
            image_data = base64.b64decode(data['image'].split(',')[1])
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)
            face_emotion = face_detector.predict(image_np)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
    
    # Fuse emotions
    dominant_emotion = emotion_fusion.fuse_emotions(text_emotion, speech_emotion, face_emotion)
    
    # Generate prompt with emotion context
    prompt = prompt_generator.generate_prompt(dominant_emotion, text_input)
    
    # Use LangChain's run method to process the input with memory
    response = llm_client.run(prompt)
    
    return jsonify({
        "emotion": dominant_emotion,
        "response": response
    })

if __name__ == '__main__':
    app.run(debug=True) 