import streamlit as st
from models.text_emotion import TextEmotionDetector
from models.speech_emotion import SpeechEmotionDetector
from models.face_emotion import FaceEmotionDetector
from models.emotion_fusion import EmotionFusion
from models.prompt_generator import PromptGenerator
from models.langchain_client import LangChainClient
import base64
import tempfile
import os
from PIL import Image
import numpy as np

# Initialize models
text_detector = TextEmotionDetector()
speech_detector = SpeechEmotionDetector()
face_detector = FaceEmotionDetector()
emotion_fusion = EmotionFusion()
prompt_generator = PromptGenerator()
llm_client = LangChainClient()

# Streamlit UI setup
st.title("Mental Health Chatbot")

# Text input for emotions
text_input = st.text_input("Enter your text:")

# Audio input for emotions (upload button)
audio_file = st.file_uploader("Upload an audio file (optional)", type=["mp3", "wav"])

# Image input for emotions (upload button)
image_file = st.file_uploader("Upload an image (optional)", type=["jpg", "png"])

if text_input:
    # Process text emotion
    text_emotion = text_detector.predict(text_input)
    st.write(f"Detected emotion from text: {text_emotion}")

if audio_file:
    try:
        audio_data = audio_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_data)
            speech_emotion = speech_detector.predict(tmp_file.name)
            os.unlink(tmp_file.name)
            st.write(f"Detected emotion from speech: {speech_emotion}")
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")

if image_file:
    try:
        image = Image.open(image_file)
        image_np = np.array(image)
        face_emotion = face_detector.predict(image_np)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write(f"Detected emotion from face: {face_emotion}")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Fuse emotions
dominant_emotion = emotion_fusion.fuse_emotions(text_emotion, speech_emotion, face_emotion)
st.write(f"Dominant emotion: {dominant_emotion}")

# Generate prompt with emotion context
prompt = prompt_generator.generate_prompt(dominant_emotion, text_input)
response = llm_client.run(prompt)

st.write(f"Chatbot response: {response}")
