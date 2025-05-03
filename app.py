import streamlit as st
from models.text_emotion import TextEmotionDetector
from models.speech_emotion import SpeechEmotionDetector
from models.face_emotion import FaceEmotionDetector
from models.emotion_fusion import EmotionFusion
from models.prompt_generator import PromptGenerator
from models.langchain_client import LangChainClient

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

# Setup Streamlit app
st.set_page_config(page_title="Mental Health Chatbot", page_icon="🧠")
st.title("🧠 Mental Health Chatbot")

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_emotion" not in st.session_state:
    st.session_state.last_emotion = "neutral"

# Suggestions based on emotions
suggestions = {
    "anger": "Take a deep breath and count to ten. Step away from the situation if needed.",
    "disgust": "Pause and reflect. Try to reframe or distance from the triggering situation.",
    "fear": "Try grounding techniques like 5-4-3-2-1 to stay present.",
    "happiness": "Celebrate your joy! Share your moment with someone close.",
    "neutral": "Stay mindful. Even neutral states can offer moments of clarity.",
    "sadness": "You're not alone. Reach out to someone you trust or write down your thoughts.",
    "surprise": "Take time to process what happened. Talk it out if needed."
}

# --- Input Section ---
st.subheader("💬 Share Your Current Feelings")
text_input = st.text_input("Enter your message:", key="main_text_input")
audio_file = st.file_uploader("🎙️ Upload audio (optional)", type=["mp3", "wav"])
image_file = st.file_uploader("🖼️ Upload image (optional)", type=["jpg", "png"])

# --- Input Submission Button ---
if st.button("▶️ Submit Initial Message") and (text_input.strip() or audio_file or image_file):
    # Detect text emotion if text is provided
    text_emotion = text_detector.predict(text_input) if text_input.strip() else None
    if text_emotion:
        st.write(f"**📝 Text Emotion:** `{text_emotion}`")

    # Detect speech emotion if audio is provided
    speech_emotion = None
    if audio_file:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_file.read())
                speech_emotion = speech_detector.predict(tmp_file.name)
            os.unlink(tmp_file.name)
            st.write(f"**🎧 Speech Emotion:** `{speech_emotion}`")
        except Exception as e:
            st.error(f"⚠️ Audio error: {e}")
            speech_emotion = None

    # Detect face emotion if image is provided
    face_emotion = None
    if image_file:
        try:
            image = Image.open(image_file)
            image_np = np.array(image)
            face_emotion = face_detector.predict(image_np)
            st.image(image_np, caption="Uploaded Image", use_container_width=True)
            st.write(f"**📷 Face Emotion:** `{face_emotion}`")
        except Exception as e:
            st.error(f"⚠️ Image error: {e}")
            face_emotion = None

    # Fuse emotions and determine dominant emotion
    dominant_emotion = emotion_fusion.fuse_emotions(text_emotion, speech_emotion, face_emotion) or "neutral"
    st.session_state.last_emotion = dominant_emotion
    st.write(f"**🧭 Dominant Emotion:** `{dominant_emotion}`")

    # Generate a prompt for the language model
    prompt = prompt_generator.generate_prompt(dominant_emotion, text_input, st.session_state.chat_history)
    # Get response from LangChainClient
    response = llm_client.run(user_input=text_input, dominant_emotion=dominant_emotion)
    suggestion = suggestions.get(dominant_emotion.lower(), "")
    full_response = f"{response}\n\n**🧘 Therapeutic Suggestion:** {suggestion}"

    # Update chat history
    st.session_state.chat_history.append(("You", text_input))
    st.session_state.chat_history.append(("Chatbot", full_response))

# --- Chat History Display ---
def display_chat_history():
    st.subheader("🗨️ Conversation")
    for speaker, message in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Chatbot:** {message}")

display_chat_history()

# --- Follow-up Section ---
st.subheader("💭 How are you feeling now?")
followup_input = st.text_input("You can share a follow-up message:", key="followup_input")

if st.button("🔁 Continue Conversation") and followup_input.strip():
    followup_emotion = text_detector.predict(followup_input)
    st.write(f"**📝 Follow-up Text Emotion:** `{followup_emotion}`")

    updated_emotion = emotion_fusion.fuse_emotions(followup_emotion, None, None) or "neutral"
    st.session_state.last_emotion = updated_emotion

    prompt = prompt_generator.generate_prompt(updated_emotion, followup_input, st.session_state.chat_history)
    response = llm_client.run(user_input=followup_input, dominant_emotion=updated_emotion)
    suggestion = suggestions.get(updated_emotion.lower(), "")
    full_response = f"{response}\n\n**🧘 Therapeutic Suggestion:** {suggestion}"

    st.session_state.chat_history.append(("You", followup_input))
    st.session_state.chat_history.append(("Chatbot", full_response))

    display_chat_history()
