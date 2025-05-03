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

# Initialize all models
text_detector = TextEmotionDetector()
speech_detector = SpeechEmotionDetector()
face_detector = FaceEmotionDetector()
emotion_fusion = EmotionFusion()
prompt_generator = PromptGenerator()
llm_client = LangChainClient()

# Set up page
st.set_page_config(page_title="Mental Health Chatbot", page_icon="🧠")
st.title("🧠 Mental Health Chatbot")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'last_emotion' not in st.session_state:
    st.session_state.last_emotion = "neutral"

# Suggestions dictionary
suggestions = {
    "anger": "Take a deep breath and count to ten. Step away from the situation if needed.",
    "disgust": "Pause and reflect. Try to reframe or distance from the triggering situation.",
    "fear": "Try grounding techniques like 5-4-3-2-1 to stay present.",
    "happiness": "Celebrate your joy! Share your moment with someone close.",
    "neutral": "Stay mindful. Even neutral states can offer moments of clarity.",
    "sadness": "You're not alone. Reach out to someone you trust or write down your thoughts.",
    "surprise": "Take time to process what happened. Talk it out if needed."
}

# Inputs
text_input = st.text_input("💬 Enter your message:")
audio_file = st.file_uploader("🎙️ Upload an audio file (optional)", type=["mp3", "wav"])
image_file = st.file_uploader("🖼️ Upload an image (optional)", type=["jpg", "png"])

# Emotion detection
text_emotion = speech_emotion = face_emotion = None

if text_input.strip():
    text_emotion = text_detector.predict(text_input)
    st.write(f"**📝 Text Emotion:** `{text_emotion}`")

if audio_file:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            speech_emotion = speech_detector.predict(tmp_file.name)
        os.unlink(tmp_file.name)
        st.write(f"**🎧 Speech Emotion:** `{speech_emotion}`")
    except Exception as e:
        st.error(f"⚠️ Error processing audio: {e}")

if image_file:
    try:
        image = Image.open(image_file)
        image_np = np.array(image)
        face_emotion = face_detector.predict(image_np)
        st.image(image_np, caption='Uploaded Image', use_container_width=True)
        st.write(f"**📷 Face Emotion:** `{face_emotion}`")
    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")

# If any input was provided, process the dominant emotion and generate response
if text_input.strip() or audio_file or image_file:
    dominant_emotion = emotion_fusion.fuse_emotions(text_emotion, speech_emotion, face_emotion)
    dominant_emotion = dominant_emotion or "neutral"
    st.session_state.last_emotion = dominant_emotion

    st.write(f"**🧭 Dominant Emotion:** `{dominant_emotion}`")

    prompt = prompt_generator.generate_prompt(dominant_emotion, text_input, st.session_state.chat_history)
    response = llm_client.run(prompt, dominant_emotion)
    suggestion = suggestions.get(dominant_emotion.lower(), "")
    full_response = f"{response}\n\n**🧘 Therapeutic Suggestion:** {suggestion}"

    st.session_state.chat_history.append(("User", text_input))
    st.session_state.chat_history.append(("Bot", full_response))

# Display chat history
def display_chat_history():
    st.subheader("🗨️ Conversation")
    for speaker, message in st.session_state.chat_history:
        if speaker == "User":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Chatbot:** {message}")

display_chat_history()

# Follow-up section
st.subheader("💭 How are you feeling now?")
follow_up_input = st.text_input("You can share a follow-up message:")

if st.button("🔁 Continue Conversation") and follow_up_input.strip():
    follow_up_emotion = text_detector.predict(follow_up_input)
    st.write(f"**📝 Follow-up Text Emotion:** `{follow_up_emotion}`")

    updated_emotion = emotion_fusion.fuse_emotions(follow_up_emotion, None, None) or "neutral"
    st.session_state.last_emotion = updated_emotion

    prompt = prompt_generator.generate_prompt(updated_emotion, follow_up_input, st.session_state.chat_history)
    response = llm_client.run(prompt, updated_emotion)
    suggestion = suggestions.get(updated_emotion.lower(), "")
    full_updated_response = f"{response}\n\n**🧘 Therapeutic Suggestion:** {suggestion}"

    st.session_state.chat_history.append(("User", follow_up_input))
    st.session_state.chat_history.append(("Bot", full_updated_response))

    display_chat_history()
