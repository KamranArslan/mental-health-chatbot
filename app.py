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

# --- Initialize session state ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'last_emotion' not in st.session_state:
    st.session_state.last_emotion = None

# Input widgets
text_input = st.text_input("Enter your message:")
audio_file = st.file_uploader("Upload an audio file (optional)", type=["mp3", "wav"])
image_file = st.file_uploader("Upload an image (optional)", type=["jpg", "png"])

# Emotion detection
text_emotion = None
speech_emotion = None
face_emotion = None

if text_input:
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
        st.image(image_np, caption='Uploaded Image', use_container_width=True)
        st.write(f"Detected emotion from face: {face_emotion}")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Fuse emotions and respond
if text_input or audio_file or image_file:
    dominant_emotion = emotion_fusion.fuse_emotions(text_emotion, speech_emotion, face_emotion)
    st.session_state.last_emotion = dominant_emotion
    st.write(f"Dominant emotion: {dominant_emotion}")

    # Therapeutic suggestions dictionary
  suggestions = {
    "anger": "Take a deep breath and count to ten. Step away from the situation if needed.",
    "sadness": "You're not alone. Reach out to someone you trust or write down your thoughts.",
    "fear": "Try grounding techniques like 5-4-3-2-1 to stay present.",
    "happiness": "Celebrate your joy! Share your moment with someone close.",
    "disgust": "Pause and reflect. Try to reframe or distance from the triggering situation.",
    "surprise": "Take time to process what happened. Talk it out if needed.",
    "neutral": "Stay mindful. Even neutral states can offer moments of clarity.",
    "anxiety": "Breathe in for 4, hold for 4, out for 4. Repeat. You're in control.",
    "confusion": "Break things down into smaller pieces. It's okay to ask questions.",
    "loneliness": "Reach out—even a small interaction can help. You matter.",
    "hope": "Keep feeding that hope with small steps forward. You're doing great.",
    "boredom": "Try something creative or new—a short walk, journaling, or music.",
    "guilt": "Forgive yourself. Learn from it and let it go—you deserve compassion too.",
    "shame": "You're human, and flaws don't define you. Speak kindly to yourself.",
    "excitement": "Channel this into something productive or expressive. Embrace the spark.",
    "pride": "Be proud of your accomplishments—share them or reflect on your growth.",
    "jealousy": "Acknowledge it without judgment. Use it to inspire growth.",
    "embarrassment": "It’s okay—everyone messes up. Laugh it off and move on.",
    "relief": "Enjoy this moment of ease. Take time to recharge.",
    "insecurity": "You are enough. Focus on your strengths, not your doubts.",
    "grief": "Let yourself feel. Grieving is not linear, and healing takes time.",
    "overwhelmed": "Make a simple list. Focus on one thing at a time. Breathe.",
    "burnout": "Rest is productive. You don’t have to earn a break—you need it.",
    "trust": "Let trust grow naturally. Keep communicating clearly.",
    "distrust": "Protect your boundaries, but remain open to healing connections.",
    "love": "Share your feelings or express them in a kind gesture. Love grows when shared.",
    "frustration": "Step back, take a break, then return with a fresh mind.",
    "inspiration": "Use this energy to create or plan something meaningful.",
    "curiosity": "Explore safely and ask questions—learning is a great path to peace.",
    "emptiness": "Try reconnecting with activities or people that bring meaning.",
}

    # Generate prompt with emotional context and conversation history
    prompt = prompt_generator.generate_prompt(dominant_emotion, text_input)

    # Send to LLM client with memory
    response = llm_client.run(prompt)

    # Combine LLM response with coping suggestion
    suggestion = suggestions.get(dominant_emotion.lower(), "")
    full_response = f"{response}\n Therapeutic Suggestion: {suggestion}"

    # Update chat history
    st.session_state.chat_history.append(("User", text_input))
    st.session_state.chat_history.append(("Bot", full_response))

# Display conversation history
st.subheader("Conversation")
for speaker, message in st.session_state.chat_history:
    if speaker == "User":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Chatbot:** {message}")
