class PromptGenerator:
    def __init__(self):
        self.emotion_prompts = {
            "anger": (
                "I notice you're feeling angry. Would you like to talk about what's causing this anger? "
                "It's completely normal to feel this way, and I'm here to listen.\n\n"
                "👉 A helpful technique might be to take a few deep breaths or write down what's upsetting you. "
                "Would you like to try that?"
            ),
            "disgust": (
                "I sense that you're feeling disgusted. This can be a challenging emotion to process. "
                "Would you like to explore what triggered this feeling?\n\n"
                "👉 Sometimes stepping back or focusing on something pleasant can help reduce this feeling. "
                "Would you like a distraction or to process it more deeply?"
            ),
            "fear": (
                "I can see that you're feeling afraid. Fear is a natural response to perceived threats. "
                "Would you like to discuss what's making you feel this way?\n\n"
                "👉 Try grounding yourself by naming 3 things you see, 2 things you hear, and 1 thing you feel. "
                "This can help calm anxiety. Would you like me to guide you through this?"
            ),
            "happiness": (
                "I'm glad to see you're feeling happy! Would you like to share what's bringing you joy? "
                "Celebrating positive moments can help build resilience.\n\n"
                "👉 How can we keep this good feeling going? Maybe plan something fun or reflect on what made it meaningful?"
            ),
            "neutral": (
                "I notice you seem neutral right now. How are you feeling about things in general? "
                "Sometimes taking a moment to reflect can be helpful.\n\n"
                "👉 Would you like to explore how your day has been so far, or check in on your goals?"
            ),
            "sadness": (
                "I can see that you're feeling sad. It's okay to feel this way, and you don't have to go through it alone. "
                "Would you like to talk about what's troubling you?\n\n"
                "👉 It might help to write a letter to yourself or take a short walk to clear your mind. "
                "Would you like some comforting words or a distraction?"
            ),
            "surprise": (
                "I notice you seem surprised. This can be both positive and challenging. "
                "Would you like to talk about what caught you off guard?\n\n"
                "👉 It may help to take a moment to process this. How are you feeling about it now?"
            )
        }

    def generate_prompt(self, emotion, user_input="", previous_conversation=""):
        """
        Generate a prompt incorporating the detected emotion, user input, and conversation history
        to send to the LangChain client for response generation.

        Args:
            emotion (str): The detected dominant emotion.
            user_input (str): The user's message (text, speech transcription, or summary of image input).
            previous_conversation (str): The prior conversation history to maintain context.

        Returns:
            str: A context-aware, empathetic prompt tailored for LangChain.
        """
        # Use neutral prompt if emotion not found
        base_prompt = self.emotion_prompts.get(emotion, self.emotion_prompts["neutral"])

        if previous_conversation:
            return f"{base_prompt}\n\n{previous_conversation}\n\nUser: {user_input}"

        if user_input:
            return f"{base_prompt}\n\nUser: {user_input}"

        return base_prompt
