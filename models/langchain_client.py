import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainClient:
    """
    A chatbot client using LangChain and Groq API for supportive mental health interactions.
    """

    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        # Initialize Groq LLM
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=500,
            top_p=0.9
        )

        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # ✅ FIX: Explicitly specify input variables
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", (
                    "You are a supportive and empathetic mental health chatbot. "
                    "You provide therapeutic responses based on the user's emotional state. "
                    "Respond with warmth and clarity, offering gentle encouragement, validation, and practical coping strategies when appropriate. "
                    "Avoid giving medical advice or diagnoses. Keep your responses brief, supportive, and focused on helping the user feel heard and understood. "
                    "The user's dominant emotion is: {dominant_emotion}"
                )),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ],
            input_variables=["input", "dominant_emotion"]  # ✅ THIS LINE FIXES THE ERROR
        )

        # Create chain
        self.chain = LLMChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt
        )

    def run(self, user_input: str, dominant_emotion: str = "neutral") -> str:
        """
        Process user input through the LangChain chain.

        Args:
            user_input (str): The user's message.
            dominant_emotion (str): The detected dominant emotion (default is "neutral").

        Returns:
            str: The AI-generated supportive response.
        """
        try:
            logger.info("Running with input: '%s' | emotion: '%s'", user_input, dominant_emotion)
            response = self.chain.invoke({
                "input": user_input,
                "dominant_emotion": dominant_emotion
            })

            if isinstance(response, dict):
                return response.get("text", "I'm here for you.").strip()
            elif isinstance(response, str):
                return response.strip()
            elif isinstance(response, AIMessage):
                return response.content.strip()
            else:
                logger.warning("Unexpected response type: %s", type(response))
                return "I'm here for you. Please share more with me."
        except Exception as e:
            logger.error("❌ Error in LangChainClient.run: %s", str(e))
            return f"Error occurred: {str(e)}"

    def get_conversation_history(self) -> list:
        """
        Return the conversation history.

        Returns:
            list: List of formatted conversation messages.
        """
        formatted_history = []
        for message in self.memory.chat_memory.messages:
            role = getattr(message, 'type', 'unknown')
            content = getattr(message, 'content', '')
            formatted_history.append(f"{role.capitalize()}: {content}")
        return formatted_history
