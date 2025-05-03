import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, AIMessage

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

        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Prompt template including dominant emotion and message history
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a supportive and empathetic mental health chatbot. "
                "You provide therapeutic responses based on the user's emotional state. "
                "Respond with warmth and clarity, offering gentle encouragement, validation, and practical coping strategies when appropriate. "
                "Avoid giving medical advice or diagnoses. Keep your responses brief, supportive, and focused on helping the user feel heard and understood. "
                "The user's dominant emotion is: {dominant_emotion}"
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Initialize the LangChain LLMChain
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
            # Add dominant emotion to system memory for better context
            self.memory.chat_memory.add_message(
                SystemMessage(content=f"Dominant emotion: {dominant_emotion}")
            )

            logger.info("Running with input: '%s' | emotion: '%s'", user_input, dominant_emotion)

            # Run the chain
            response = self.chain.invoke({
                "input": user_input,
                "dominant_emotion": dominant_emotion
            })

            # Handle different possible response types
            if isinstance(response, str):
                return response
            elif isinstance(response, dict):
                return response.get("text", "I'm here for you.")
            elif isinstance(response, AIMessage):
                return response.content
            else:
                return str(response)

        except Exception as e:
            logger.error("Error in LangChainClient.run: %s", str(e))
            return (
                "I'm sorry, I'm having trouble generating a response right now. "
                "Please try again shortly."
            )

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
