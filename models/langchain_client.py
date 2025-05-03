import os
import logging
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

# Logging configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LangChainClient:
    def __init__(self):
        logger.info("Initializing LangChainClient...")

        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            logger.error("❌ GROQ_API_KEY is missing from environment variables.")
            raise ValueError("❌ GROQ_API_KEY is missing from environment variables.")
        else:
            logger.info("✅ GROQ_API_KEY found in environment variables.")

        # Initialize conversation history as a list of message objects
        self.history = []

        # Define prompt template with system message, history, and user input
        try:
            logger.info("Defining prompt template...")
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", (
                        "You are a supportive and empathetic mental health chatbot. "
                        "Provide brief, calming responses. Do not give medical advice. "
                        "The user's dominant emotion is: {dominant_emotion}"
                    )),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}")
                ]
            )
            logger.info("✅ Prompt template defined.")
        except Exception as e:
            logger.error(f"❌ Error defining prompt template: {e}")
            raise

        # Initialize the language model
        try:
            logger.info("Loading LLM model...")
            self.llm = ChatGroq(
                api_key=self.api_key,
                model_name="llama3-70b-8192",
                temperature=0.7,
                max_tokens=500,
                model_kwargs={"top_p": 0.9}
            )
            logger.info("✅ Model loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Error loading LLM model: {e}")
            raise

        # Initialize LLMChain without memory
        try:
            logger.info("Initializing LLMChain...")
            self.chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt,
                verbose=True
            )
            logger.info("✅ LLMChain initialized successfully.")
        except Exception as e:
            logger.error(f"❌ Error initializing LLMChain: {e}")
            raise

    def run(self, user_input: str, dominant_emotion: str = "neutral") -> str:
        try:
            logger.info("Processing user input: %s (emotion: %s)", user_input, dominant_emotion)
            logger.debug("User input type: %s", type(user_input))
            logger.debug("Dominant emotion type: %s", type(dominant_emotion))

            # Invoke the chain with current input, emotion, and history
            response = self.chain.invoke(
                {
                    "input": user_input,
                    "dominant_emotion": dominant_emotion,
                    "history": self.history
                }
            )

            logger.debug("Response received: %s", response)
            # Extract the text from the response dictionary
            bot_response = response.get("text", "").strip()

            # Update conversation history
            self.history.append(HumanMessage(content=user_input))
            self.history.append(AIMessage(content=bot_response))

            # Limit history to last 10 messages (5 exchanges)
            if len(self.history) > 10:
                self.history = self.history[-10:]

            return bot_response
        except Exception as e:
            logger.error("❌ Error in LangChainClient.run: %s", str(e))
            return f"Error occurred: {str(e)}"

    def get_conversation_history(self):
        logger.info("Retrieving conversation history...")
        formatted = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                formatted.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"Assistant: {msg.content}")
        logger.debug("Conversation history: %s", formatted)
        return formatted
