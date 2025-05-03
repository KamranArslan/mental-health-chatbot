from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.schema import AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import logging

# Logging config
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logs
logger = logging.getLogger(__name__)

class LangChainClient:
    def __init__(self):
        logger.info("Initializing LangChainClient...")
        
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")

        # Check if the API key is missing
        if not self.api_key:
            logger.error("❌ GROQ_API_KEY is missing from environment variables.")
            raise ValueError("❌ GROQ_API_KEY is missing from environment variables.")
        else:
            logger.info("✅ GROQ_API_KEY found in environment variables.")

        try:
            # ✅ Load model
            logger.info("Loading LLM model...")
            self.llm = ChatGroq(
                api_key=self.api_key,
                model_name="llama-3-70b-8192",  # Double-check model name on Groq
                temperature=0.7,
                max_tokens=500,
                top_p=0.9
            )
            logger.info("✅ Model loaded successfully.")

            # ✅ Initialize memory
            logger.info("Initializing memory...")
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            logger.info("✅ Memory initialized.")

            # ✅ Define prompt template
            logger.info("Defining prompt template...")
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", (
                        "You are a supportive and empathetic mental health chatbot. "
                        "Provide brief, calming responses. Do not give medical advice. "
                        "The user's dominant emotion is: {dominant_emotion}"
                    )),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ]
            )
            logger.info("✅ Prompt template defined successfully.")

            # ✅ Define the chain
            logger.info("Initializing LLMChain...")
            self.chain = LLMChain(
                llm=self.llm,
                memory=self.memory,
                prompt=self.prompt,
                input_variables=["input", "dominant_emotion"]  # ✅ Required!
            )
            logger.info("✅ LLMChain initialized successfully.")
        except Exception as e:
            logger.error(f"❌ Error during initialization: {e}")
            raise

    def run(self, user_input: str, dominant_emotion: str = "neutral") -> str:
        try:
            logger.info("Processing user input: %s (emotion: %s)", user_input, dominant_emotion)

            # Debugging: Check the types of user input and emotion
            logger.debug("User input type: %s", type(user_input))
            logger.debug("Dominant emotion type: %s", type(dominant_emotion))

            # Attempt to invoke the LLM chain
            response = self.chain.invoke({
                "input": user_input,
                "dominant_emotion": dominant_emotion
            })

            # Inspect the structure of the response
            logger.debug("Response received: %s", response)
            if isinstance(response, dict):
                logger.debug("Response type is dict, extracting text.")
                return response.get("text", "I'm here for you.").strip()
            elif isinstance(response, str):
                logger.debug("Response type is string.")
                return response.strip()
            elif isinstance(response, AIMessage):
                logger.debug("Response type is AIMessage.")
                return response.content.strip()
            else:
                logger.warning("Unknown response type: %s", type(response))
                return "I'm here for you. Please share more with me."
        except Exception as e:
            logger.error("❌ Error in LangChainClient.run: %s", str(e))
            return f"Error occurred: {str(e)}"

    def get_conversation_history(self):
        logger.info("Retrieving conversation history...")
        formatted = []
        for msg in self.memory.chat_memory.messages:
            role = getattr(msg, 'type', 'unknown')
            content = getattr(msg, 'content', '')
            formatted.append(f"{role.capitalize()}: {content}")
        
        logger.debug("Conversation history: %s", formatted)
        return formatted
