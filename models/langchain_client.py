from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.schema import AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import logging

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("❌ GROQ_API_KEY is missing from environment variables.")

        # ✅ Load model
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name="llama-3-70b-8192",  # Double-check model name on Groq
            temperature=0.7,
            max_tokens=500,
            top_p=0.9
        )

        # ✅ Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # ✅ Define prompt template CORRECTLY with input variables
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

        # ✅ Define input variables here explicitly
        self.chain = LLMChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt,
            input_variables=["input", "dominant_emotion"]  # ✅ Required!
        )

    def run(self, user_input: str, dominant_emotion: str = "neutral") -> str:
        try:
            logger.info("Processing input: %s (emotion: %s)", user_input, dominant_emotion)
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
                return "I'm here for you. Please share more with me."
        except Exception as e:
            logger.error("❌ Error in LangChainClient.run: %s", str(e))
            return f"Error occurred: {str(e)}"

    def get_conversation_history(self):
        formatted = []
        for msg in self.memory.chat_memory.messages:
            role = getattr(msg, 'type', 'unknown')
            content = getattr(msg, 'content', '')
            formatted.append(f"{role.capitalize()}: {content}")
        return formatted
