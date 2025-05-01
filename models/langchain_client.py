import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain

class LangChainClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Initialize the Groq LLM
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
        
        # Create a chat prompt template that includes the conversation history and dominant emotion
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a supportive and empathetic mental health chatbot. "
                      "You provide therapeutic responses based on the user's emotional state. "
                      "Respond with warmth and clarity, offering gentle encouragement, validation, and practical coping strategies when appropriate. "
                      "Avoid giving medical advice or diagnoses. Keep your responses brief, supportive, and focused on helping the user feel heard and understood. "
                      "The user's dominant emotion is: {dominant_emotion}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Create a chain that combines the prompt, memory, and LLM
        self.chain = LLMChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt
        )
    
    def run(self, user_input, dominant_emotion=None):
        """
        Process user input through the LangChain chain with memory.
        
        Args:
            user_input (str): The user's message
            dominant_emotion (str): The dominant emotion detected (optional)
            
        Returns:
            str: The AI's response
        """
        try:
            if dominant_emotion:
                # Update the conversation history with the emotional context
                self.memory.chat_memory.add_message("system", f"Dominant emotion: {dominant_emotion}")

            # Run the chain with the user input
            response = self.chain.invoke({"input": user_input})
            return response["text"]
        except Exception as e:
            return f"I apologize, but I'm having trouble generating a response right now. Error: {str(e)}"
    
    def get_conversation_history(self):
        """
        Return the conversation history for debugging or display purposes.
        
        Returns:
            list: The conversation history as a list of messages
        """
        return self.memory.chat_memory.messages
