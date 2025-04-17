import os
from groq import Groq
from dotenv import load_dotenv

class LLMClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=self.api_key)
    
    def generate_response(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.5,
                max_completion_tokens=300,
                top_p=0.9,
                stream=True
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            return f"I apologize, but I'm having trouble generating a response right now. Error: {str(e)}" 