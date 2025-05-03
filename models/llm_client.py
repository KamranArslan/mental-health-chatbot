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
                stream=True  # Stream flag enabled
            )

            # Process the streamed response
            response_content = ""
            for chunk in completion:
                response_content += chunk.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Check if content was retrieved
            if response_content.strip():
                return response_content
            else:
                return "I apologize, but the response was empty. Please try again."
        
        except Exception as e:
            # Log the error to the backend (optional for debugging)
            # You could also log the error to a logging system if needed
            print(f"Error occurred: {str(e)}")
            # Return user-friendly error message
            return f"I apologize, but I'm having trouble generating a response right now. Error: {str(e)}"
