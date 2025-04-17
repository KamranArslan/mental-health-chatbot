# AI Therapy Chatbot

A multimodal therapy chatbot that analyzes emotions through text, speech, and facial expressions to provide therapeutic responses.

## Features

- Text emotion detection using BERT
- Speech emotion recognition using Keras
- Facial expression recognition using Keras
- Emotion fusion using majority voting
- Therapeutic response generation using Groq's Llama model with LangChain
- Conversation memory to maintain context across messages
- Modern web interface built with Flask and HTML/CSS/JavaScript

## Setup

1. Make a new conda environment:
conda create -n chatbotenv python=3.11
conda activate chatbotenv

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your Groq API key(alaready added):
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
4. Place your model files in the appropriate directories(already placed):
   - BERT text model in `models/text_model/`
   - Speech model in `models/speech_model.h5`
   - Face model in `models/face_model.h5`

## Usage

1. Start the Flask app:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to `http://localhost:5000`
3. Enter text, record audio, or take a photo (you can provide any combination of inputs)
4. Click "Send" to get a therapeutic response
   - If only one input is provided, the response will be based on that input's emotion
   - If multiple inputs are provided, the response will be based on the majority emotion
   - The chatbot will remember previous messages to provide more contextual responses

## Project Structure

```
.
├── app.py                  # Main Flask application
├── templates/
│   └── index.html          # Frontend HTML template
├── models/
│   ├── text_emotion.py     # Text emotion detection
│   ├── speech_emotion.py   # Speech emotion detection
│   ├── face_emotion.py     # Facial expression recognition
│   ├── emotion_fusion.py   # Emotion fusion logic
│   ├── prompt_generator.py # Therapeutic prompt generation
│   └── langchain_client.py # LangChain integration with Groq
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Requirements

- Python 3.8+
- TensorFlow 2.12.0+
- PyTorch 2.0.0+
- Flask 2.0.0+
- Hugging Face Transformers 4.30.0+
- Groq 0.3.0+
- LangChain 0.1.0+
- Other dependencies listed in requirements.txt

## License

MIT License 