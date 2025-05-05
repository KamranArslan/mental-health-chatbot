import os
import gdown
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TextEmotionDetector:
    def __init__(self, model_dir="models/text_model"):
        # Path for downloading the model
        file_id = "1_fpbFoc22N_CGPKyMjrn-h7qiGfuMtW7"
        model_file = os.path.join(model_dir, "model.safetensors")
        os.makedirs(model_dir, exist_ok=True)

        # Download the model if not already present
        if not os.path.exists(model_file):
            print("Downloading model from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_file, quiet=False)

        # Ensure we're using CPU
        self.device = torch.device("cpu")

        # Check if model file exists
        print(f"Checking if model file exists at: {model_file}")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found at {model_file}")

        # ✅ Load the model using transformers (auto-detect safetensors!)
        print(f"Loading model from directory: {model_dir}")
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Emotion labels
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()

        return self.emotions[predicted_class]
