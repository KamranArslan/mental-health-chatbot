import os
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class TextEmotionDetector:
    def __init__(self, model_dir="models/text_model"):
        # Google Drive file ID
        file_id = "1_fpbFoc22N_CGPKyMjrn-h7qiGfuMtW7"
        output_path = os.path.join(model_dir, "model.safetensors")
        os.makedirs(model_dir, exist_ok=True)

        # Download model if not already present
        if not os.path.exists(output_path):
            print("Downloading model from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)
        
        # Force CPU for compatibility
        self.device = torch.device("cpu")

        # Load tokenizer and model from the downloaded folder
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

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
