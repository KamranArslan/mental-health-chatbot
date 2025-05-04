import os
import gdown
from transformers import AutoTokenizer
import torch
from safetensors.torch import load_file  # for loading safetensors format

class TextEmotionDetector:
    def __init__(self, model_dir="models/text_model"):
        # Google Drive file ID for the model
        file_id = "1_fpbFoc22N_CGPKyMjrn-h7qiGfuMtW7"
        output_path = os.path.join(model_dir, "model.safetensors")
        os.makedirs(model_dir, exist_ok=True)

        # Download model if not already present
        if not os.path.exists(output_path):
            print("Downloading model from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)

        # Force CPU for compatibility (no GPU usage)
        self.device = torch.device("cpu")

        # Load tokenizer (ensure you have the correct tokenizer files in model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Load model using safetensors (instead of from_pretrained, we use safetensors for the custom format)
        model_path = os.path.join(model_dir, "model.safetensors")
        self.model = load_file(model_path)  # Load model with safetensors
        self.model.to(self.device)  # Move model to CPU
        self.model.eval()  # Set the model to evaluation mode

        # Emotion labels (modify if your model has different labels)
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

    def predict(self, text):
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move inputs to CPU

        # Inference (no gradient computation needed)
        with torch.no_grad():
            outputs = self.model(**inputs)  # Get model predictions
            predictions = torch.softmax(outputs.logits, dim=1)  # Apply softmax to get probabilities
            predicted_class = torch.argmax(predictions, dim=1).item()  # Get the index of the highest probability

        # Return the predicted emotion label
        return self.emotions[predicted_class]

