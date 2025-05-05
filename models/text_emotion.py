import os
import torch
import gdown
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class TextEmotionDetector:
    def __init__(self, model_dir="models/text_model", folder_id="16kzp5dCqddSM6nCZjZOF7fqSONwTKPyV"):
        os.makedirs(model_dir, exist_ok=True)

        # List of files (replace these with your actual file IDs and names)
        self.files = {
            'model.safetensors': '1_fpbFoc22N_CGPKyMjrn-h7qiGfuMtW7',
            'config.json': '17YZajDP7TodcX45jT9gFBFYzJwLIl2ty',
            'tokenizer_config.json': '1edkqaA_lnpyocyyK__F7rVHvSvZcgeMg',
            'vocab.txt': '1r3PVi7jMxD99A61vnyBhT0sfrRiX-I7s',
            'special_tokens_map.json': '1BtE2B07sy60fFkTuleyqCEY6IKwZsVK4'
        }

        # Download files from Google Drive without authentication
        for filename, fileid in self.files.items():
            file_path = os.path.join(model_dir, filename)
            if not os.path.exists(file_path):
                print(f"Downloading {filename} from Google Drive...")
                url = f"https://drive.google.com/uc?id={fileid}"
                gdown.download(url, file_path, quiet=False)

        # Set the device to CPU explicitly
        self.device = torch.device("cpu")  # Ensures only CPU is used

        # Check if model and tokenizer files are downloaded
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        self.model.to(self.device)  # Move the model to CPU (ensures it's on the right device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Emotion labels
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

    def predict(self, text):
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        # Move inputs to the correct device (CPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict emotion
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()

        return self.emotions[predicted_class]
