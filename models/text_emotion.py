import os
import torch
import gdown
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class TextEmotionDetector:
    def __init__(self, model_dir="models/text_model", folder_id="16kzp5dCqddSM6nCZjZOF7fqSONwTKPyV"):
        os.makedirs(model_dir, exist_ok=True)

        self.files = {
            'model.safetensors': '1_fpbFoc22N_CGPKyMjrn-h7qiGfuMtW7',
            'config.json': '17YZajDP7TodcX45jT9gFBFYzJwLIl2ty',
            'tokenizer_config.json': '1edkqaA_lnpyocyyK__F7rVHvSvZcgeMg',
            'vocab.txt': '1r3PVi7jMxD99A61vnyBhT0sfrRiX-I7s',
            'special_tokens_map.json': '1BtE2B07sy60fFkTuleyqCEY6IKwZsVK4'
        }

        # Download files from Google Drive
        for filename, fileid in self.files.items():
            file_path = os.path.join(model_dir, filename)
            if not os.path.exists(file_path) or (filename == 'model.safetensors' and os.path.getsize(file_path) < 100 * 1024 * 1024):  # Check for small model file
                print(f"Downloading {filename} from Google Drive...")
                url = f"https://drive.google.com/uc?id={fileid}"
                try:
                    gdown.download(url, file_path, quiet=False)
                except Exception as e:
                    print(f"Failed to download {filename}: {e}")
                    raise

        # Verify file sizes
        print("File sizes:")
        for filename in self.files:
            file_path = os.path.join(model_dir, filename)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"  {filename}: {file_size:.2f} MB")

        # Set the device to CPU
        self.device = torch.device("cpu")

        # Load model
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
            self.model.to(self.device)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            print("Tokenizer loaded successfully")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise

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
