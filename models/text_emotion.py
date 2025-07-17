import os
import torch
import gdown
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import safetensors  # Ensure installed: pip install safetensors


class TextEmotionDetector:
    def __init__(self, model_dir="models/text_model"):
        os.makedirs(model_dir, exist_ok=True)

        self.files = {
            'model.safetensors': '1_fpbFoc22N_CGPKyMjrn-h7qiGfuMtW7',
            'config.json': '17YZajDP7TodcX45jT9gFBFYzJwLIl2ty',
            'tokenizer_config.json': '1edkqaA_lnpyocyyK__F7rVHvSvZcgeMg',
            'vocab.txt': '1r3PVi7jMxD99A61vnyBhT0sfrRiX-I7s',
            'special_tokens_map.json': '1BtE2B07sy60fFkTuleyqCEY6IKwZsVK4'
        }

        # Download files if not present or if model file looks too small
        for filename, fileid in self.files.items():
            file_path = os.path.join(model_dir, filename)
            if not os.path.exists(file_path) or (
                filename == 'model.safetensors' and os.path.getsize(file_path) < 100 * 1024 * 1024
            ):
                print(f"📥 Downloading {filename}...")
                url = f"https://drive.google.com/uc?id={fileid}"
                try:
                    gdown.download(url, file_path, quiet=False)
                except Exception as e:
                    print(f"❌ Failed to download {filename}: {e}")
                    raise

        # Check file sizes
        print("\n✅ Downloaded file sizes:")
        for filename in self.files:
            path = os.path.join(model_dir, filename)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {filename}: {size_mb:.2f} MB")

        # Set device (only CPU in this example)
        self.device = torch.device("cpu")

        # Load model (safetensors)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=True,
                use_safetensors=True  # Important for safetensors loading
            )
            self.model.to(self.device)
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=True
            )
            print("✅ Tokenizer loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            raise

        # Emotion labels
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

    def predict(self, text):
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")

        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        return {
            "text": text,
            "predicted_emotion": self.emotions[pred_class],
            "confidence": float(probs[0][pred_class]),
            "all_probabilities": {emotion: float(p) for emotion, p in zip(self.emotions, probs[0])}
        }
