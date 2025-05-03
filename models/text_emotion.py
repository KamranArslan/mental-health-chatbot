from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class TextEmotionDetector:
    def __init__(self, model_path="models/text_model"):
        # Force CPU for compatibility
        self.device = torch.device("cpu")  # Change to "cuda" if GPU is enabled elsewhere

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Define emotion labels (update if model uses different classes)
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

    def predict(self, text):
        # Tokenize input and move to device
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict emotion
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()

        return self.emotions[predicted_class]
