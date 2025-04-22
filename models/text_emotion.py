from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class TextEmotionDetector:
    def __init__(self, model_path="models/text_model"):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Try moving to device (can raise NotImplementedError in restricted environments)
            test_tensor = torch.tensor([0.0]).to(self.device)
        except (AssertionError, NotImplementedError, RuntimeError):
            # Fallback to CPU if anything fails
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Define emotion labels (adjust based on your model's output)
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()

        return self.emotions[predicted_class]
