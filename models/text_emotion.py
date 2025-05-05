import os
import gdown
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

class TextEmotionDetector:
    def __init__(self, model_dir="models/text_model", folder_id="16kzp5dCqddSM6nCZjZOF7fqSONwTKPyV"):
        os.makedirs(model_dir, exist_ok=True)

        # Google Drive API setup
        self.folder_id = folder_id
        self.service = self.authenticate_drive()

        # Get list of file ids in the folder
        self.files = self.list_files_in_folder(self.folder_id)

        # Download files if they don't exist
        for filename, fileid in self.files.items():
            file_path = os.path.join(model_dir, filename)
            if not os.path.exists(file_path):
                print(f"Downloading {filename} from Google Drive...")
                url = f"https://drive.google.com/uc?id={fileid}"
                gdown.download(url, file_path, quiet=False)

        # Path to the model file
        model_file = os.path.join(model_dir, "model.safetensors")

        # Ensure we're using CPU
        self.device = torch.device("cpu")

        # Check if model file exists
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found at {model_file}")

        # Load the model using transformers (auto-detect safetensors)
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

    def authenticate_drive(self):
        """Authenticate with Google Drive API"""
        SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        return build('drive', 'v3', credentials=creds)

    def list_files_in_folder(self, folder_id):
        """List all files in the specified Google Drive folder"""
        query = f"'{folder_id}' in parents"
        results = self.service.files().list(q=query).execute()
        items = results.get('files', [])

        files = {}
        for item in items:
            files[item['name']] = item['id']
        return files

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()

        return self.emotions[predicted_class]
