import torch
from transformers import DistilBertTokenizer
from model_distilbert import DistilbertClassifier

MODEL_PATH = "../src/Modeling/models/distilbert_base-all_tuning.pth"

class ModelHandler:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilbertClassifier("distilbert-base-uncased", 2)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()

    def predict(self, text: str):
        inputs = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'].flatten()
        attention_masks = inputs['attention_mask'].flatten()

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_masks)
            probs = torch.nn.functional.softmax(outputs, dim=-1)
            label = torch.argmax(probs, dim=-1).item()
            confidence = torch.max(probs).item()
            return label, round(confidence, 3)

# Create a global instance to avoid reloading
model_handler = ModelHandler()