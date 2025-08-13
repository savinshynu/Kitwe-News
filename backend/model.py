import torch
import pandas as pd
from transformers import DistilBertTokenizer
from model_distilbert import DistilbertClassifier
from typing import List

MODEL_PATH = "models/distilbert_base-all_tuning.pth"

class ModelHandler:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilbertClassifier("distilbert-base-uncased", 2)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()

    def predict(self, text: List[str]):
        inputs = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs) # output logits
            probs = torch.nn.functional.softmax(outputs, dim=1) # applying softmax to get the probs
            confidence, label = torch.max(probs, dim=1) # output of torch.max is (max value, indices)
            return label.tolist(), confidence.tolist()

# Create a global instance to avoid reloading
model_handler = ModelHandler()

if __name__ == "__main__":
    import requests
    df = pd.read_csv('/home/savin/Omdena/Kitwe-News/data/raw-new.csv')
    df_test = df.iloc[:5,:]
    df_inp = (df_test['Headline'] + " " + df_test['Description']).astype('str')
    #lab, conf = model_handler.predict(df_inp.tolist())
    #print(lab)
    #print(conf)
    response = requests.post("http://localhost:8000/predict", json={'texts':df_inp.tolist()})
    #response = requests.get("http://localhost:8000")
    print(response.status_code)
    print(response.json())
