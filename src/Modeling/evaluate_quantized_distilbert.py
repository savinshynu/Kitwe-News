import torch
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from transformers import DistilBertTokenizerFast

from dataset import NewsDataset
from model_distilbert import DistilbertClassifier

filepath = "/Users/savin/Omdena-Projects/Kitwe-News/data/data-final-cleaned-llm.csv"
df = pd.read_csv(filepath)

# Train, Validation and Test splitting of the data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=43, stratify=df['Target'])
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=43, stratify=test_df['Target'])
print(train_df.shape, val_df.shape, test_df.shape)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

## Getting the dataset
## Dataset class allow the retrieval of features and labels one sample at a time
train_dataset = NewsDataset(train_df, tokenizer)
val_dataset = NewsDataset(val_df, tokenizer)
test_dataset = NewsDataset(test_df, tokenizer)

batch_size = 16
#Getting the dataloader
# Dataloader API will shuffle the data before each epoch and returns a minibatch of features and labels mentioned by the label size
# Basically collect several samples and convert that into a minibatch
# Here each batch will have 64 integer ids, their attention mask and class labels
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

# device
device = torch.device('cpu')
print(device)

# Calling the model here
dist_model = DistilbertClassifier("distilbert-base-uncased", 2).to(device)

# quantization step
dist_quant = torch.quantization.quantize_dynamic(
    dist_model,
    {torch.nn.Linear},  # Only quantize linear layers, attention layers are not supported for quantization yet.
    dtype=torch.qint8   # 8-bit integers
)

# load weights into the defined model
dist_quant.load_state_dict(torch.load("models/distilbert_quantized.pth"))

# Testing the results of the quantized model 
# Set the model to the evaluatio mode
dist_quant.eval()


y_out_list = []
y_true_list = []

with torch.no_grad(): # no need to create the compuatational graph
    for input_ids, attention_masks, labels in tqdm(test_dataloader):
        
        print(labels.shape)
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)

        # This will basically output the logits [batch, class]
        ypred = dist_quant(input_ids=input_ids, attention_mask=attention_masks)

        ypred = torch.argmax(ypred, dim=1)
        y_out_list.append(ypred)
        y_true_list.append(labels)

        del input_ids, attention_masks, ypred

y_out_test = torch.cat(y_out_list, dim=0).numpy() # Convert it to numpy array
y_true_test = torch.cat(y_true_list, dim=0).numpy()

print("Test data:")
print(f" Accuracy: {accuracy_score(y_true_test, y_out_test)} \n\
Precision : {precision_score(y_true_test, y_out_test)} \n\
Recall: {recall_score(y_true_test, y_out_test)} \n\
F1-score:{f1_score(y_true_test, y_out_test)}")

print(classification_report(y_true_test, y_out_test, target_names=['Genuine', 'Fake']))