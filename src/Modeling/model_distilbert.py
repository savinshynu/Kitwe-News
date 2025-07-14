from torch import nn
from transformers import DistilBertForSequenceClassification

# Lets define the model class
class DistilbertClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        # initiating a direct classification head here, so additional linear layers are not needed
        self.distilbert = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes) # The model will output would be [batch_size, sequence_length, hidden_states]

    def forward(self, input_ids, attention_mask):
        # This model is alredy using the pooler of the CLS token and the final linear layer for classification and it outputs the logits.
        logits = self.distilbert(input_ids=input_ids, attention_mask=attention_mask).logits # [batch_size, sequence_length, hidden_states]
        return logits