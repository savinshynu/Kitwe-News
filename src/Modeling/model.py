from torch import nn
from transformers import BertModel

# Lets define the model class
class BertClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name) # The model will output would be [batch_size, sequence_length, hidden_states]
        # hidden_states = 768 for the base model
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask) # [batch_size, sequence_length, hidden_states]
        # Now we need to collect the hidden state of CLS (first) token
        # cls_hidden_state = output[:,0,:] # calling the first token
        # pooler output = tanh(dense(cls_hidden_state))
        pooler = output.pooler_output
        drop = self.dropout(pooler)
        logits = self.fc(drop)
        return logits