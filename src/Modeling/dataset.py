import torch
from torch.utils.data import Dataset


# Let's define the dataset class now
# Make sure the __init_ class has access to all the data
# __getitem__ focusses on a single sample from the data defined in the __init__ class
# The returns of __getitem__ function should be a tensor as well.
# Otherwise collection of data in the batch throws an error


class NewsDataset(Dataset):
    def __init__(self, df, tokenizer):
        super().__init__()
        self.df = df
        self.text = (self.df['Headline'] + " " + self.df['Description']).astype('str')  # pandas dataframe
        #self.text = (self.df['Source'] + self.df['Category'] + self.df['Link'] + self.df['Author']+ self.df['Headline'] + self.df['Description']).astype('str')
        self.labels = self.df['Target']
        self.tokenizer = tokenizer
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        text = self.text.iloc[idx]
        label = self.labels.iloc[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")
        input_id = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()
        return input_id, attention_mask, torch.tensor(label, dtype=torch.int64)