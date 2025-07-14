import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, DistilBertTokenizerFast, get_scheduler

from dataset import NewsDataset
from model_distilbert import DistilbertClassifier


def main(df_filepath, model_savepath,  model_name="distilbert-base-uncased"):
    df = pd.read_csv(df_filepath)
    
    # *********
    #df_new = df.iloc[:100,:] # for testing
    #print(df_new.shape)
    #train_df, test_df = train_test_split(df_new, test_size=0.2, random_state=43, stratify=df_new['Target'])
    # *********

    # Train, Validation and Test splitting of the data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=43, stratify=df['Target'])
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=43, stratify=test_df['Target'])
    #print(train_df.shape, val_df.shape, test_df.shape)

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

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

    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device}")
    #device = "cpu"

    learning_rate = 1e-5 #3e-4
    epochs =  2

    # Calling the model here
    model = DistilbertClassifier(model_name, 2).to(device)

    # 1. We will try to freeze the BERT parameters and see how well it performs
    # For Freezing the whole BERT parameters
    #for param in model.bert.parameters():
    #    param.requires_grad = False  # Freeze BERT
    
    # 2. Unfreeze the whole BERT (by default)
    # just mentioning here explicitly
    for param in model.distilbert.parameters():
        param.requires_grad = True  # Unfreeze BERT

    """
    # 3. For unfreezing the top layers and let it change the weights
    # This BERT model has 12 encoder layers and we will freeze the last 2 layers here
    for name, param in model.bert.named_parameters():
        if name.startswith("encoder.layer.10") or name.startswith("encoder.layer.11"):
            param.requires_grad = True # Unfreeze
        else:
            param.requires_grad = False # Freeze
    """
    # defining the loss function and optimizer 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # defining the scheduler to stabilize the training here:
    # setting the warm up:
    # Linear sheduler will increase the learning from rate from zero to the value set up (during the warmup) and decreases there after during training
    # each step is a batch here
    total_steps = len(train_dataloader)*epochs
    scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=50, num_training_steps=total_steps)

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    # Train the model here
    for e in range(epochs):
        model.train()
        train_running_loss = 0
        for input_ids, attention_masks, labels in tqdm(train_dataloader):
            # collect the data in each batch 
            # loading all the data to device
            #print(device)

            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)

            # feed forward part of the model
            # This will basically output the logits [batch, class]
            ypred = model(input_ids=input_ids, attention_mask=attention_masks)

            # zeros the gradients in each batch, otherwise they will accumulate. Because gradient needs to be calculate wrt to the loss of each batch.
            # zeroes the params.grad field
            optimizer.zero_grad()

            #calculate the cross entropy loss
            loss = criterion(ypred, labels) 

            train_running_loss += loss.item() # converting torch to scalar

            # backpropogation
            loss.backward() # calculate the gradient of the loss wrt each parameter and update the parameter.grad field
            
            # Update the weights, current parameter - (learning rate * parameter.grad)
            optimizer.step()

            # update the learning rate for each training step
            # lr  = lr * (step/total_steps) during warmup or lr*(1- step/total_steps) after warm up
            scheduler.step()

            # delete unnecessary data in memory
            del input_ids, attention_masks, ypred, loss

            # empty unused cache
            torch.mps.empty_cache()

        train_loss = train_running_loss/len(train_dataloader)
        train_losses.append(train_loss)


        # Set the model to the evaluatio mode
        model.eval()
        val_running_loss = 0
        with torch.no_grad(): # no need to create the compuatational graph
            for input_ids, attention_masks, labels in tqdm(val_dataloader):
                # collect the data in each batch 
                # loading all the data to device
                # Whenever you load the data or model, load them directly onto the device. Do not initialize them locally in cpu memory and then load it to the 
                # GPU memory. That causes lots of confusions. Then it look like the data is not initialized in the memory.
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)
        
                # feed forward part of the model
                # This will basically output the logits [batch, class]
                ypred = model(input_ids=input_ids, attention_mask=attention_masks)
        
        
                #calculate the binary cross entropy loss
                loss = criterion(ypred, labels) 
        
                val_running_loss += loss.item() # converting torch to scalar

                del input_ids, attention_masks, ypred

                # empty unused cache
                torch.mps.empty_cache()
            
            val_loss = val_running_loss/len(val_dataloader)
            val_losses.append(val_loss)

        # print statistics across each epoch
        print("*"*30)
        print(f"Epoch: {e}")
        print(f"Train loss: {train_loss: 0.4f}")
        print(f"Validation loss: {val_loss: 0.4f}")
        print("*"*30)

    # save the history as a data frame
    df = pd.DataFrame(data={"train_loss":train_losses, "val_loss":val_losses})
    df.to_csv(model_savepath+"hist_distilbert_base-all_tuning.csv")

    #save pytorch model
    torch.save(model.state_dict(), model_savepath+'distilbert_base-all_tuning.pth')
    #model_scripted = torch.jit.script(model) # Export to TorchScript
    #model_scripted.save(model_savepath+'bert_base_uncased.pt') # Save

if __name__ == "__main__":
    filepath = "/Users/savin/Omdena-Projects/Kitwe-News/data/data-final-cleaned-llm.csv"
    model_savepath = "models/"
    main(filepath, model_savepath)
