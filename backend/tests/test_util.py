# unit tests for backend
import torch
import pandas as pd
from backend import model_handler

def test_model_handler():
    df = pd.read_csv('../../data/raw-new.csv')
    df_test = df.iloc[:2,:]
    df_inp = (df_test['Headline'] + " " + df_test['Description']).astype('str')
    lab, conf = model_handler.predict(df_inp.tolist())
    assert lab==torch.tensor([0, 0])
    assert conf==torch.tensor([0.6053, 0.5604])
