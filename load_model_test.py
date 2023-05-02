import os
import torch
import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train
from torch.utils.data import random_split

def load_model():
    PATH = 'first_train_probably_bad'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(PATH, map_location=device)
    return model

def predict_sentiment(model, dataset, text):
    model.eval() # set on eval mode. won't keep track of gradients => faster! 
    with torch.no_grad():
        test_vector = torch.LongTensor(dataset.vectorizer.transform([text]).toarray()) #format test data

        output = model(test_vector)
        prediction = output.item()

        if prediction > 0.5:
            print(f'{prediction:0.3}: Bigger than 0.5')
        else:
            print(f'{prediction:0.3}: Less than 0.5')