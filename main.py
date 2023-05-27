import os
import torch
import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from networks.StartingNetwork2 import StartingNetwork2
from train_functions.starting_train import starting_train
from torch.utils.data import random_split

from importlib import reload

constants = reload(constants)

def load_starting_dataset():
    # Initalize dataset and model. Then train the model!
    data_path = "train.csv" #TODO: make sure you have train.csv downloaded in your project! this assumes it is in the project's root directory (ie the same directory as main) but you can change this as you please
    # 1306122 rows -- they gave us 12122002 by default
    # qid, question_text, target
    # test_csv has no targets

    whole_dataset = StartingDataset(data_path)
    return whole_dataset

def main():
    """
    Reads in data and 
    """
    
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE, "train_val_split": constants.TRAIN_VAL_SPLIT}

    # TODO: Add GPU support. This line of code might be helpful.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print("Device: ", device)

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    whole_dataset = load_starting_dataset()

    # define the sizes of your training and validation sets
    train_size = int(constants.TRAIN_VAL_SPLIT * len(whole_dataset))
    val_size = len(whole_dataset) - train_size

    # use random_split to split the dataset into non-overlapping
    # training and validation sets
    train_dataset, val_dataset = random_split(whole_dataset, [train_size, val_size])
    # why do you need two identical Dataset objects?
    # should be changed in some way so that val_dataset only does validation

    #model = load_model()
    #predict_sentiment(model, whole_dataset, "this is a true statement")

    model = StartingNetwork2(len(whole_dataset.token2idx), 128, 64)
    model = model.to(device)
    starting_train(train_dataset=train_dataset, val_dataset=val_dataset, model=model, hyperparameters=hyperparameters, n_eval=constants.N_EVAL, device=device) # call the training function from starting_train.py
    # hyperparameters from constants.py
    # can customize model
    # return whole_dataset
    file_name = get_timestamp()
    torch.save(model.state_dict(), 'saved_models/' + file_name + '.pth' )
    return model

from extra_functions import predict_sentiment

def predict_sentiment_test():
    whole_dataset = load_starting_dataset()
    model = StartingNetwork(len(whole_dataset.token2idx))
    model.load_state_dict(torch.load('saved_models/model.pth'))
    question = "How do you find the nth term in a Fibonacci sequence?"
    predict_sentiment(model, question, whole_dataset)

from datetime import datetime
def get_timestamp():
    # Get current time
    current_time = datetime.now()

    # Format the time as a sortable string
    sortable_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    return sortable_time


if __name__ == "__main__":
    finished_model = main()
    #predict_sentiment_test()