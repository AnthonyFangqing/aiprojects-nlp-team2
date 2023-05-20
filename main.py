import os
import torch
import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train
from torch.utils.data import random_split

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

    # Initalize dataset and model. Then train the model!
    data_path = "train.csv" #TODO: make sure you have train.csv downloaded in your project! this assumes it is in the project's root directory (ie the same directory as main) but you can change this as you please
    # 1306122 rows -- they gave us 12122002 by default
    # qid, question_text, target
    # test_csv has no targets

    whole_dataset = StartingDataset(data_path)

    # define the sizes of your training and validation sets
    # train_size = int(constants.TRAIN_VAL_SPLIT * len(whole_dataset))
    # val_size = len(whole_dataset) - train_size

    # use random_split to split the dataset into non-overlapping
    # training and validation sets
    # train_dataset, val_dataset = random_split(whole_dataset, [train_size, val_size])
    # why do you need two identical Dataset objects?
    # should be changed in some way so that val_dataset only does validation

    #model = load_model()
    #predict_sentiment(model, whole_dataset, "this is a true statement")

    # will implement train/eval split later, when we actually use the val_dataset
    model = StartingNetwork(len(whole_dataset.token2idx))
    model = model.to(device)
    starting_train(train_dataset=whole_dataset, val_dataset=whole_dataset, model=model, hyperparameters=hyperparameters, n_eval=constants.N_EVAL, device=device) # call the training function from starting_train.py
    # hyperparameters from constants.py
    # can customize model
    #return whole_dataset
    torch.save(model.state_dict(), 'save/model.pth' )
    return model


if __name__ == "__main__":
    finished_model = main()