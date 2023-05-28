import torch
import torch.nn as nn


class StartingNetwork2(torch.nn.Module):
    """
    Basic logistic regression example. You may need to double check the dimensions :)
    """

    def __init__(self, vocab_size, hidden1, hidden2):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, hidden1) # What could that number mean!?!?!? Ask an officer to find out :)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.functional.relu

    def forward(self, x):
        '''
        x (tensor): the input to the model
        '''
        x = x.float()
        #print("Shape 0: ", x.shape)
        x = self.fc1(x.squeeze())
        x = self.relu(x)
        #print("Shape 1: ", x.shape)
        x = self.fc2(x)
        x = self.relu(x)
        #print("Shape 2: ", x.shape)
        x = self.fc3(x)
        x = x.squeeze()
        #print("Shape 3: ", x.shape)
        return x


