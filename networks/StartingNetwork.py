import torch
import torch.nn as nn


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression example. You may need to double check the dimensions :)
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, 50) # What could that number mean!?!?!? Ask an officer to find out :)
        self.fc2 = nn.Linear(50, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        x (tensor): the input to the model
        '''
        x = self.fc1(x.squeeze(1).float())
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


