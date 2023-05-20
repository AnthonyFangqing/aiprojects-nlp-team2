from matplotlib import text
from sklearn import datasets
import torch
from train_functions.starting_train import compute_accuracy
import main


def predict_sentiment(the_model, question): 
    the_model.eval() # set on eval mode. won't keep track of gradients => faster! 
    with torch.no_grad():
        test_vector = torch.LongTensor(datasets.vectorizer.transform([question]).toarray()) #format test data

        output = the_model(test_vector)
        prediction = torch.sigmoid(output).item()

        if prediction > 0.5:
            print(f'{prediction:0.3}: Sincere sentiment')
        else:
            print(f'{prediction:0.3}: Insincere sentiment')
