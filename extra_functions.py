from matplotlib import text
from sklearn import datasets
import torch
from train_functions.starting_train import compute_accuracy

def predict_sentiment(model, question, dataset): 
    """
    Given a model, question text, and dataset, print the model's response to the question
    Uses the dataset's vectorizer to process the question
    """
    model.eval() # set on eval mode. won't keep track of gradients => faster! 
    with torch.no_grad():
        test_vector = torch.LongTensor(dataset.vectorizer.transform([question]).toarray()) #format test data

        output = model(test_vector)
        prediction = torch.sigmoid(output).item()

        if prediction > 0.5:
            print(f'{prediction:0.3}: Sincere sentiment')
        else:
            print(f'{prediction:0.3}: Insincere sentiment')
