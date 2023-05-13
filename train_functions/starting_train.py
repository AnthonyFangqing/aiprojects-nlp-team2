import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval, device):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    train_losses = []
    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        losses = []
        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            # Zero the gradients
            model.zero_grad()

            # Forward propagate
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Backpropagation and gradient descent
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Periodically evaluate our model + log to Tensorboard
            # skipped for now
            if step % n_eval == -1:
                # Compute training loss and accuracy
                with torch.no_grad():
                    train_loss, train_acc = evaluate(train_loader, model, loss_fn)

                 # Log training results to console
                print(f"Step {step}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")

                # Compute validation loss and accuracy
                # with torch.no_grad():
                    # val_loss, val_acc = evaluate(val_loader, model, loss_fn)

                # Log validation results to console
                # print(f"Step {step}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard. 
                # Don't forget to turn off gradient calculations!
                # evaluate(val_loader, model, loss_fn)

            step += 1
        epoch_loss = sum(losses) / step # average the losses
        train_losses.append(epoch_loss)
        tqdm.write(f'Epoch #{epoch + 1}\tTrain Loss: {epoch_loss:.3f}')
        print()

# probably also doesn't work
def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total

# doesn't work
def evaluate(loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.
    """
    model.eval()  # Put the model in evaluation mode
    total_loss, total_acc = 0, 0
    n_examples = 0

    with torch.no_grad():  # Turn off gradient calculations
        for batch in loader:
            inputs, labels = batch
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            acc = compute_accuracy(outputs, labels)

            total_loss += loss.item() * len(inputs)
            total_acc += acc * len(inputs)
            n_examples += len(inputs)

    avg_loss = total_loss / n_examples
    avg_acc = total_acc / n_examples

    print(f"Validation loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}")

    model.train()  # Put the model back in training mode