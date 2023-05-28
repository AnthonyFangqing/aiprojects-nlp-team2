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
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
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

            # Periodically evaluate our model
            # TODO: log to Tensorboard
            if step % n_eval == 0:
                train_loss = loss
                train_acc = compute_accuracy(outputs, labels)
                print(f"Step {step}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
                evaluate(val_loader, model, loss_fn, device)

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
    #print("Output size:", outputs.size())
    #print("Label size:", labels.size())
    n_correct = (torch.round(outputs.squeeze()) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total

# doesn't work
def evaluate(loader, model, loss_fn, device):
    """
    Computes the loss and accuracy of a model on the validation dataset.
    Essentially does 1 epoch of calculations using a batch_size amount of data from the validation data loader
    """
    model.eval()  # Put the model in evaluation mode
    total_losses, total_accs = [], []
    total = 0
    with torch.no_grad():  # Turn off gradient calculations
        for batch in loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            acc = compute_accuracy(outputs, labels)

            total_losses.append(loss)
            total_accs.append(acc)

            total+=1

    avg_loss = sum(total_losses) / total
    avg_acc = sum(total_accs) / total

    print(f"Validation loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}")

    model.train()  # Put the model back in training mode

    return (avg_loss, avg_acc)