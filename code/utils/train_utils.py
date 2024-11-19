import torch

"""
Train the model on the training set and validate on the validation set
Returns the training and validation losses for each epoch
"""
def train_model(model, train_loader, epochs, optimizer, criterion, device, val_loader=None):
    print(f"Training on {len(train_loader.dataset)} samples")
    model.train()
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss)
        if val_loader:
            val_loss = evaluate_model(model, val_loader, criterion, device)
            val_losses.append(val_loss)

    return train_losses, val_losses


"""
Evaluate the model on the test set
Returns the average loss over the whole test set
"""
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            losses.append(loss.item())

    return sum(losses) / len(losses)