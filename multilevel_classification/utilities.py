import torch
from tqdm import tqdm

def eval_model(device, model, data_loader):
    model.eval() # Set model to eval mode
    true_preds, num_preds = 0., 0.

    with torch.no_grad(): # Deactivate gradients for the following code
        for data_inputs, data_labels in data_loader:

            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds) # Sigmoid to map predictions between 0 and 1
            pred_labels = (preds >= 0.5).long() # Binarize predictions to 0 and 1

            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (pred_labels == data_labels).sum()
            num_preds += data_labels.shape[0]

    acc = true_preds / num_preds
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")

def compute_accuracy(device, model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def train_with_metrics(device, model, optimizer, train_loader, val_loader, loss_module,
                       num_epochs=50, early_stop_threshold=None, early_stop_patience=2):
    """Train model and print per-epoch train/validation metrics.

    Parameters:
    - num_epochs: maximum number of epochs to run
    - early_stop_threshold: if set (float), stop when train loss is below this value
      for `early_stop_patience` consecutive epochs.
    - early_stop_patience: number of consecutive epochs the train loss must be below
      the threshold to trigger early stopping.
    """
    consecutive_below = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_module(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_acc = compute_accuracy(device, model, train_loader)
        val_acc = compute_accuracy(device, model, val_loader) if val_loader else None
        print(f"Epoch {epoch+1:3d}: train_loss={avg_train_loss:.4f} train_acc={100*train_acc:.2f}% "
              + (f"val_acc={100*val_acc:.2f}%" if val_acc is not None else ""))

        # Early stopping based on consecutive train loss condition
        if early_stop_threshold is not None:
            if avg_train_loss < early_stop_threshold:
                consecutive_below += 1
            else:
                consecutive_below = 0

            if consecutive_below >= early_stop_patience:
                print(f"Early stopping: train loss < {early_stop_threshold} for {early_stop_patience} consecutive epochs.")
                break


def train_model(device, model, optimizer, data_loader, loss_module, num_epochs=100):
    # Set model to train mode
    model.train()

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for data_inputs, data_labels in data_loader:

            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            ## Step 2: Run the model on the input data
            preds = model(data_inputs)

            ## Step 3: Calculate the loss
            loss = loss_module(preds, data_labels)

            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()

            ## Step 5: Update the parameters
            optimizer.step()

def calculate_mesh_grid(x_lower, x_upper, step, device):
    x1 = torch.arange(x_lower, x_upper, step=step, device=device)
    x2 = torch.arange(x_lower, x_upper, step=step, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2, indexing='ij')
    return xx1, xx2