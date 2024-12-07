import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device, scheduler=None, patience=5, min_delta=0.001):
    best_model = model
    best_val_loss = float("inf")
    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training Loop
        model.train()
        train_loss, correct_preds = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if hasattr(model, "requires_reshape") and model.requires_reshape:
                inputs = inputs.unsqueeze(1)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += (preds == labels).sum().item()

        # Validation Loop
        val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)

        # Adjust learning rate if needed
        if scheduler:
            scheduler.step(val_loss)

        # Print Metrics
        train_acc = correct_preds / len(train_loader.dataset)
        print(f"Train Loss: {train_loss / len(train_loader.dataset):.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Check for Improvement
        if val_loss < best_val_loss - min_delta:
            print("Validation loss improved. Saving best model...")
            best_val_loss = val_loss
            best_accuracy = val_acc
            best_model = model
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience: {patience_counter}/{patience}")

        # Early Stopping Trigger
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    return best_model



# Evaluation Loop (Generalized)
def evaluate_model(model, data_loader, criterion, device):
    model.eval()

    total_loss, correct_preds = 0.0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if hasattr(model, "requires_reshape") and model.requires_reshape:
                inputs = inputs.unsqueeze(1)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            # Predictions
            _, preds = torch.max(outputs, 1)
            correct_preds += (preds == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct_preds / len(data_loader.dataset)

    return accuracy, avg_loss
