import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def train_cnn_lstm(model, train_loader, val_loader, num_epochs, criterion, optimizer, device, scheduler=None, patience=10, min_delta=0.001):
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

            # Reshape input for CNN-LSTM
            inputs = inputs.permute(0, 2, 1)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += (preds == labels).sum().item()

        # Validation Loop
        val_acc, val_loss, val_metrics = evaluate_cnn_lstm(model, val_loader, criterion, device)

        # Learning Rate Adjustment
        if scheduler:
            scheduler.step(val_loss)

        # Print Metrics
        train_acc = correct_preds / len(train_loader.dataset)
        print(f"Train Loss: {train_loss / len(train_loader.dataset):.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print("Validation Metrics by Class:")
        for class_idx, metrics in val_metrics.items():
            print(f"Class {class_idx}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1']:.4f}")

        # Save Best Model
        if val_loss < best_val_loss - min_delta:
            print("Validation loss improved. Saving best model...")
            best_val_loss = val_loss
            best_accuracy = val_acc
            best_model = model
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    
    print("--------")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    return best_model


def evaluate_cnn_lstm(model, data_loader, criterion, device):
    model.eval()

    total_loss, correct_preds = 0.0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Reshape input for CNN-LSTM
            inputs = inputs.permute(0, 2, 1)

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

    # Calculate per-class metrics --> @NOTE I only have this section implemented for the CNNLSTM model because it's most promising
    metrics = {}
    for class_idx in set(y_true):
        precision = precision_score(y_true, y_pred, labels=[class_idx], average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, labels=[class_idx], average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=[class_idx], average='macro', zero_division=0)
        metrics[class_idx] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    return accuracy, avg_loss, metrics
