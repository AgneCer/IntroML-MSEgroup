import torch
from tqdm import tqdm
from test import test_model
from utils import save_model


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, logger, filename, current_model):

    best_accuracy = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for _, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Step the scheduler
        scheduler.step()
        print(f"PRINT LABELS: {labels}")
        train_accuracy = 100. * correct / total
        train_loss = running_loss / len(train_loader)

        val_loss, val_accuracy = test_model(model, val_loader, criterion, device)

        # Save model if validation accuracy is the best so far
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model(model, filename, current_model)

        # Log the metrics to wandb if logger is enabled
        if logger:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
    wandb.finish()