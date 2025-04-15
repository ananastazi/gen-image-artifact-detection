import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import copy
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import os

def train_one_epoch(model, criterion, optimizer, dataloader, device, phase):
    if phase == 'train':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    processed_samples = 0

    pbar = tqdm(dataloader, desc=f"{phase.capitalize()} Epoch Processing", leave=False)
    for inputs, labels in pbar:
        if inputs.numel() == 0:
            continue

        inputs = inputs.to(device)
        labels = labels.to(device)

        if phase == 'train':
            optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        processed_samples += inputs.size(0)
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    if processed_samples == 0:
        print(f"Warning: No samples processed in {phase} phase.")
        return 0.0, 0.0

    epoch_loss = running_loss / processed_samples
    epoch_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    return epoch_loss, epoch_f1


def train_model(model: nn.Module,
                     criterion: nn.Module,
                     optimizer: optim.Optimizer,
                     scheduler,
                     dataloaders: dict[str, DataLoader],
                     device: torch.device,
                     num_epochs: int,
                     patience: int,
                     model_save_path: str):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print(f"Starting training for {num_epochs} epochs on {device}...")

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # --- Training Phase ---
        train_loss, train_f1 = train_one_epoch(model, criterion, optimizer, dataloaders['train'], device, 'train')
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        print(f'Train Loss: {train_loss:.4f} Micro F1: {train_f1:.4f}')

        # --- Validation Phase ---
        val_loss, val_f1 = train_one_epoch(model, criterion, optimizer, dataloaders['val'], device, 'val')
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        print(f'Val   Loss: {val_loss:.4f} Micro F1: {val_f1:.4f}')

        # --- LR Scheduler Step ---
        if scheduler:
             if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                 scheduler.step(val_f1)
             else:
                 scheduler.step()
             # Log current learning rate
             # current_lr = optimizer.param_groups[0]['lr']
             # print(f"  LR Scheduler stepped. Current LR: {current_lr:.6f}")


        # --- Check for Improvement & Early Stopping ---
        if val_f1 > best_val_f1:
            print(f"  Validation F1 improved ({best_val_f1:.4f} --> {val_f1:.4f}). Saving model...")
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            try:
                torch.save(model.state_dict(), model_save_path)
            except Exception as e:
                print(f"  Error saving model: {e}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  Validation F1 did not improve. ({epochs_no_improve}/{patience} epochs without improvement)")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Micro F1: {best_val_f1:.4f}')

    # Load best model weights back into the model
    if best_val_f1 > 0:
        model.load_state_dict(best_model_wts)
        print(f"Loaded best model weights (Val F1: {best_val_f1:.4f})")
    else:
        print("Warning: No improvement detected. Model weights remain as initialized or from last epoch.")

    return model, history


def evaluate_model(model: nn.Module,
                   dataloader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    processed_samples = 0

    print("\n--- Starting Evaluation ---")
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for inputs, labels in pbar:
            if inputs.numel() == 0: continue

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            processed_samples += inputs.size(0)
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    if processed_samples == 0:
        print("Error: No samples found in the evaluation dataloader.")
        return {"loss": float('inf'), "micro_f1": 0.0, "accuracy": 0.0, "cm": np.zeros((2, 2))}

    total_loss = running_loss / processed_samples
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    print(f'Evaluation Results:')
    print(f'  Loss: {total_loss:.4f}')
    print(f'  Micro F1 Score: {micro_f1:.4f}')
    print(f'  Accuracy: {accuracy:.4f}')
    print(f'  Confusion Matrix:\n{cm}')

    return {"loss": total_loss, "micro_f1": micro_f1, "accuracy": accuracy, "cm": cm}


def infer_model(model: nn.Module,
                dataloader: DataLoader,
                device: torch.device):
    model.eval()
    all_preds = []
    all_probs = []

    print("\n--- Starting Inference ---")
    pbar = tqdm(dataloader, desc="Inferring", leave=False)
    with torch.no_grad():
        for batch in pbar:
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                inputs = batch[0]
            else:
                 inputs = batch

            if inputs.numel() == 0: continue

            inputs = inputs.to(device)
            outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    print(f"Inference complete. Generated {len(all_preds)} predictions.")
    return np.array(all_preds), np.array(all_probs)