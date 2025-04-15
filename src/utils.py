import torch
import numpy as np
import random
import matplotlib.pyplot as plt

def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed_value}")

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch:
        print("Warning: Collate function received an empty batch after filtering.")
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

def plot_history(history: dict, save_path: str):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    if 'train_loss' in history:
        plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot Micro F1
    plt.subplot(1, 2, 2)
    if 'train_f1' in history:
        plt.plot(history['train_f1'], label='Train Micro F1')
    if 'val_f1' in history:
        plt.plot(history['val_f1'], label='Val Micro F1')
    plt.legend()
    plt.title('Micro F1 History')
    plt.xlabel('Epoch')
    plt.ylabel('Micro F1 Score')

    plt.tight_layout()
    try:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving plot to {save_path}: {e}")
    plt.show()
