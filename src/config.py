import torch
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CONFIG = {
    # --- Paths ---
    "data_dir": os.path.join(PROJECT_ROOT, "data"),
    "train_dir_name": "train",
    "test_dir_name": "test",
    "output_dir": os.path.join(PROJECT_ROOT, "outputs"),
    "save_model_name": "best_classifier.pth",
    "submission_name": "result.csv",
    "plot_name": "training_history.png",

    # --- Model ---
    "model_name": "resnet18",
    "num_classes": 2,
    "use_pretrained": True,

    # --- Training ---
    "batch_size": 32,
    "num_epochs": 15,
    "learning_rate": 0.001,
    "optimizer": "AdamW", # 'AdamW' or 'SGD'
    "momentum": 0.9, # For SGD
    "weight_decay": 0.0001, # 'StepLR', 'ReduceLROnPlateau', or None
    "scheduler": "StepLR", # For StepLR
    "lr_scheduler_step": 7,
    "lr_scheduler_gamma": 0.1,
    "early_stopping_epochs": 5,

    # --- Data ---
    "img_size": 224,
    "val_split": 0.15,

    # --- System ---
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "num_workers": 0,
}