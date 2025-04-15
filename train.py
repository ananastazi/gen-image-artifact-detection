# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import argparse
from pathlib import Path
import pprint

from src.config import DEFAULT_CONFIG
from src.utils import set_seed, plot_history, collate_fn_skip_none
from src.data_loader import ArtifactDataset, DatasetWrapper, get_transforms
from src.model import build_model
from src.engine import train_model, evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train an artifact classification model.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_CONFIG["data_dir"], help="Root directory for the dataset.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"], help="Directory to save outputs (model, plots).")
    parser.add_argument("--model_name", type=str, default=DEFAULT_CONFIG["model_name"], help="Model architecture name.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"], help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"], help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"], help="Learning rate.")
    parser.add_argument("--img_size", type=int, default=DEFAULT_CONFIG["img_size"], help="Image size for transforms.")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"], help="Random seed for reproducibility.")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"], help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"], help="Device to use ('cuda:0' or 'cpu').")
    parser.add_argument("--save_name", type=str, default=None, help="Specific filename for the saved model (e.g., 'resnet18_seed42.pth').")
    parser.add_argument("--freeze_backbone", action='store_true', help="If set, freeze all layers except the final classifier (feature extraction).")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"], help="Weight decay for optimizer.")
    parser.add_argument("--optimizer", type=str, default=DEFAULT_CONFIG["optimizer"], help="Optimizer ('AdamW' or 'SGD').")
    parser.add_argument("--scheduler", type=str, default=DEFAULT_CONFIG["scheduler"], help="LR Scheduler ('StepLR', 'ReduceLROnPlateau', None).")
    args = parser.parse_args()
    return args

def main(config):
    # --- Setup ---
    pprint.pprint(config)
    set_seed(config["seed"])
    device = torch.device(config["device"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Data Loading ---
    print("\n--- Loading Data ---")
    data_transforms = get_transforms(config['img_size'])
    train_dir_path = os.path.join(config["data_dir"], config["train_dir_name"])
    test_dir_path = os.path.join(config["data_dir"], config["test_dir_name"])

    try:
        full_train_dataset = ArtifactDataset(
            image_dir=train_dir_path,
            transform=data_transforms['train']
        )

        # Split train into train/validation
        train_size = int((1 - config["val_split"]) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        print(f"Splitting training data: {train_size} train, {val_size} validation.")
        train_subset, val_subset = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(config["seed"])
        )

        val_dataset_transformed = DatasetWrapper(val_subset, data_transforms['val'])

        train_dataset = train_subset

        test_dataset = ArtifactDataset(
            image_dir=test_dir_path,
            transform=data_transforms['test']
        )
    except (FileNotFoundError, ValueError) as e:
         print(f"Error loading dataset: {e}")
         return

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset_transformed)}
    print(f"Dataset sizes: Train={dataset_sizes['train']}, Val={dataset_sizes['val']}, Test={len(test_dataset)}")

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True, collate_fn=collate_fn_skip_none),
        'val': DataLoader(val_dataset_transformed, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True, collate_fn=collate_fn_skip_none),
        'test': DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True, collate_fn=collate_fn_skip_none)
    }

    # --- Model, Loss, Optimizer, Scheduler ---
    print("\n--- Setting up Model ---")
    model = build_model(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        use_pretrained=config['use_pretrained']
    ).to(device)

    # --- Loss, Optimizer, Scheduler Setup ---
    criterion = nn.CrossEntropyLoss()
    params_to_optimize = model.parameters()

    # --- Handle Feature Extraction ---
    if config.get("freeze_backbone", False):
        print("--- Feature Extraction Mode: Freezing backbone layers ---")
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only the final classifier parameters
        classifier_params = []
        if config['model_name'].startswith("resnet") and hasattr(model, 'fc'):
            print("  Unfreezing ResNet 'fc' layer.")
            for param in model.fc.parameters():
                param.requires_grad = True
            classifier_params = model.fc.parameters()
        elif config['model_name'].startswith("efficientnet") and hasattr(model, 'classifier') and isinstance(
                model.classifier[-1], nn.Linear):
            print("  Unfreezing EfficientNet final classifier layer.")
            for param in model.classifier[-1].parameters():
                param.requires_grad = True
            classifier_params = model.classifier[-1].parameters()
        elif config['model_name'].startswith("vit"):
            if hasattr(model, 'heads') and hasattr(model.heads, 'head'):
                print("  Unfreezing ViT 'heads.head' layer.")
                for param in model.heads.head.parameters():
                    param.requires_grad = True
                classifier_params = model.heads.head.parameters()
            elif hasattr(model, 'head'):
                print("  Unfreezing ViT 'head' layer.")
                for param in model.head.parameters():
                    param.requires_grad = True
                classifier_params = model.head.parameters()
            else:
                print("  Warning: Could not find specific ViT head to unfreeze. Training ALL parameters instead.")
                for param in model.parameters():  # Fallback: unfreeze everything if head isn't found
                    param.requires_grad = True
                classifier_params = model.parameters()  # Revert to all params
        else:
            print(
                f"  Warning: Unsupported model '{config['model_name']}' for specific unfreezing. Training ALL parameters instead.")
            for param in model.parameters():  # Fallback: unfreeze everything
                param.requires_grad = True
            classifier_params = model.parameters()  # Revert to all params

        params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
        # Set up a higher LR when only training the head
        config['learning_rate'] = config.get('head_lr', config['learning_rate'] * 5)

    # Setup optimizer
    if config['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(params_to_optimize, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(params_to_optimize, lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    print(f"Using optimizer: {config['optimizer']}")

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Number of trainable parameters: {num_trainable:,}")

    # Setup scheduler
    scheduler = None
    if config['scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_scheduler_step'], gamma=config['lr_scheduler_gamma'])
        print(f"Using StepLR scheduler.")
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=config['lr_scheduler_gamma'], patience=config['early_stopping_epochs'] // 2, verbose=True)
        print(f"Using ReduceLROnPlateau scheduler based on validation F1.")
    elif config['scheduler'] is not None:
         print(f"Warning: Unsupported scheduler '{config['scheduler']}'. No scheduler will be used.")

    if config.get("save_name"):
        model_save_filename = config["save_name"]
        if not model_save_filename.endswith(".pth"):
            model_save_filename += ".pth"
    else:
        model_save_filename = config["save_model_name"]

    model_save_path = str(output_dir / model_save_filename)
    print(f"Model will be saved to: {model_save_path}")

    # --- Training ---
    print("\n--- Starting Training ---")
    trained_model, history = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
        device=device,
        num_epochs=config['num_epochs'],
        patience=config['early_stopping_epochs'],
        model_save_path=model_save_path
    )

    # --- Plotting History ---
    if history:
        plot_save_path = str(output_dir / config["plot_name"])
        plot_history(history, plot_save_path)

    # --- Evaluation on Test Set ---
    print("\n--- Evaluating on Test Set using Best Model ---")
    try:
        eval_model = build_model(config['model_name'], config['num_classes'], use_pretrained=False).to(device)
        eval_model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Loaded best model weights from {model_save_path}")

        test_metrics = evaluate_model(
            model=eval_model,
            dataloader=dataloaders['test'],
            criterion=criterion,
            device=device
        )
        print("\nFinal Test Metrics:")
        pprint.pprint(test_metrics)

    except FileNotFoundError:
        print(f"Error: Could not find saved model at {model_save_path}. Skipping test evaluation.")
    except Exception as e:
        print(f"Error during test evaluation: {e}")

if __name__ == "__main__":
    args = parse_args()

    run_config = DEFAULT_CONFIG.copy()

    cli_args = vars(args)
    if cli_args.get("freeze_backbone"):
        run_config["freeze_backbone"] = True

    run_config.update({k: v for k, v in vars(args).items() if v is not None})
    run_config["device"] = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.save_name:
        run_config["save_name"] = args.save_name

    main(run_config)
