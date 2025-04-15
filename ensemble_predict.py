import torch
from torch.utils.data import DataLoader
import os
import argparse
import pandas as pd
from pathlib import Path
import pprint
import numpy as np
from typing import List, Tuple

from src.config import DEFAULT_CONFIG
from src.utils import collate_fn_skip_none
from src.data_loader import ArtifactDataset, get_transforms
from src.model import build_model
from src.engine import infer_model

def parse_args():
    parser = argparse.ArgumentParser(description="Run ENSEMBLE inference using trained artifact classifiers.")

    # --- Arguments for data and general settings ---
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images to classify.")
    parser.add_argument("--output_csv", type=str, default="ensemble_predictions.csv", help="Path to save the final ensemble predictions CSV.")
    parser.add_argument("--img_size", type=int, default=DEFAULT_CONFIG["img_size"], help="Image size used during training.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"], help="Batch size for inference (per model).")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"], help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"], help="Device ('cuda:0' or 'cpu').")
    parser.add_argument("--num_classes", type=int, default=DEFAULT_CONFIG["num_classes"], help="Number of classes models were trained for.")

    # --- Argument for specifying ensemble models ---
    parser.add_argument(
        "--model_configs",
        nargs='+',
        required=True,
        help="List of model configurations for the ensemble. "
             "Format for each: 'model_architecture:path/to/weights.pth'. "
             "Example: resnet18:outputs/resnet18_seed42.pth resnet34:outputs/resnet34_seed42.pth"
    )

    args = parser.parse_args()
    return args

def parse_model_configs(config_strings: List[str]) -> List[Tuple[str, str]]:
    """ Parses the --model_configs argument strings into (architecture, path) tuples. """
    parsed_configs = []
    for config_str in config_strings:
        parts = config_str.split(':', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid model config format: '{config_str}'. "
                             "Expected 'architecture:path/to/weights.pth'")
        architecture, weight_path = parts
        if not Path(weight_path).is_file():
             warnings.warn(f"Weight file not found at specified path: {weight_path}", UserWarning)
        parsed_configs.append((architecture, weight_path))
    if not parsed_configs:
         raise ValueError("No valid model configurations provided.")
    return parsed_configs


def main(args):
    print("--- Ensemble Inference Configuration ---")
    pprint.pprint(vars(args))
    device = torch.device(args.device)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Parse Model Configurations ---
    try:
        ensemble_models_info = parse_model_configs(args.model_configs)
        print(f"\nFound {len(ensemble_models_info)} models for ensembling:")
        for arch, path in ensemble_models_info:
             print(f"  - Arch: {arch}, Path: {path}")
    except ValueError as e:
        print(f"Error parsing model configurations: {e}")
        return

    # --- Load Data ---
    print(f"\n--- Loading data from {args.input_dir} ---")
    try:
        inference_transforms = get_transforms(args.img_size)['test']
        inference_dataset = ArtifactDataset(
            image_dir=args.input_dir,
            transform=inference_transforms
        )
        filenames = [os.path.basename(f) for f in inference_dataset.image_files]
    except (FileNotFoundError, ValueError) as e:
         print(f"Error loading inference data: {e}")
         return

    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_skip_none
    )

    # --- Run Inference for Each Model ---
    all_model_probs = [] # List to store probability arrays (N_models x N_samples x N_classes)

    for i, (model_arch, model_weights_path) in enumerate(ensemble_models_info):
        print(f"\n--- Running inference for Model {i+1}/{len(ensemble_models_info)} ({model_arch}) ---")
        try:
            # Build model structure
            model = build_model(
                model_name=model_arch,
                num_classes=args.num_classes,
                use_pretrained=False
            ).to(device)
            # Load trained weights
            model.load_state_dict(torch.load(model_weights_path, map_location=device))
            print(f"  Loaded weights from {model_weights_path}")

            # Perform inference
            # infer_model returns (preds_array, probs_array)
            _, model_probs = infer_model(
                model=model,
                dataloader=inference_dataloader,
                device=device
            )

            # Check if predictions were generated for this model
            if model_probs.shape[0] == 0:
                print(f"  Warning: Model {i+1} generated 0 predictions. Skipping.")
                continue

            all_model_probs.append(model_probs)
            print(f"  Generated {model_probs.shape[0]} probability sets.")

        except FileNotFoundError:
            print(f"  Error: Model file not found at {model_weights_path}. Skipping this model.")
            continue
        except Exception as e:
            print(f"  Error loading or running inference for model {model_weights_path}: {e}. Skipping this model.")
            continue

    # --- Ensemble Predictions ---
    if not all_model_probs:
        print("\nError: No models successfully generated predictions. Cannot create ensemble.")
        return

    # Ensure all probability arrays have the same number of samples
    num_samples = all_model_probs[0].shape[0]
    if not all(p.shape[0] == num_samples for p in all_model_probs):
        print("\nError: Models generated different numbers of predictions. Cannot average.")
        return

    print(f"\n--- Combining probabilities from {len(all_model_probs)} models ---")
    # Stack probabilities (N_models, N_samples, N_classes)
    stacked_probs = np.stack(all_model_probs, axis=0)
    avg_probs = np.mean(stacked_probs, axis=0)
    ensemble_preds = np.argmax(avg_probs, axis=1)

    print(f"Generated {len(ensemble_preds)} ensemble predictions.")

    # --- Save Results ---
    if len(ensemble_preds) != len(filenames):
         print(f"Warning: Number of ensemble predictions ({len(ensemble_preds)}) does not match number of "
               f"files found ({len(filenames)}). This might happen if some images failed "
               f"to load during inference. Resulting CSV might be incomplete or misaligned.")
         filenames = filenames[:len(ensemble_preds)]

    print(f"\n--- Saving ensemble predictions to {args.output_csv} ---")
    submission_df = pd.DataFrame({
        'filename': filenames,
        'ensemble_prediction': ensemble_preds,
        'avg_probability_0': avg_probs[:, 0],
        'avg_probability_1': avg_probs[:, 1]
    })
    try:
        submission_df.to_csv(args.output_csv, index=False)
        print("Ensemble predictions saved successfully.")
        print(submission_df.head())
    except Exception as e:
        print(f"Error saving ensemble predictions to CSV: {e}")


if __name__ == "__main__":
    import warnings
    args = parse_args()
    main(args)