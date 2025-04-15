import torch
from torch.utils.data import DataLoader
import os
import argparse
import pandas as pd
from pathlib import Path
import pprint

from src.config import DEFAULT_CONFIG
from src.utils import collate_fn_skip_none
from src.data_loader import ArtifactDataset, get_transforms
from src.model import build_model
from src.engine import infer_model

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using a trained artifact classifier.")
    PROJECT_ROOT = Path(__file__).resolve().parent
    default_model_path = PROJECT_ROOT / DEFAULT_CONFIG["output_dir"] / DEFAULT_CONFIG["save_model_name"]
    default_input_dir = PROJECT_ROOT / DEFAULT_CONFIG["data_dir"] / DEFAULT_CONFIG["test_dir_name"]
    parser.add_argument("--model_path", type=str, default=str(default_model_path), help="Path to the trained model .pth file.")
    parser.add_argument("--input_dir", type=str, default=str(default_input_dir), help="Directory containing images to classify.")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Path to save the output CSV file (relative to project root).")
    parser.add_argument("--img_size", type=int, default=DEFAULT_CONFIG["img_size"], help="Image size used during training.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"], help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"], help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"], help="Device ('cuda:0' or 'cpu').")
    parser.add_argument("--model_name", type=str, default=DEFAULT_CONFIG["model_name"], help="Name of the model architecture used for training.")
    parser.add_argument("--num_classes", type=int, default=DEFAULT_CONFIG["num_classes"], help="Number of classes model was trained for.")
    args = parser.parse_args()
    return args

def main(args):
    print("--- Inference Configuration ---")
    pprint.pprint(vars(args))
    device = torch.device(args.device)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    # --- Load Model ---
    print(f"\n--- Loading model {args.model_name} from {args.model_path} ---")
    try:
        model = build_model(
            model_name=args.model_name,
            num_classes=args.num_classes,
            use_pretrained=False
        ).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded successfully from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        return
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return

    # --- Inference ---
    preds, probs = infer_model(
        model=model,
        dataloader=inference_dataloader,
        device=device
    )

    # --- Save Results ---
    if len(preds) == 0:
         print("Error: No predictions were generated (possibly all images failed to load).")
         return

    # --- Match Predictions to Filenames ---
    if len(preds) != len(filenames):
         print(f"Warning: Number of predictions ({len(preds)}) does not match number of "
               f"files found ({len(filenames)}). This might happen if some images failed "
               f"to load during inference. Resulting CSV might be incomplete or misaligned.")
         filenames = filenames[:len(preds)]


    print(f"\n--- Saving {len(preds)} predictions to {args.output_csv} ---")
    submission_df = pd.DataFrame({
        'filename': filenames,
        'prediction': preds,
        'probability_0': probs[:, 0],
        'probability_1': probs[:, 1]
    })
    try:
        submission_df.to_csv(args.output_csv, index=False)
        print("Predictions saved successfully.")
        print(submission_df.head())
    except Exception as e:
        print(f"Error saving predictions to CSV: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)