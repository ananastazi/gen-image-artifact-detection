# Artifact Detection

## 1. Task Description

**Introduction:** Modern AI image generation models sometimes produce images with unintended artifacts, especially with complex subjects like human faces (e.g., distorted hands, fingers, eyes, text fragments, mask remnants). This project aims to build an automated Quality Assurance (QA) system to detect such artifacts.

**Goal:** Develop a binary image classifier to distinguish between generated images **with artifacts (Class 0)** and **artifact-less images (Class 1)**. This is framed as a classification challenge where the primary goal is to maximize the **Micro F1 score** on a hidden test set.

**Input Data:**
*   A `train` directory containing 1800 labeled images.
*   A `test` directory containing 200 unlabeled images.
*   Filenames follow the pattern: `image_<frame_index>_<class label>.png` (label is 0 or 1 for training images).

## 2. Solution Overview

This solution employs a deep learning approach using **transfer learning** with standard Convolutional Neural Network (CNN) or Vision Transformer (ViT) architectures pre-trained on ImageNet.

**Core Strategy:**
1.  **Load Pre-trained Models:** Utilize models like ResNet, EfficientNet, or ViT from `torchvision`.
2.  **Adapt for Binary Classification:** Replace the model's final classification layer to output 2 classes (artifact/no artifact).
3.  **Fine-tuning / Feature Extraction:** Train the adapted model on the provided artifact dataset. Optionally freeze the backbone for feature extraction.
4.  **Data Augmentation:** Apply augmentations to the training data to improve robustness and generalization.
5.  **Training Pipeline:** Implement a standard training loop with validation, early stopping based on validation Micro F1 score, and learning rate scheduling.
6.  **Evaluation:** Measure performance using the required Micro F1 score, along with accuracy and confusion matrix on the test set.
7.  **Ensembling:** Provide functionality to combine predictions from multiple trained models (potentially different architectures or seeds) using probability averaging (soft voting) for potentially improved results.

## 3. Features

*   **Flexible Model Choice:** Easily switch between ResNet, EfficientNet, ViT variants via configuration.
*   **Transfer Learning:** Leverages pre-trained ImageNet weights for faster convergence and potentially better performance.
*   **Data Augmentation:** Includes common augmentations (Crop, Flip, Jitter, Rotation) for training data.
*   **Standard Training Loop:** Includes validation, CrossEntropyLoss, AdamW/SGD optimizers, StepLR/ReduceLROnPlateau schedulers.
*   **Early Stopping:** Prevents overfitting by stopping training if validation Micro F1 doesn't improve.
*   **Micro F1 Optimization:** Training loop saves the model with the best validation Micro F1 score.
*   **Reproducibility:** Seed setting for `random`, `numpy`, and `torch`.
*   **Inference Scripts:** Separate scripts for predicting with a single model (`predict.py`) or an ensemble (`ensemble_predict.py`).
*   **Training History Plotting:** Generates plots for loss and Micro F1 over epochs.
*   **Command-Line Interface:** Allows configuration overrides for easy experimentation.
*   **Ensemble Prediction:** Supports averaging probabilities from multiple models.

## 4. Project Structure

    task1
    ├── train.py                # Main script to train models
    ├── predict.py              # Script for inference with a single model
    ├── ensemble_predict.py     # Script for inference with multiple models (ensemble)
    ├── src/
    │   ├── __init__.py         # Makes 'src' a Python package
    │   ├── config.py           # Default configuration parameters
    │   ├── data_loader.py      # Dataset class, transforms, dataloaders
    │   ├── engine.py           # Training, evaluation, inference core logic
    │   ├── model.py            # Model building function
    │   └── utils.py            # Utility functions (seed, plotting, collate)
    ├── data/
    │   ├── train/              # Training images (image_..._0.png, image_..._1.png)
    │   │   └── ...
    │   └── test/               # Test images (image_..._?.png - label unknown)
    │       └── ...
    ├── outputs/                # Directory for saving models, plots, predictions
    │   ├── resnet18_run1/      # Example output subdir for a run
    │   │   ├── best_classifier.pth
    │   │   ├── training_history.png
    │   │   └── test_predictions.csv
    │   └── ensemble_predictions.csv # Example ensemble output
    ├── requirements.txt        # Python dependencies
    ├── dev_process.txt         # A document that describes process of solution 
    └── README.md               # This file

## 5. Prerequisites & Environment Setup (GPU Recommended)

*   Python 3.7+ (Author used **Python 3.10**)
*   `pip` for package installation
*   Virtual Environment (Strongly Recommended)
*   NumPy version `<=2.0` (Due to potential compatibility issues with newer PyTorch versions and NumPy 2.0+).
*   **NVIDIA GPU with CUDA (Strongly Recommended for Training):**
    *   **Check GPU Compute Capability:** Before proceeding, identify your NVIDIA GPU's Compute Capability. You can find this on the [NVIDIA CUDA GPUs page](https://developer.nvidia.com/cuda-gpus).
    *   **Install Compatible CUDA Toolkit:** Download and install a version of the CUDA Toolkit that is compatible with both your GPU's Compute Capability and the desired PyTorch version. Refer to the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).
    *   **Install Compatible PyTorch:** This is the most critical step. You **must** install a PyTorch build that explicitly matches your installed CUDA Toolkit version. Visit the [PyTorch Previous Versions page](https://pytorch.org/get-started/previous-versions/) to find the correct installation command. **Do not** just run `pip install torch`.
    *   **Author's Setup Example:** For reference, this project was developed and tested with:
        *   GPU Compute Capability: **8.6**
        *   CUDA Toolkit Version: **11.3** (compatible range 11.1 - 11.4 often works)
        *   PyTorch Version: `torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`
        *   Python Version: **3.10**
        *   NumPy Version: **1.26.4** (or any version `<=2.0`)
    *   **If not using GPU:** The code supports running on CPU (`--device cpu`), but training will be significantly slower. Ensure PyTorch (CPU version) and other dependencies are installed.

## 6. Installation

1. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Linux/macOS:
    source venv/bin/activate
    # Windows (cmd):
    .\venv\Scripts\activate
    # Windows (PowerShell):
    # .\venv\Scripts\Activate.ps1 (might require Set-ExecutionPolicy)
    ```
2. **Install PyTorch (Crucial - Use Correct Command):**
    *   Go to the [PyTorch Previous Versions page](https://pytorch.org/get-started/previous-versions/).
    *   Find the command corresponding to your desired PyTorch version (like 1.12.1), your OS, package manager (`pip`), and your **installed CUDA version** (e.g., `cu113` for CUDA 11.3).
    *   Run that specific command. For example (matching author's setup):
        ```bash
        pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
        ```
    *   **Do not** proceed until PyTorch with the correct CUDA support is installed.

3. **Install other dependencies:**
    Once PyTorch is correctly installed, install the remaining packages from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` lists packages like `numpy`, `matplotlib`, `scikit-learn`, `tqdm`, etc., but **exclude** `torch`, `torchvision`, `torchaudio` if you installed them manually in the previous step, or ensure the versions match exactly)*

## 7. Data Setup
1. Inside data/, create two subdirectories: train and test.
2. Place all your training images (named image_<index>_<label>.png) into the data/train/ directory.
3. Place all your test images into the data/test/ directory. The labels in the filenames for test images don't matter, as they aren't used during inference.

## 8. Configuration
* Default parameters are set in src/config.py.
* Most parameters can be overridden via command-line arguments when running train.py, predict.py, or ensemble_predict.py. 

## 9. Usage Guide
### Training a Model
Use train.py to train a new model. The script will:
* Load data, split train/validation sets.
* Build the specified model architecture.
* Train the model using specified hyperparameters.
* Perform early stopping based on validation Micro F1.
* Save the best model weights (.pth file) to the output directory.
* Save a plot of the training history (.png file).
* Evaluate the best model on the test set and print metrics.

**Example**: Train a ResNet-34 model for 25 epochs with specific LR and output directory.

    python train.py \
    --model_name resnet34 \
    --epochs 25 \
    --batch_size 32 \
    --lr 0.0003 \
    --img_size 256 \
    --optimizer AdamW \
    --scheduler ReduceLROnPlateau \
    --output_dir ./outputs/resnet34_run1 \
    --save_name resnet34_best.pth \
    --device cuda:0 \
    --num_workers 4 \
    --seed 123

### Running Inference (Single Model)
Use predict.py to generate predictions for images in a directory using a trained model.

**Example**: Generate predictions using the trained ResNet-34 model on the data/test directory.
    
    python predict.py \
    --model_path ./outputs/resnet34_run1/resnet34_best.pth \
    --model_name resnet34 \
    --input_dir ./data/test \
    --output_csv ./outputs/resnet34_run1/test_predictions.csv \
    --img_size 256 \
    --device cuda:0

This will create a CSV file with filenames, predicted labels (0 or 1), and probabilities for each class.

### Running Ensemble Inference
Use ensemble_predict.py to combine predictions from multiple trained models.

**Example**: Ensemble predictions from a ResNet-18 and a ResNet-34 model.

    python ensemble_predict.py \
    --input_dir ./data/test \
    --output_csv ./outputs/ensemble_r18_r34.csv \
    --img_size 224 \
    --device cuda:0 \
    --model_configs \
        resnet18:./outputs/resnet18_run1/resnet18_best.pth \
        resnet34:./outputs/resnet34_run1/resnet34_best.pth

(Ensure the model architectures and paths match your trained models)
This averages the predicted probabilities from the specified models before making the final class prediction (soft voting) and saves the results to a CSV.

## 10. Final Metrics
This section presents the final performance metrics achieved on the **test set** for several models trained using this codebase. All models listed below were trained for a maximum of **15 epochs** using **seed 42**, with early stopping based on validation Micro F1, and with using the default full fine-tuning approach for all models apart from resnet34.

**Metrics Reported:**

*   **Micro F1 Score:** The primary target metric (equivalent to accuracy in binary classification).
*   **Accuracy:** Overall percentage of correct predictions.
*   **Loss:** Cross-Entropy loss value on the test set.
*   **Confusion Matrix (CM):** Shows the distribution of predictions:
    ```
    [[True Negative (TN), False Positive (FP)],
     [False Negative (FN), True Positive (TP)]]
    ```
    *   Class 0: Artifact Present
    *   Class 1: Artifact Absent (Artifact-less)

---

### `resnet18_ft_seed42`

*   **Micro F1 Score:** 0.9450
*   **Accuracy:** 0.9450
*   **Test Loss:** 0.1847
*   **Confusion Matrix:**
    ```
    [[ 14   6]   # Correctly identified 14 artifact images, misclassified 6 artifact-less as artifacts.
     [  5 175]]  # Missed 5 artifact images, correctly identified 175 artifact-less images.
    ```

---

### `resnet34_ft_seed42`

*   **Micro F1 Score:** 0.9300
*   **Accuracy:** 0.9300
*   **Test Loss:** 0.2188
*   **Confusion Matrix:**
    ```
    [[  8  12]   # Correctly identified 8 artifact images, misclassified 12 artifact-less as artifacts.
     [  2 178]]  # Missed 2 artifact images, correctly identified 178 artifact-less images.
    ```

---

### `efficientnetb0_ft_seed42`

*   **Micro F1 Score:** 0.9600
*   **Accuracy:** 0.9600
*   **Test Loss:** 0.1805
*   **Confusion Matrix:**
    ```
    [[ 13   7]   # Correctly identified 13 artifact images, misclassified 7 artifact-less as artifacts.
     [  1 179]]  # Missed only 1 artifact image, correctly identified 179 artifact-less images.
    ```

---

### `vit_b_16_ft_seed42`

*   **Micro F1 Score:** 0.9000
*   **Accuracy:** 0.9000
*   **Test Loss:** 0.2961
*   **Confusion Matrix:**
    ```
    [[  0  20]   # Failed to identify *any* artifact images (0 True Negatives), misclassified 20 artifact-less as artifacts.
     [  0 180]]  # Did not miss any artifact images (because it predicted none), correctly identified all 180 artifact-less images.
    ```
    *(Note: The ViT model seems to have heavily biased towards predicting Class 1 (artifact-less) in this specific run, failing completely on Class 0.)*

---

**Summary:**

Based on these specific runs (15 epochs, seed 42):

*   **`efficientnetb0_ft_seed42` achieved the highest Micro F1 score (0.9600)** on the test set, demonstrating the best performance among these examples. It also had the lowest test loss and missed the fewest artifact images (FN=1).
*   `resnet18_ft_seed42` performed well (0.9450 F1).
*   `resnet34_ft_seed42` performed slightly lower (0.9300 F1).
*   `vit_b_16_ft_seed42` performed poorly (0.9000 F1) in this specific run, failing to detect any images with artifacts (Class 0).