import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image
import os
import glob
import re
import warnings

class ArtifactDataset(Dataset):
    def __init__(self, image_dir, transform=None, file_extension=".png"):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Directory not found: {image_dir}")

        file_paths = sorted(glob.glob(os.path.join(image_dir, f"*{file_extension}")))

        if not file_paths:
             raise FileNotFoundError(f"No '{file_extension}' files found in {image_dir}")

        skipped_files = 0
        for f_path in file_paths:
            fname = os.path.basename(f_path)
            match = re.search(r'_(\d)\.(png)$', fname, re.IGNORECASE)
            if match:
                self.labels.append(int(match.group(1)))
                self.image_files.append(f_path)
            else:
                warnings.warn(f"Could not parse label from filename: {fname}. Skipping file.", UserWarning)
                skipped_files += 1

        if skipped_files > 0:
            print(f"Skipped {skipped_files} files due to parsing errors.")

        if not self.image_files:
             raise ValueError(f"No valid image files found in {image_dir} after parsing.")

        print(f"Found {len(self.image_files)} images in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}. Skipping.")
            return None
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Skipping.")
            return None

class DatasetWrapper(Dataset):
    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform
        if not isinstance(self.subset.dataset, ArtifactDataset):
             raise TypeError("DatasetWrapper expects subset.dataset to be an ArtifactDataset")

    def __getitem__(self, index):
        original_dataset: ArtifactDataset = self.subset.dataset
        original_index = self.subset.indices[index]

        img_path = original_dataset.image_files[original_index]
        label = original_dataset.labels[original_index]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except FileNotFoundError:
             print(f"Error: Image file not found at {img_path} (via Subset index {index}). Skipping.")
             return None
        except Exception as e:
            print(f"Error in DatasetWrapper loading image for index {index} (path: {img_path}): {e}. Skipping.")
            return None

    def __len__(self):
        return len(self.subset)

def get_transforms(img_size: int) -> dict:
    # ImageNet stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    return data_transforms
