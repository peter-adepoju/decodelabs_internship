"""
src/data_utils.py
=================
Image loading, PyTorch Dataset class, and federated site partitioning.

Contents
--------
  TBDataset          : PyTorch Dataset for TB chest X-ray images
  build_transforms   : Create train / val / test transforms
  dirichlet_split    : Partition a dataset across simulated FL sites
  uniform_split      : Partition a dataset uniformly across clients
  get_class_weights  : Compute per-class weights for imbalanced datasets
  make_mock_dataset  : Create a tiny synthetic dataset for testing

Import note
-----------
torch and torchvision are imported lazily inside TBDataset,
build_transforms, and get_class_weights. Pure-NumPy utilities
(dirichlet_split, uniform_split, make_mock_dataset) can therefore be
imported even when PyTorch is unavailable or its DLLs are broken.
"""

import os
import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image

# torch / torchvision: imported lazily inside each function/class that needs them.


# ──────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────

_TBDataset_cls = None  # module-level cache

def TBDataset(
    dataframe: pd.DataFrame,
    transform: Callable | None = None,
    image_col: str = "image_path",
    label_col: str = "label",
):
    """
    Factory that returns a TBDataset instance.

    torch.utils.data.Dataset is imported lazily on first call so that
    importing data_utils never touches PyTorch (avoids DLL errors when
    only NumPy utilities are needed).

    Parameters
    ----------
    dataframe : pd.DataFrame
        Table of image paths and labels. Must contain 'image_path' and 'label'.
    transform : callable or None
        Torchvision transforms to apply to each image.
    image_col : str
        Column name for image paths.
    label_col : str
        Column name for labels.

    Returns
    -------
    Instance of a torch.utils.data.Dataset subclass.
    Each __getitem__ returns (image: Tensor, label: int, path: str).
    """
    global _TBDataset_cls

    if _TBDataset_cls is None:
        try:
            from torch.utils.data import Dataset
        except Exception as e:
            raise ImportError(
                "PyTorch is required for TBDataset but could not be imported.\n"
                f"Original error: {e}"
            ) from e

        class _TBDataset(Dataset):
            """
            PyTorch Dataset for tuberculosis chest X-ray classification.

            Expects a CSV or DataFrame with at least two columns:
              - 'image_path' : absolute or relative path to the CXR image file
              - 'label'      : 0 = TB negative, 1 = TB positive
            """

            def __init__(self, dataframe, transform=None,
                         image_col="image_path", label_col="label"):
                self.df = dataframe.reset_index(drop=True)
                self.transform = transform
                self.image_col = image_col
                self.label_col = label_col

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                image_path = str(row[self.image_col])
                label = int(row[self.label_col])

                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    raise FileNotFoundError(
                        f"Could not load image at index {idx}: {image_path}\n"
                        f"Original error: {e}"
                    ) from e

                if self.transform is not None:
                    image = self.transform(image)

                return image, label, image_path

        _TBDataset_cls = _TBDataset

    return _TBDataset_cls(dataframe, transform=transform,
                          image_col=image_col, label_col=label_col)


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────

def build_transforms(
    image_size: int = 224,
    split: str = "train",
    normalize_mean: list[float] | None = None,
    normalize_std: list[float] | None = None,
):
    """
    Build torchvision transform pipelines for train / val / test splits.

    We use ImageNet mean/std because the ResNet backbone was pretrained
    on ImageNet. Even though CXRs don't look like natural images, this
    normalisation is standard practice and tends to work well.

    Parameters
    ----------
    image_size : int
        Target image size (square). Default 224 for ResNet.
    split : str
        One of 'train', 'val', 'test'.
        Only training gets data augmentation.
    normalize_mean : list of float or None
        Per-channel means. Default: ImageNet values.
    normalize_std : list of float or None
        Per-channel stds. Default: ImageNet values.

    Returns
    -------
    torchvision.transforms.Compose
    """
    try:
        import torchvision.transforms as T
    except Exception as e:
        raise ImportError(
            "torchvision is required for build_transforms but could not be "
            f"imported.\nOriginal error: {e}"
        ) from e

    if normalize_mean is None:
        normalize_mean = [0.485, 0.456, 0.406]
    if normalize_std is None:
        normalize_std = [0.229, 0.224, 0.225]

    normalize = T.Normalize(mean=normalize_mean, std=normalize_std)

    if split == "train":
        transform = T.Compose([
            T.Resize((image_size + 20, image_size + 20)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            normalize,
        ])
    else:
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            normalize,
        ])

    return transform


# ──────────────────────────────────────────────
# Federated site partitioning
# ──────────────────────────────────────────────

def dirichlet_split(
    labels: np.ndarray | list[int],
    num_clients: int,
    alpha: float = 0.5,
    min_samples: int = 10,
    seed: int = 42,
    max_retries: int = 100,
) -> list[list[int]]:
    """
    Partition dataset indices across FL clients using a Dirichlet distribution.

    Parameters
    ----------
    labels : array-like of int
        Class label for each sample in the full dataset.
    num_clients : int
        Number of FL clients (simulated hospital sites).
    alpha : float
        Dirichlet concentration parameter. Lower = more heterogeneous.
    min_samples : int
        Minimum number of samples per client (safety check).
    seed : int
        Random seed for reproducibility.
    max_retries : int
        How many times to resample Dirichlet proportions before giving up.
        Prevents infinite loops when alpha is very low relative to dataset size.

    Returns
    -------
    list of list of int
        client_indices[i] = list of dataset indices assigned to client i.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    classes = np.unique(labels)
    num_samples = len(labels)

    # Validate floor is achievable at all
    if min_samples * num_clients > num_samples:
        raise ValueError(
            f"Impossible constraint: min_samples ({min_samples}) × num_clients "
            f"({num_clients}) = {min_samples * num_clients} exceeds total samples "
            f"({num_samples}). Lower min_samples or reduce num_clients."
        )

    # Build per-class index lists (shuffled)
    class_indices = {c: np.where(labels == c)[0].tolist() for c in classes}
    for c in classes:
        rng.shuffle(class_indices[c])

    for attempt in range(max_retries):
        client_indices = [[] for _ in range(num_clients)]

        for c in classes:
            cls_idx = class_indices[c].copy()
            n_cls = len(cls_idx)

            # Sample proportions and convert to integer counts
            proportions = rng.dirichlet(np.ones(num_clients) * alpha)
            counts = (proportions * n_cls).astype(int)

            # Fix rounding so counts sum exactly to n_cls
            remainder = n_cls - counts.sum()
            # Distribute leftover samples to the clients with largest fractional parts
            fractional = (proportions * n_cls) - counts
            top_clients = np.argsort(fractional)[::-1][:remainder]
            counts[top_clients] += 1

            # Assign class indices to clients
            start = 0
            for client_id, count in enumerate(counts):
                end = start + count
                client_indices[client_id].extend(cls_idx[start:end])
                start = end

        # Check the floor constraint
        sizes = [len(idx) for idx in client_indices]
        if min(sizes) >= min_samples:
            return client_indices

        # Log retry at high verbosity (optional — remove if too noisy)
        if attempt == 0:
            print(
                f"  [dirichlet_split] Attempt {attempt+1}: min client size = "
                f"{min(sizes)}, retrying with fresh Dirichlet draw..."
            )

    # All retries exhausted — return the best partition found, with a warning
    sizes = [len(idx) for idx in client_indices]
    import warnings
    warnings.warn(
        f"dirichlet_split: could not satisfy min_samples={min_samples} after "
        f"{max_retries} retries. Smallest client has {min(sizes)} samples. "
        f"Consider increasing alpha (current={alpha}), lowering min_samples, "
        f"or using a larger dataset. Returning best available partition.",
        UserWarning,
        stacklevel=2,
    )
    return client_indices


def uniform_split(
    n_samples: int,
    num_clients: int,
    seed: int = 42,
) -> list[list[int]]:
    """
    Partition dataset indices uniformly across clients.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    num_clients : int
        Number of clients.
    seed : int
        Random seed.

    Returns
    -------
    list of list of int
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    return [list(chunk) for chunk in np.array_split(indices, num_clients)]


# ──────────────────────────────────────────────
# Class weights for imbalanced datasets
# ──────────────────────────────────────────────

def get_class_weights(labels: list[int] | np.ndarray):
    """
    Compute per-class weights inversely proportional to class frequency.

    TB datasets are often imbalanced (fewer TB-positive than negative cases).
    Class-weighted loss helps the model not ignore the minority class.

    Parameters
    ----------
    labels : array-like of int
        Class labels for the training set.

    Returns
    -------
    torch.Tensor of shape (num_classes,)
        Weight for each class. Higher weight = rarer class.

    Example
    -------
    >>> weights = get_class_weights([0, 0, 0, 1])
    >>> # Class 1 has weight 3x class 0 (because 3 negative vs 1 positive)
    """
    try:
        import torch
    except Exception as e:
        raise ImportError(
            "PyTorch is required for get_class_weights but could not be "
            f"imported.\nOriginal error: {e}"
        ) from e

    labels = np.array(labels)
    classes, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / counts.astype(float)
    weights = weights / weights.sum() * len(classes)
    return torch.tensor(weights, dtype=torch.float32)


# ──────────────────────────────────────────────
# Mock dataset generator (for tests)
# ──────────────────────────────────────────────

def make_mock_dataset(
    save_dir: str | Path = "data/mock",
    num_images: int = 40,
    image_size: int = 64,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a tiny synthetic dataset of fake chest X-ray images.

    Images are random grayscale noise — they are NOT real X-rays.
    This dataset is used ONLY for unit tests and quick pipeline checks.

    Parameters
    ----------
    save_dir : str or Path
        Directory to save the synthetic images.
    num_images : int
        Total number of images to generate.
    image_size : int
        Width and height of each image in pixels.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns ['image_path', 'label', 'site']
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    records = []

    for i in range(num_images):
        label = int(i % 2)
        site = f"site_{(i % 5) + 1}"

        pixel_array = rng.integers(0, 256, size=(image_size, image_size), dtype=np.uint8)
        image = Image.fromarray(pixel_array, mode="L")

        filename = save_dir / f"mock_{i:04d}_label{label}.png"
        image.save(filename)

        records.append({
            "image_path": str(filename.resolve()),
            "label": label,
            "site": site,
            "dataset": "mock",
        })

    df = pd.DataFrame(records)
    csv_path = save_dir / "mock_manifest.csv"
    df.to_csv(csv_path, index=False)
    print(f"Mock dataset saved: {len(df)} images → {save_dir}")
    print(f"Manifest: {csv_path}")
    return df