"""Generic tabular-to-image pipeline based on DeepInsight.

Feed it any CSV with a label column and it produces:
  - images.npy        (N, H, W) float32 image tensor
  - labels.npy        (N,) integer-encoded labels
  - class_names.txt   mapping index -> original label value
  - feature_names.txt features used (numeric columns only)
  - pipeline.pkl      fitted normalizer + ImageTransformer, reusable on new data
  - samples.png       one sample image per class, for a quick sanity check

Usage:
    python3 pipeline.py data.csv --label-column Label --output-dir out
    python3 pipeline.py data.csv --label-column Label --binary-benign BENIGN --pixels 40
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyDeepInsight import ImageTransformer
from sklearn.preprocessing import MinMaxScaler


class TabularImagePipeline:
    """Normalizes tabular features and maps them to 2D images via DeepInsight."""

    def __init__(self, pixels=(40, 40), feature_extractor='pca', discretization='qtb'):
        self.normalizer = MinMaxScaler()
        self.it = ImageTransformer(feature_extractor=feature_extractor,
                                   discretization=discretization, pixels=pixels)

    def fit(self, X):
        X_norm = self.normalizer.fit_transform(X)
        self.it.fit(X_norm)
        return self

    def transform(self, X, img_format='scalar'):
        X_norm = np.clip(self.normalizer.transform(X), 0, 1)
        return self.it.transform(X_norm, img_format=img_format)

    def fit_transform(self, X, img_format='scalar'):
        return self.fit(X).transform(X, img_format=img_format)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


def load_dataset(csv_path, label_column, binary_benign=None):
    """Loads a CSV and returns (X, y, feature_names, class_names).

    Cleans inf/NaN rows, keeps only numeric feature columns, and encodes
    labels as integers (binary if binary_benign is given, multiclass otherwise).
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. "
                         f"Available columns: {list(df.columns)}")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    labels = df[label_column]
    features = df.drop(columns=[label_column]).select_dtypes(include=[np.number])
    if features.shape[1] == 0:
        raise ValueError("No numeric feature columns found in the dataset.")

    if binary_benign is not None:
        y = np.where(labels.astype(str) == binary_benign, 0, 1)
        class_names = [binary_benign, "OTHER"]
    else:
        class_names, y = np.unique(labels.astype(str), return_inverse=True)
        class_names = class_names.tolist()

    return features.values, y, features.columns.tolist(), class_names


def plot_class_samples(images, y, class_names, save_path):
    """Saves a figure with the first sample image of each class."""
    classes = np.unique(y)
    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 4.5))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, classes):
        idx = np.where(y == c)[0][0]
        ax.imshow(images[idx], cmap='inferno', interpolation='nearest')
        ax.set_title(class_names[c])
        ax.axis('off')
    plt.suptitle("DeepInsight sample images per class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run(csv_path, label_column, output_dir, binary_benign=None, pixels=40,
        feature_extractor='pca', discretization='qtb'):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {csv_path}...")
    X, y, feature_names, class_names = load_dataset(csv_path, label_column, binary_benign)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} numeric features, "
          f"{len(class_names)} classes")

    pipeline = TabularImagePipeline(pixels=(pixels, pixels),
                                    feature_extractor=feature_extractor,
                                    discretization=discretization)
    print(f"Fitting and transforming to {pixels}x{pixels} images...")
    images = pipeline.fit_transform(X)
    print(f"Generated image tensor: {images.shape}")

    np.save(os.path.join(output_dir, "images.npy"), images)
    np.save(os.path.join(output_dir, "labels.npy"), y)
    with open(os.path.join(output_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))
    with open(os.path.join(output_dir, "class_names.txt"), "w") as f:
        f.write("\n".join(f"{i}\t{name}" for i, name in enumerate(class_names)))
    pipeline.save(os.path.join(output_dir, "pipeline.pkl"))
    plot_class_samples(images, y, class_names, os.path.join(output_dir, "samples.png"))

    print(f"All outputs saved to: {output_dir}")
    return images, y


def main():
    parser = argparse.ArgumentParser(description="Convert a tabular CSV dataset to DeepInsight images.")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument("--label-column", required=True, help="Name of the label column")
    parser.add_argument("--output-dir", required=True, help="Directory for generated outputs")
    parser.add_argument("--binary-benign", default=None,
                        help="If set, labels equal to this value become 0 and all others 1")
    parser.add_argument("--pixels", type=int, default=40, help="Image side length (default: 40)")
    parser.add_argument("--feature-extractor", default="pca", choices=["pca", "kpca", "tsne"],
                        help="Dimensionality reduction for the feature layout (default: pca)")
    parser.add_argument("--discretization", default="qtb", choices=["bin", "qtb"],
                        help="Coordinate discretization method (default: qtb)")
    args = parser.parse_args()

    run(args.csv_path, args.label_column, args.output_dir,
        binary_benign=args.binary_benign, pixels=args.pixels,
        feature_extractor=args.feature_extractor, discretization=args.discretization)


if __name__ == "__main__":
    main()
