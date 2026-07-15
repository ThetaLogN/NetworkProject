import os
import numpy as np
import pandas as pd
from utils import Normalizer

def main():
    print("=== Phase 1: Preprocessing & Normalization ===")
    
    # Construct base directory of the script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Paths relative to the script location
    raw_data_path = os.path.join(BASE_DIR, "data", "CICIDS2017_sample.csv")
    X_save_path = os.path.join(BASE_DIR, "data", "X_clean.npy")
    y_save_path = os.path.join(BASE_DIR, "data", "y_clean.npy")
    features_save_path = os.path.join(BASE_DIR, "data", "feature_names.txt")
    
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_data_path}. Please download it first.")

    # 1. Load the dataset
    print(f"Loading raw dataset from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)
    print(f"Raw dataset shape: {df.shape}")

    # 2. Data Cleaning
    # Strip spaces from column names
    df.columns = df.columns.str.strip()
    
    # Replace inf and -inf values with NaN, then drop NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"Dataset shape after removing NaN/infinite rows: {df.shape}")

    # Separate features and labels
    X = df.drop(columns=["Label"]).values
    feature_names = df.drop(columns=["Label"]).columns.tolist()
    labels = df["Label"].values
    
    # Encode target classes to binary: 0 for BENIGN, 1 for Attack
    y = np.where(labels == "BENIGN", 0, 1)
    print(f"Binary class distribution: BENIGN: {np.sum(y == 0)} | ATTACK: {np.sum(y == 1)}")

    # 3. Normalize features to [0, 1] range
    print("Normalizing features...")
    normalizer = Normalizer()
    X_norm = normalizer.fit_transform(X)
    
    # 4. Save results to disk
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    np.save(X_save_path, X_norm)
    np.save(y_save_path, y)
    
    with open(features_save_path, "w") as f:
        f.write("\n".join(feature_names))
        
    print(f"Preprocessed features saved to: {X_save_path} (Shape: {X_norm.shape})")
    print(f"Preprocessed labels saved to: {y_save_path} (Shape: {y.shape})")
    print(f"Feature names saved to: {features_save_path}")
    print("=== Phase 1: Completed successfully ===")

if __name__ == "__main__":
    main()
