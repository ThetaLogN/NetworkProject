import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from pyDeepInsight import ImageTransformer

def plot_feature_layout(it, feature_names, save_dir):
    """Plots the 2D PCA representation of network features and the pixel grid mapping."""
    coords = it.coords()
    xrot = it._xrot
    pixels = it.pixels
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: PCA Projection and Bounding box
    ax = axes[0]
    ax.scatter(xrot[:, 0], xrot[:, 1], color='crimson', s=60, edgecolors='black', alpha=0.8, zorder=3)
    
    # Annotate some interesting network features to show clustering
    for i, txt in enumerate(feature_names):
        if any(keyword in txt for keyword in ["Packets", "Duration", "Flag", "Length", "IAT"]):
            if i % 4 == 0:
                ax.annotate(txt, (xrot[i, 0], xrot[i, 1]), fontsize=7, alpha=0.9, weight='bold')
            
    if len(xrot) >= 3:
        hull = ConvexHull(xrot)
        for simplex in hull.simplices:
            ax.plot(xrot[simplex, 0], xrot[simplex, 1], 'k--', alpha=0.5)
            
    ax.set_title("2D Projection of Network Feature Coordinates", fontsize=12)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, linestyle='--', alpha=0.5)

    # Right: Integer Pixel Grid Coordinates mapping
    ax = axes[1]
    ax.scatter(coords[:, 1], coords[:, 0], color='navy', s=60, edgecolors='black', alpha=0.8, zorder=3)
    
    for i, txt in enumerate(feature_names):
         if any(keyword in txt for keyword in ["Packets", "Duration", "Flag", "Length", "IAT"]):
            if i % 4 == 0:
                ax.annotate(txt, (coords[i, 1], coords[i, 0]), fontsize=7, alpha=0.9)
            
    ax.set_xlim(-1, pixels[1])
    ax.set_ylim(-1, pixels[0])
    ax.set_title(f"Discretized Feature Grid Mapping ({pixels[0]}x{pixels[1]})", fontsize=12)
    ax.set_xlabel("Pixel Column (Width)")
    ax.set_ylabel("Pixel Row (Height)")
    ax.grid(True, which='both', color='gray', linestyle=':', linewidth=0.5)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "cicids_feature_layout.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved feature layout plot to: {plot_path}")


def main():
    print("=== Phase 2: Feature Mapping & Bounding Box Layout Fitting ===")
    
    # Construct base directory of the script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Paths relative to the script location
    X_path = os.path.join(BASE_DIR, "data", "X_clean.npy")
    features_path = os.path.join(BASE_DIR, "data", "feature_names.txt")
    model_dir = os.path.join(BASE_DIR, "models")
    model_save_path = os.path.join(model_dir, "transformer.pkl")
    image_dir = os.path.join(BASE_DIR, "image")
    
    if not os.path.exists(X_path) or not os.path.exists(features_path):
        raise FileNotFoundError("Cleaned preprocessed data or features text file not found. Please run Phase 1 first.")

    # 1. Load data
    print(f"Loading cleaned dataset from {X_path}...")
    X_clean = np.load(X_path)
    
    with open(features_path, "r") as f:
        feature_names = [line.strip() for line in f.readlines()]
        
    print(f"Loaded features list with {len(feature_names)} features.")

    # 2. Fit ImageTransformer (40x40 pixel resolution)
    grid_size = (40, 40)
    print(f"Fitting ImageTransformer with PCA & Quantile Binning onto a {grid_size} grid...")
    it = ImageTransformer(feature_extractor='pca', discretization='qtb', pixels=grid_size)
    it.fit(X_clean)
    print("Transformer layout fitted successfully.")

    # 3. Plot and save feature layout
    os.makedirs(image_dir, exist_ok=True)
    plot_feature_layout(it, feature_names, image_dir)

    # 4. Serialize and save fitted transformer
    os.makedirs(model_dir, exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump(it, f)
        
    print(f"Fitted layout transformer serialized and saved to: {model_save_path}")
    print("=== Phase 2: Completed successfully ===")

if __name__ == "__main__":
    main()
