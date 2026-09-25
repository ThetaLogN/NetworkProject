import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_sample_images(images, y, save_dir, class_names):
    """Plots and saves example generated network images for normal vs. attack traffic,
    as well as the overall dataset mean image."""
    # Find indices for Benign and Attack
    idx_benign = np.where(y == 0)[0][0]
    idx_attack = np.where(y == 1)[0][0]
    overall_mean = np.mean(images, axis=0)
    
    # 1. Comparison plot (Benign, Attack, Dataset Overall Mean)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(images[idx_benign], cmap='inferno', interpolation='nearest')
    axes[0].set_title(f"Normal Flow Image (Class: {class_names[0]})")
    axes[0].axis('off')
    
    axes[1].imshow(images[idx_attack], cmap='inferno', interpolation='nearest')
    axes[1].set_title(f"Intrusion/Attack Flow Image (Class: {class_names[1]})")
    axes[1].axis('off')

    axes[2].imshow(overall_mean, cmap='inferno', interpolation='nearest')
    axes[2].set_title("Overall Dataset Mean Image (All Flows)")
    axes[2].axis('off')
    
    plt.suptitle("DeepInsight Transformed CICIDS2017 Flow Images", fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "cicids_deepinsight_all.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved sample transformed images plot to: {plot_path}")

    # 2. Standalone single dataset overall mean image
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(overall_mean, cmap='inferno', interpolation='nearest')
    ax.set_title("CICIDS2017 - Overall Dataset Mean Image (All Flows)")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    single_plot_path = os.path.join(save_dir, "cicids_deepinsight_alone.png")
    plt.savefig(single_plot_path, dpi=150)
    plt.close()
    print(f"Saved dataset overall mean image to: {single_plot_path}")

    # 3. Export overall mean as .npy
    overall_npy_path = os.path.join(save_dir, "cicids_dataset_overall_mean.npy")
    np.save(overall_npy_path, overall_mean)
    print(f"Saved dataset overall mean array to: {overall_npy_path}")


def main():
    print("=== Phase 3: Image Transformation & Dataset Generation ===")
    
    # Construct base directory of the script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Paths relative to the script location
    X_path = os.path.join(BASE_DIR, "data", "X_clean.npy")
    y_path = os.path.join(BASE_DIR, "data", "y_clean.npy")
    model_path = os.path.join(BASE_DIR, "models", "transformer.pkl")
    image_dir = os.path.join(BASE_DIR, "image")
    images_save_path = os.path.join(image_dir, "cicids_images.npy")
    labels_save_path = os.path.join(image_dir, "cicids_labels.npy")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Cleaned preprocessed data or fitted transformer model not found. Please run Phases 1 and 2 first.")

    # 1. Load preprocessed data and labels
    X_clean = np.load(X_path)
    y_clean = np.load(y_path)
    print(f"Loaded dataset of shape: {X_clean.shape}")

    # 2. Load serialized ImageTransformer model
    print(f"Loading fitted transformer model from {model_path}...")
    with open(model_path, 'rb') as f:
        it = pickle.load(f)
        
    class_names = ["BENIGN", "ATTACK"]

    # 3. Transform all tabular samples to 2D image matrices
    # Format 'scalar' returns (N, H, W) float32 arrays
    print("Transforming tabular samples to 2D image matrices...")
    transformed_images = it.transform(X_clean, img_format='scalar')
    print(f"Transformation complete. Generated image tensor shape: {transformed_images.shape}")

    # 4. Plot and save sample traffic flows
    os.makedirs(image_dir, exist_ok=True)
    plot_sample_images(transformed_images, y_clean, image_dir, class_names)

    # 5. Export generated images and labels as NumPy .npy files
    np.save(images_save_path, transformed_images)
    np.save(labels_save_path, y_clean)
    
    print(f"Saved transformed images dataset to: {images_save_path}")
    print(f"Saved labels dataset to: {labels_save_path}")
    print("=== Phase 3: Completed successfully ===")

if __name__ == "__main__":
    main()
