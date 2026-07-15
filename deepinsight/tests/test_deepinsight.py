import numpy as np
from pyDeepInsight import ImageTransformer

def test_image_transformer_shapes():
    # Generate dummy tabular data: 100 samples, 20 features
    np.random.seed(42)
    X = np.random.rand(100, 20)
    
    # Initialize transformer with a small pixel grid
    pixels = (32, 32)
    it = ImageTransformer(feature_extractor='pca', discretization='bin', pixels=pixels)
    
    # Fit
    it.fit(X)
    coords = it.coords()
    
    # Each of the 20 features must have a 2D coordinate mapped in pixel grid bounds
    assert coords.shape == (20, 2)
    assert np.all(coords[:, 0] >= 0) and np.all(coords[:, 0] < pixels[0])
    assert np.all(coords[:, 1] >= 0) and np.all(coords[:, 1] < pixels[1])
    
    # Transform with scalar format
    img_scalar = it.transform(X, img_format='scalar')
    assert img_scalar.shape == (100, 32, 32)
    
    # Transform with rgb format
    img_rgb = it.transform(X, img_format='rgb')
    assert img_rgb.shape == (100, 32, 32, 3)
    
    # Transform with pytorch format
    img_pytorch = it.transform(X, img_format='pytorch')
    assert img_pytorch.shape == (100, 3, 32, 32)


def test_quantile_transformation_discretization():
    np.random.seed(42)
    X = np.random.rand(50, 15)
    
    it = ImageTransformer(feature_extractor='pca', discretization='qtb', pixels=20)
    it.fit(X)
    coords = it.coords()
    
    assert coords.shape == (15, 2)
    assert np.all(coords >= 0) and np.all(coords < 20)
