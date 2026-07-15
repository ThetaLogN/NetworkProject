import sys
from test_deepinsight import (
    test_image_transformer_shapes,
    test_quantile_transformation_discretization
)

if __name__ == "__main__":
    print("Running DeepInsight tests manually...")
    try:
        test_image_transformer_shapes()
        print("✓ test_image_transformer_shapes passed.")
        
        test_quantile_transformation_discretization()
        print("✓ test_quantile_transformation_discretization passed.")
        
        print("\nAll tests passed successfully!")
    except AssertionError as e:
        print(f"✗ Test assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)
