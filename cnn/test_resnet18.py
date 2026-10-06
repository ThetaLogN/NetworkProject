import os
import sys

# Permette l'esecuzione sia dalla root che dall'interno della cartella cnn
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from torchinfo import summary
try:
    from cnn.resnet18 import get_standard_resnet18
except ModuleNotFoundError:
    from resnet18 import get_standard_resnet18

model = get_standard_resnet18(num_classes=2)
# Genera un report completo per l'input 40x40 di DeepInsight
summary(model, input_size=(1, 3, 40, 40), col_names=["input_size", "output_size", "num_params", "mult_adds"])
