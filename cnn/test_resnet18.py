import os
import sys
import torch
from torch.profiler import profile, ProfilerActivity

# Permette l'esecuzione sia dalla root che dall'interno della cartella cnn
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from torchinfo import summary
try:
    from cnn.resnet18 import get_standard_resnet18
except ModuleNotFoundError:
    from resnet18 import get_standard_resnet18

# 1. Inizializzazione del modello in modalità valutazione
model = get_standard_resnet18(num_classes=2)
model.eval()

# 2. Riepilogo architetturale con torchinfo
print("=== 1. RIEPILOGO ARCHITETTURA (torchinfo) ===")
summary(model, input_size=(1, 3, 40, 40), col_names=["input_size", "output_size", "num_params", "mult_adds"])

# 3. Profilazione dell'overhead di UNA decisione con torch.profiler
print("\n=== 2. OVERHEAD DECISIONE SINGOLA FOTO (torch.profiler) ===")

# Simuliamo l'arrivo di 1 singola foto DeepInsight (40x40 su 3 canali)
img = torch.randn(1, 3, 40, 40)

# Breve riscaldamento (warmup) per stabilizzare la cache della CPU
with torch.no_grad():
    for _ in range(5):
        _ = model(img)

# Profilazione ufficiale della decisione
with torch.no_grad():
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True
    ) as prof:
        decisione = model(img) # <-- Il modello elabora la foto ed emette la decisione

# 4. Stampa la tabella ufficiale dei 10 operatori con maggior costo di tempo e memoria
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# 5. Salva la timeline visuale interattiva
prof.export_chrome_trace("decision_trace.json")
print("Traccia salvata in: decision_trace.json (aprila su ui.perfetto.dev per vederla graficamente)")
