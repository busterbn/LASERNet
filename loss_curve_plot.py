import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# -------- SETTINGS --------
HISTORY_PATH = "clippedgrad/history.json"     # path to your file
SAVE_DIR = "."                    # where to save plots
SKIP_FIRST_EPOCH = True           # set False if you want full curve
# --------------------------

# Load history
with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

train_loss = np.array(history["train_loss"])
val_loss   = np.array(history["val_loss"])
train_mae  = np.array(history["train_mae"])
val_mae    = np.array(history["val_mae"])

# Epoch axis
epochs = np.arange(1, len(train_loss) + 1)

# Skip first epoch if wanted
if SKIP_FIRST_EPOCH:
    epochs     = epochs[1:]
    train_loss = train_loss[1:]
    val_loss   = val_loss[1:]
    train_mae  = train_mae[1:]
    val_mae    = val_mae[1:]

# ------------------ LOSS PLOT ------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(Path(SAVE_DIR) / "loss_curve.png", dpi=200)
plt.close()

# ------------------ MAE PLOT ------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_mae, label="Train MAE")
plt.plot(epochs, val_mae, label="Val MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Training & Validation MAE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(Path(SAVE_DIR) / "mae_curve.png", dpi=200)
plt.close()

print("Saved:")
print(" - loss_curve.png")
print(" - mae_curve.png")
