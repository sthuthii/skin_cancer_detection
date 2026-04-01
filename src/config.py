import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# ======================
# CONFIG (CPU OPTIMIZED)
# ======================
CFG = {
    "image_size": 128,          # 🔥 reduced for CPU
    "batch_size": 8,            # 🔥 smaller batch
    "epochs": 1,                # FL uses per-round training
    "num_classes": 7,
    "lr": 1e-4,
    "use_amp": False,           # ❗ disable AMP (CPU)
    "checkpoint_dir": "./checkpoints",
    "grad_clip": 1.0
}

DEVICE = torch.device("cpu")   # 🔥 force CPU

os.makedirs(CFG["checkpoint_dir"], exist_ok=True)

CKPT_PATH = os.path.join(CFG["checkpoint_dir"], "best_model.pt")

# ======================
# DATA LOADING (LOCAL)
# ======================

# 🔥 CHANGE THIS TO YOUR LOCAL PATH
DATA_DIR = "data/HAM10000"

df = pd.read_csv(os.path.join(DATA_DIR, "HAM10000_metadata.csv"))

# ======================
# LABEL ENCODING
# ======================
label_map = {
    "nv": 0,
    "mel": 1,
    "bkl": 2,
    "bcc": 3,
    "akiec": 4,
    "df": 5,
    "vasc": 6
}

df["label"] = df["dx"].map(label_map)

# ======================
# IMAGE PATHS
# ======================
img1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
img2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")

def get_path(x):
    p1 = os.path.join(img1, x + ".jpg")
    p2 = os.path.join(img2, x + ".jpg")
    return p1 if os.path.exists(p1) else p2

df["path"] = df["image_id"].apply(get_path)

# ======================
# TRAIN / VAL SPLIT (GLOBAL - ONLY FOR BASELINE)
# ======================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

print("Total samples:", len(df))
print("Train samples:", len(train_df))
print("Val samples:", len(val_df))