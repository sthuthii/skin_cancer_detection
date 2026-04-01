import flwr as fl
import pandas as pd
import os

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataset import SkinDataset, get_tf
from src.model import MobileNetAttentionModel   # 🔥 updated model
from src.config import CFG, DEVICE


# ======================
# LOAD DATA (LOCAL)
# ======================
DATA_DIR = "data/HAM10000"

df = pd.read_csv(os.path.join(DATA_DIR, "HAM10000_metadata.csv"))

label_map = {"nv":0,"mel":1,"bkl":2,"bcc":3,"akiec":4,"df":5,"vasc":6}
df["label"] = df["dx"].map(label_map)

img1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
img2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")

def get_path(x):
    p1 = os.path.join(img1, x + ".jpg")
    p2 = os.path.join(img2, x + ".jpg")
    return p1 if os.path.exists(p1) else p2

df["path"] = df["image_id"].apply(get_path)


# ======================
# NON-IID CLIENT SPLIT 🔥
# ======================
def get_client_data(df, client_id):

    if client_id == 0:
        df_client = df[df["label"].isin([0, 1])]

    elif client_id == 1:
        df_client = df[df["label"].isin([2, 3])]

    else:
        df_client = df[df["label"].isin([4, 5, 6])]

    df_client = df_client.sample(frac=1, random_state=42)

    train_df, val_df = train_test_split(
        df_client,
        test_size=0.2,
        stratify=df_client["label"],
        random_state=42
    )

    print(f"\nClient {client_id}")
    print("Train:", len(train_df), "Val:", len(val_df))
    print(df_client["label"].value_counts())

    train_loader = DataLoader(
        SkinDataset(train_df, get_tf("train")),
        batch_size=CFG["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        SkinDataset(val_df, get_tf("val")),
        batch_size=CFG["batch_size"],
        shuffle=False
    )

    return train_loader, val_loader


# ======================
# CLIENT FUNCTION
# ======================
from fl.client import SkinClient   # keep import here to avoid circular issues

def client_fn(cid: str):
    model = MobileNetAttentionModel().to(DEVICE)

    train_loader, val_loader = get_client_data(df, int(cid))

    return SkinClient(model, train_loader, val_loader)


# ======================
# START SIMULATION
# ======================
if __name__ == "__main__":
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=3),  # 🔥 reduced for CPU
    )