#use for multi terminal setup

import flwr as fl
import pandas as pd
import os

from src.model import MobileNetAttentionModel
from src.config import DEVICE
from fl.client import SkinClient
from fl.simulation import get_client_data   # reuse your function


# Load dataset
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


# Client launcher
def start_client(cid):
    model = MobileNetAttentionModel().to(DEVICE)

    train_loader, val_loader = get_client_data(df, cid)

    client = SkinClient(model, train_loader, val_loader)

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client,
    )


if __name__ == "__main__":
    import sys
    cid = int(sys.argv[1])  # pass client id from terminal
    start_client(cid)