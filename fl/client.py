import flwr as fl
import torch
import torch.nn as nn
from src.train import train_one_epoch
from src.validate import validate
from src.model import FusionModel
from src.config import CFG, DEVICE


class SkinClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()

    # 🔹 Send model weights to server
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    # 🔹 Receive global weights from server
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    #Local training (1 epoch per round)
    def fit(self, parameters, config):
        self.set_parameters(parameters)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=CFG["lr"])
        scaler = torch.amp.GradScaler("cuda")

        train_loss = train_one_epoch(
            self.model,
            self.train_loader,
            self.criterion,
            optimizer,
            scaler
        )

        print(f"[Client] Train Loss: {train_loss:.4f}")

        return self.get_parameters(config), len(self.train_loader.dataset), {}

    # Validation
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        acc, auc, f1 = validate(self.model, self.val_loader, self.criterion)

        print(f"[Client] Val Acc: {acc:.3f}, AUC: {auc:.4f}, F1: {f1:.4f}")

        return float(1 - acc), len(self.val_loader.dataset), {
            "auc": float(auc) if not torch.isnan(torch.tensor(auc)) else 0.0,
            "f1": float(f1),
        }