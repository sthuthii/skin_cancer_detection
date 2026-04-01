import torch
import torch.nn as nn
import timm


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class MobileNetAttentionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # Backbone
        self.backbone = timm.create_model(
            "mobilenetv3_large_100",
            pretrained=True,
            num_classes=0
        )

        self.feat_dim = 1280

        # Attention
        self.attn = AttentionBlock(self.feat_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)      # (B, 1280)

        features = self.attn(features)   # Apply attention

        return self.classifier(features)