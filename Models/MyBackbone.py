import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet18, resnet50,
    ResNet18_Weights, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    mobilenet_v3_small, mobilenet_v3_large, MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights,
)

# --------- helpers ----------
class MLP(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=256, out_dim=256, usenorm=True, usedout=False):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(hidden_dim, out_dim)
        self.usenorm = usenorm
        self.usedout = usedout
        self.dout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.l1(x)
        if self.usenorm:
            x = self.bn(x)
        x = self.relu(x)
        if self.usedout:
            x = self.dout(x)
        x = self.l2(x)
        return x


# --------- unified model ----------
class MyBackbone(nn.Module):
    """
    Supports:
      - 'resnet18', 'resnet50'
      - 'densenet121'
      - 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3'
      - 'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'

    Returns: logits, z, p (same as your MyResNet)
    """
    def __init__(self, arch: str, pretrain: bool, num_classes: int, projectDim: int):
        super().__init__()
        self.arch = arch.lower()

        # ---------------- ResNets ----------------
        if self.arch == "resnet50":
            self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrain else None)
            nfea = self.net.fc.in_features
            self.encoder = nn.Sequential(*list(self.net.children())[:-1])  # up to avgpool

        elif self.arch == "resnet18":
            self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrain else None)
            nfea = self.net.fc.in_features
            self.encoder = nn.Sequential(*list(self.net.children())[:-1])  # up to avgpool

        # ---------------- DenseNet ----------------
        elif self.arch == "densenet121":
            self.net = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrain else None)
            nfea = self.net.classifier.in_features
            self.encoder = nn.Sequential(
                self.net.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

        # ---------------- EfficientNet ----------------
        elif self.arch in {"efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"}:
            if self.arch == "efficientnet_b0":
                weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrain else None
                self.net = efficientnet_b0(weights=weights)
            elif self.arch == "efficientnet_b1":
                weights = EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrain else None
                self.net = efficientnet_b1(weights=weights)
            elif self.arch == "efficientnet_b2":
                weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrain else None
                self.net = efficientnet_b2(weights=weights)
            else:  # b3
                weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrain else None
                self.net = efficientnet_b3(weights=weights)

            nfea = self.net.classifier[1].in_features
            self.encoder = nn.Sequential(self.net.features, self.net.avgpool)

        # ---------------- MobileNet ----------------
        elif self.arch == "mobilenet_v2":
            self.net = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2 if pretrain else None)
            # classifier = [Dropout, Linear]
            nfea = self.net.classifier[1].in_features
            self.encoder = nn.Sequential(
                self.net.features,
                nn.AdaptiveAvgPool2d((1, 1)),
            )

        elif self.arch in {"mobilenet_v3_small", "mobilenet_v3_large"}:
            if self.arch == "mobilenet_v3_small":
                self.net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrain else None)
            else:
                self.net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrain else None)

            # classifier = [Linear, Hardswish, Dropout, Linear]
            # final linear is classifier[3]
            nfea = self.net.classifier[3].in_features
            self.encoder = nn.Sequential(
                self.net.features,
                nn.AdaptiveAvgPool2d((1, 1)),
            )

        else:
            raise ValueError(
                f"Unknown arch='{arch}'. Use one of: "
                f"resnet18, resnet50, densenet121, "
                f"efficientnet_b0/b1/b2/b3, "
                f"mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large"
            )

        # SSL head
        self.projector = MLP(nfea, projectDim * 2, projectDim)
        self.predictor = nn.Sequential(
            nn.Linear(projectDim, projectDim * 2),
            nn.BatchNorm1d(projectDim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(projectDim * 2, projectDim),
        )

        # classifier head
        self.fc = MLP(projectDim, projectDim // 2, num_classes, usenorm=False, usedout=True)

    def freezeNet(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreezeNet(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)     # [B, nfea]
        z = self.projector(h)       # [B, projectDim]
        z_ = F.relu(z)
        logits = self.fc(z_)        # [B, num_classes]
        p = self.predictor(z_)      # [B, projectDim]
        return logits, z, p
