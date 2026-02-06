import torch
import torch.nn as nn

from monai.networks.nets import resnet

class CTOnly(nn.Module):
    def __init__(self, ct_embed_dim=256, dropout=0.3):
        super().__init__()
        self.ct = resnet.resnet18(spatial_dims=3, n_input_channels=1, num_classes=ct_embed_dim, pretrained=False)
        self.head = nn.Sequential(
            nn.Linear(ct_embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

    def forward(self, ct, x_clin=None, ehr_present=None):
        emb = self.ct(ct)
        return self.head(emb)

class ClinicalMLP(nn.Module):
    def __init__(self, in_dim, embed_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class LateFusion(nn.Module):
    def __init__(self, clinical_dim, ct_embed_dim=256, clin_embed_dim=128, dropout=0.3, use_ehr_present=True):
        super().__init__()
        self.ct = resnet.resnet18(spatial_dims=3, n_input_channels=1, num_classes=ct_embed_dim, pretrained=False)
        self.clin = ClinicalMLP(clinical_dim, embed_dim=clin_embed_dim, dropout=dropout)
        self.use_ehr_present = use_ehr_present

        fusion_in = ct_embed_dim + clin_embed_dim + (1 if use_ehr_present else 0)

        self.head = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

    def forward(self, ct, x_clin, ehr_present):
        ct_emb = self.ct(ct)
        clin_emb = self.clin(x_clin)

        # Gate clinical embedding by ehr_present (so MosMed contributes 0)
        g = ehr_present.view(-1, 1)
        clin_emb = clin_emb * g

        if self.use_ehr_present:
            z = torch.cat([ct_emb, clin_emb, g], dim=1)
        else:
            z = torch.cat([ct_emb, clin_emb], dim=1)

        return self.head(z)
