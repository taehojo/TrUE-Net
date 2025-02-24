import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2, nhead=4, num_layers=2, d_model=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb = F.relu(x_emb)
        x_emb = self.dropout(x_emb)
        out = self.transformer(x_emb)
        out = out.mean(dim=1)
        return self.fc(out)
