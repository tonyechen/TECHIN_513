"""
models/cnn_bilstm.py
--------------------
CNN + BiLSTM classifier for EDM section labeling.

Architecture:
    1D CNN  → learns local spectral/energy patterns (~0.5–2 s windows)
    BiLSTM  → learns long-range temporal structure (buildup→drop→breakdown arc)
    Linear head → 4-class output (calm / breakdown / buildup / drop)

Public API
----------
    EDMSegmentNet          – nn.Module (architecture only)
    FrameSequenceDataset   – Dataset that slices flat frame arrays into windows
    compute_class_weights  – Inverse-frequency weights for imbalanced labels
    train_model(X, y, ...)            → (model, scaler)
    predict_frames(X, model, scaler)  → y_pred  (int array, frame-level)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from collections import Counter
import joblib


# ── Architecture ──────────────────────────────────────────────────────────────

class EDMSegmentNet(nn.Module):
    """
    1D CNN → BiLSTM → classifier head.

    Input:  (batch, seq_len, n_features)  e.g. (32, 64, 21)
    Output: (batch, seq_len, n_classes)   e.g. (32, 64, 4)
    """

    def __init__(self, n_features=21, n_classes=4, lstm_hidden=128, dropout=0.3):
        super().__init__()

        # 1D CNN — treats features as channels, operates across time
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,         128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # BiLSTM — captures long-range temporal context in both directions
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)   # → (batch, features, seq_len) for Conv1d
        x = self.cnn(x)           # → (batch, 128, seq_len)
        x = x.permute(0, 2, 1)   # → (batch, seq_len, 128) for LSTM
        x, _ = self.lstm(x)      # → (batch, seq_len, 256)
        x = self.classifier(x)   # → (batch, seq_len, n_classes)
        return x


# ── Dataset ───────────────────────────────────────────────────────────────────

class FrameSequenceDataset(Dataset):
    """
    Slices a flat (N_frames, n_features) array into overlapping
    (seq_len, n_features) windows for the CNN+BiLSTM.

    seq_len=64 ≈ 1.5 s at hop=1024, sr=22050.
    """

    def __init__(self, X, y, seq_len=64, stride=32):
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len
        self.indices = list(range(0, len(X) - seq_len + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        s = self.indices[i]
        return self.X[s:s + self.seq_len], self.y[s:s + self.seq_len]


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(y, n_classes=4):
    """Inverse-frequency class weights to handle calm-heavy imbalance."""
    counts  = Counter(y.tolist())
    total   = sum(counts.values())
    weights = torch.zeros(n_classes)
    for cls in range(n_classes):
        weights[cls] = total / (n_classes * max(counts.get(cls, 1), 1))
    return weights


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    all_X, all_y,
    n_features  = 21,
    n_classes   = 4,
    seq_len     = 64,
    stride      = 16,
    batch_size  = 64,
    epochs      = 30,
    lr          = 1e-3,
    lstm_hidden = 128,
    dropout     = 0.3,
    output_dir  = './output',
    device      = None,
):
    """
    Train the CNN+BiLSTM on (all_X, all_y) frame arrays.

    Returns
    -------
    model  : trained EDMSegmentNet (best val-F1 weights restored)
    scaler : fitted StandardScaler (apply before inference)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Normalise
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(all_X).astype(np.float32)

    # Train / val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, all_y, test_size=0.15, stratify=all_y, random_state=42
    )
    print(f'Train frames: {len(X_train):,}   Val frames: {len(X_val):,}')

    train_ds = FrameSequenceDataset(X_train, y_train, seq_len, stride)
    val_ds   = FrameSequenceDataset(X_val,   y_val,   seq_len, stride=seq_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Model, loss, optimiser
    model         = EDMSegmentNet(n_features, n_classes, lstm_hidden, dropout).to(device)
    class_weights = compute_class_weights(torch.tensor(all_y), n_classes).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler     = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_dl), epochs=epochs
    )

    best_val_f1, best_weights = 0.0, None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            loss   = criterion(logits.reshape(-1, n_classes), y_b.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for X_b, y_b in val_dl:
                preds = model(X_b.to(device)).argmax(dim=-1).cpu().numpy().flatten()
                all_preds.extend(preds)
                all_true.extend(y_b.numpy().flatten())

        val_f1 = f1_score(all_true, all_preds, average='weighted', zero_division=0)
        print(f'Epoch {epoch:3d}/{epochs}  loss={total_loss/len(train_dl):.4f}  val_f1={val_f1:.3f}')

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f'           ↑ new best saved')

    # Restore best weights
    model.load_state_dict(best_weights)
    model.eval()
    print(f'\nBest val F1: {best_val_f1:.3f}')

    # Final report on val set
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_b, y_b in val_dl:
            preds = model(X_b.to(device)).argmax(dim=-1).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_true.extend(y_b.numpy().flatten())
    print('\nValidation classification report:')
    print(classification_report(
        all_true, all_preds,
        target_names=['calm', 'breakdown', 'buildup', 'drop'],
        zero_division=0
    ))

    # Save
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'edm_cnn_lstm.pt'))
    joblib.dump({
        'scaler': scaler, 'n_features': n_features, 'n_classes': n_classes,
        'lstm_hidden': lstm_hidden, 'dropout': dropout, 'seq_len': seq_len,
    }, os.path.join(output_dir, 'edm_cnn_lstm_bundle.joblib'))
    print(f'\n✅ Model  → {output_dir}/edm_cnn_lstm.pt')
    print(f'✅ Bundle → {output_dir}/edm_cnn_lstm_bundle.joblib')

    return model, scaler


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_frames(X_raw, model, scaler, seq_len=64, device=None):
    """
    Run frame-level inference on a pre-extracted feature matrix.

    Parameters
    ----------
    X_raw  : np.ndarray  (N_frames, n_features)  — unscaled features
    model  : trained EDMSegmentNet
    scaler : fitted StandardScaler from train_model()

    Returns
    -------
    y_pred : np.ndarray  (N_frames,)  — integer class predictions
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_scaled = scaler.transform(X_raw).astype(np.float32)
    X_tensor = torch.tensor(X_scaled).unsqueeze(0).to(device)  # (1, N, feat)

    model.eval()
    model.to(device)
    all_logits = []
    CHUNK = 2048
    with torch.no_grad():
        for i in range(0, X_tensor.shape[1], CHUNK):
            all_logits.append(model(X_tensor[:, i:i + CHUNK, :]))
    logits = torch.cat(all_logits, dim=1)           # (1, N, 4)
    return logits.squeeze(0).argmax(dim=-1).cpu().numpy()
