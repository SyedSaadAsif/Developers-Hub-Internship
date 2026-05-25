# =============================================================================
# Task 3: Multimodal ML – Housing Price Prediction Using Images + Tabular Data
# =============================================================================
# Objective:
#   Predict housing prices by fusing structured tabular features (sq footage,
#   bedrooms, location, etc.) with visual features extracted from house images
#   via a pre-trained CNN.
#
# Datasets:
#   - Tabular : Housing Sales CSV  (e.g. Kaggle House Prices dataset)
#                → https://www.kaggle.com/c/house-prices-advanced-regression-techniques
#                → Place as 'train.csv' in the same directory.
#   - Images  : House image folder  (JPEGs/PNGs named by house ID)
#                → Place inside an 'images/' subdirectory.
#
# Architecture:
#   Image Branch  : ResNet-18 (pre-trained on ImageNet, last FC removed)
#                   → 512-dim feature vector per image
#   Tabular Branch: Numeric imputation + batch norm layer
#   Fusion        : Concatenate → Fully connected → Price prediction
#
# Metrics:  MAE (Mean Absolute Error), RMSE (Root Mean Squared Error)
#
# Requirements:
#   pip install torch torchvision scikit-learn pandas numpy matplotlib pillow
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

TABULAR_CSV    = "train.csv"        # Kaggle Housing CSV path
IMAGE_DIR      = "images/"          # Folder containing house images (<Id>.jpg)
TARGET_COL     = "SalePrice"        # Regression target
IMG_SIZE       = 224                # CNN expects 224×224 (ResNet standard)
BATCH_SIZE     = 16
NUM_EPOCHS     = 30
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-4               # L2 regularisation
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT     = "best_multimodal.pt"   # Path to save best model weights

# ---------------------------------------------------------------------------
# Tabular features to use (subset of Kaggle House Prices columns).
# Adjust to match your own dataset's column names.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "GrLivArea",        # Above-ground living area (sq ft)
    "TotalBsmtSF",      # Total basement area
    "GarageArea",       # Garage area
    "OverallQual",      # Overall material/finish quality (1-10)
    "OverallCond",      # Overall condition rating (1-10)
    "YearBuilt",        # Original construction year
    "YearRemodAdd",     # Remodel year
    "FullBath",         # Full bathrooms
    "BedroomAbvGr",     # Bedrooms above ground
    "TotRmsAbvGrd",     # Total rooms above ground
    "LotArea",          # Lot size (sq ft)
    "Fireplaces",       # Number of fireplaces
    "GarageCars",       # Garage capacity (cars)
]


# ---------------------------------------------------------------------------
# 2. Image Transforms
# ---------------------------------------------------------------------------

# Training: add augmentation to regularise the CNN
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),          # Horizontal mirror
    transforms.ColorJitter(brightness=0.2,
                           contrast=0.2,
                           saturation=0.1),          # Slight colour variation
    transforms.ToTensor(),
    transforms.Normalize(                            # ImageNet normalisation
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Validation / inference: deterministic
VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------------------------------------------------------------------
# 3. Multimodal Dataset
# ---------------------------------------------------------------------------

class HousingDataset(Dataset):
    """
    Combines house images with tabular features into a single dataset.

    Each item returns:
        image   : Tensor [3, IMG_SIZE, IMG_SIZE]  – from CNN preprocessing
        tabular : Tensor [n_features]             – scaled numeric features
        price   : Tensor scalar                  – log1p-transformed SalePrice
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        tabular_features: list,
        scaler: StandardScaler,
        transform,
        is_train: bool = True,
        placeholder_image: bool = True,
    ):
        """
        Args:
            df               : DataFrame slice (train or val)
            image_dir        : Directory containing <Id>.jpg images
            tabular_features : List of numeric column names to use
            scaler           : Fitted StandardScaler (fit on train only)
            transform        : torchvision transforms
            is_train         : True if training (used for target inclusion)
            placeholder_image: If True, use a black placeholder when image
                               is missing (instead of raising an error).
        """
        self.df               = df.reset_index(drop=True)
        self.image_dir        = image_dir
        self.tabular_features = tabular_features
        self.transform        = transform
        self.is_train         = is_train
        self.placeholder      = placeholder_image

        # Scale tabular features
        self.tab_array = scaler.transform(
            df[tabular_features].values.astype(np.float32)
        )

        # Log1p-transform the target to reduce skew
        if TARGET_COL in df.columns:
            self.targets = np.log1p(df[TARGET_COL].values.astype(np.float32))
        else:
            self.targets = np.zeros(len(df), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def _load_image(self, house_id):
        """Load house image by ID; return placeholder if file is missing."""
        img_path = Path(self.image_dir) / f"{house_id}.jpg"

        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
        elif self.placeholder:
            # Create a black placeholder image so training can proceed
            # without requiring images for every house
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(0, 0, 0))
        else:
            raise FileNotFoundError(f"Image not found: {img_path}")

        return self.transform(img)

    def __getitem__(self, idx):
        house_id = self.df.loc[idx, "Id"]
        image    = self._load_image(house_id)
        tabular  = torch.tensor(self.tab_array[idx], dtype=torch.float32)
        price    = torch.tensor(self.targets[idx],   dtype=torch.float32)

        return image, tabular, price


# ---------------------------------------------------------------------------
# 4. Multimodal Model Architecture
# ---------------------------------------------------------------------------

class MultimodalHousingModel(nn.Module):
    """
    Two-branch neural network that fuses CNN image features with tabular
    features for housing price regression.

    Architecture overview:
        ┌──────────────────┐    ┌──────────────────┐
        │   Image Branch   │    │  Tabular Branch   │
        │  ResNet-18 CNN   │    │  FC + BatchNorm   │
        │  (512-dim out)   │    │  (128-dim out)    │
        └────────┬─────────┘    └────────┬──────────┘
                 │                       │
                 └──────────┬────────────┘
                       Concatenate
                     (512 + 128 = 640)
                            │
                      Fusion Layers
                      (640 → 256 → 1)
    """

    def __init__(self, n_tabular: int):
        """
        Args:
            n_tabular : Number of tabular (numeric) features
        """
        super().__init__()

        # ------------------------------------------------------------------
        # Image Branch – ResNet-18 pre-trained on ImageNet
        # ------------------------------------------------------------------
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove the final classification FC layer; keep the feature extractor
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])   # Output: [B, 512, 1, 1]

        # Freeze early ResNet layers to retain low-level ImageNet features
        # Only fine-tune layer4 and above
        for name, param in self.cnn.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False

        self.img_out_dim = 512   # ResNet-18 penultimate layer width

        # ------------------------------------------------------------------
        # Tabular Branch – simple FC network with batch normalisation
        # ------------------------------------------------------------------
        self.tabular_branch = nn.Sequential(
            nn.Linear(n_tabular, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.tab_out_dim = 128

        # ------------------------------------------------------------------
        # Fusion Head – concatenates both branches and predicts price
        # ------------------------------------------------------------------
        fused_dim = self.img_out_dim + self.tab_out_dim   # 512 + 128 = 640

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),   # Single output: log1p(SalePrice)
        )

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images  : [B, 3, 224, 224]
            tabular : [B, n_tabular]

        Returns:
            preds   : [B, 1]  – predicted log1p(SalePrice)
        """
        # --- Image features ---
        img_feat = self.cnn(images)          # [B, 512, 1, 1]
        img_feat = img_feat.flatten(start_dim=1)   # [B, 512]

        # --- Tabular features ---
        tab_feat = self.tabular_branch(tabular)    # [B, 128]

        # --- Fuse and predict ---
        fused = torch.cat([img_feat, tab_feat], dim=1)   # [B, 640]
        preds = self.fusion(fused)                        # [B, 1]

        return preds.squeeze(1)   # [B]


# ---------------------------------------------------------------------------
# 5. Training Loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    """
    Run one training epoch.

    Returns:
        avg_loss : Mean loss over all batches
    """
    model.train()
    total_loss = 0.0

    for images, tabular, prices in loader:
        images  = images.to(device)
        tabular = tabular.to(device)
        prices  = prices.to(device)

        optimizer.zero_grad()
        preds = model(images, tabular)
        loss  = criterion(preds, prices)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(prices)

    return total_loss / len(loader.dataset)


def validate_epoch(model, loader, criterion, device):
    """
    Run one validation epoch.

    Returns:
        avg_loss  : Mean validation loss
        all_preds : Numpy array of predictions (in original price space)
        all_true  : Numpy array of true values  (in original price space)
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []

    with torch.no_grad():
        for images, tabular, prices in loader:
            images  = images.to(device)
            tabular = tabular.to(device)
            prices  = prices.to(device)

            preds = model(images, tabular)
            loss  = criterion(preds, prices)
            total_loss += loss.item() * len(prices)

            # Inverse-transform log1p predictions back to raw price for metrics
            all_preds.extend(np.expm1(preds.cpu().numpy()))
            all_true.extend(np.expm1(prices.cpu().numpy()))

    avg_loss  = total_loss / len(loader.dataset)
    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    return avg_loss, all_preds, all_true


# ---------------------------------------------------------------------------
# 6. Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MAE and RMSE on raw (un-logged) price predictions.

    Args:
        y_true : Ground-truth prices
        y_pred : Predicted prices

    Returns:
        dict with 'mae' and 'rmse' keys
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"mae": mae, "rmse": rmse}


# ---------------------------------------------------------------------------
# 7. Training Orchestration
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, device):
    """
    Full training loop with early-stopping via best-model checkpointing.

    Args:
        model        : MultimodalHousingModel
        train_loader : DataLoader for training set
        val_loader   : DataLoader for validation set
        device       : torch.device

    Returns:
        history : dict of lists tracking loss and metrics per epoch
    """
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # LR scheduler: halve LR if val loss plateaus for 5 epochs
    scheduler  = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    criterion  = nn.MSELoss()   # MSE in log-price space → equivalent to MSLE

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss             = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, preds, true  = validate_epoch(model, val_loader, criterion, device)
        metrics                = compute_metrics(true, preds)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(metrics["mae"])
        history["val_rmse"].append(metrics["rmse"])

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"MAE: ${metrics['mae']:,.0f} | "
            f"RMSE: ${metrics['rmse']:,.0f}"
        )

        # Save checkpoint when validation improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  ✓ Checkpoint saved (val_loss={val_loss:.4f})")

    print(f"\nBest val loss: {best_val_loss:.4f}  →  saved to '{CHECKPOINT}'")
    return history


# ---------------------------------------------------------------------------
# 8. Plotting Training History
# ---------------------------------------------------------------------------

def plot_history(history: dict) -> None:
    """
    Plot training / validation loss and validation MAE + RMSE over epochs.

    Args:
        history : dict returned by train_model()
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss (MSE in log-price space)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()

    # --- MAE & RMSE ---
    axes[1].plot(epochs, history["val_mae"],  label="Val MAE ($)")
    axes[1].plot(epochs, history["val_rmse"], label="Val RMSE ($)")
    axes[1].set_title("Validation MAE & RMSE (raw price)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("USD")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("Training history saved to 'training_history.png'")


# ---------------------------------------------------------------------------
# 9. Main Pipeline
# ---------------------------------------------------------------------------

def main():
    print(f"Using device: {DEVICE}\n")

    # ------------------------------------------------------------------
    # A) Load tabular data
    # ------------------------------------------------------------------
    print(f"Loading tabular data from: {TABULAR_CSV}")
    df = pd.read_csv(TABULAR_CSV)
    print(f"  Shape: {df.shape}")

    # Keep only rows where all selected numeric features exist
    # (Drop rows missing the target or most features)
    df = df.dropna(subset=[TARGET_COL])
    print(f"  Rows after dropping null targets: {len(df)}\n")

    # ------------------------------------------------------------------
    # B) Impute missing values in tabular features before splitting
    # ------------------------------------------------------------------
    imputer = SimpleImputer(strategy="median")
    df[NUMERIC_FEATURES] = imputer.fit_transform(df[NUMERIC_FEATURES])

    # ------------------------------------------------------------------
    # C) Train / validation split
    # ------------------------------------------------------------------
    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)}\n")

    # ------------------------------------------------------------------
    # D) Fit scaler on training tabular features only
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    scaler.fit(train_df[NUMERIC_FEATURES].values.astype(np.float32))

    # ------------------------------------------------------------------
    # E) Build Datasets & DataLoaders
    # ------------------------------------------------------------------
    train_dataset = HousingDataset(
        df=train_df,
        image_dir=IMAGE_DIR,
        tabular_features=NUMERIC_FEATURES,
        scaler=scaler,
        transform=TRAIN_TRANSFORMS,
    )
    val_dataset = HousingDataset(
        df=val_df,
        image_dir=IMAGE_DIR,
        tabular_features=NUMERIC_FEATURES,
        scaler=scaler,
        transform=VAL_TRANSFORMS,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,    # Parallel image loading
        pin_memory=True,  # Faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # F) Build model
    # ------------------------------------------------------------------
    model = MultimodalHousingModel(n_tabular=len(NUMERIC_FEATURES)).to(DEVICE)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}\n")

    # ------------------------------------------------------------------
    # G) Train
    # ------------------------------------------------------------------
    history = train_model(model, train_loader, val_loader, DEVICE)

    # ------------------------------------------------------------------
    # H) Load best checkpoint and compute final metrics
    # ------------------------------------------------------------------
    print(f"\nLoading best checkpoint from '{CHECKPOINT}'...")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))

    _, final_preds, final_true = validate_epoch(
        model,
        val_loader,
        nn.MSELoss(),
        DEVICE,
    )
    final_metrics = compute_metrics(final_true, final_preds)

    print("\n=== FINAL EVALUATION (best checkpoint) ===")
    print(f"  MAE  : ${final_metrics['mae']:,.2f}")
    print(f"  RMSE : ${final_metrics['rmse']:,.2f}")

    # ------------------------------------------------------------------
    # I) Plot training history
    # ------------------------------------------------------------------
    plot_history(history)


if __name__ == "__main__":
    main()
