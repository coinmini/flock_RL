from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import requests
import json
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict
from dotenv import load_dotenv

load_dotenv()

FED_LEDGER_BASE_URL = "https://fed-ledger-prod.flock.io/api/v1"
FLOCK_API_KEY = os.environ.get("FLOCK_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
TASK_ID = os.environ.get("TASK_ID")

class MLP(nn.Module):
    """Simple Multi-Layer Perceptron"""

    def __init__(self, in_dim: int, out_dim: int, hidden=(256, 256), act=nn.ReLU):
        super().__init__()
        layers = []
        last_dim = in_dim

        for h in hidden:
            layers.append(nn.Linear(last_dim, h))
            layers.append(act())
            last_dim = h

        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleDataset(Dataset):
    """Simple dataset wrapper for numpy arrays"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(data_file: str):
    """
    Load training data from .npz file

    Args:
        data_file: Path to the .npz data file

    Returns:
        X_train, Info_train
    """
    print(f"Loading data from {data_file}...")

    data = np.load(data_file)
    X_train, Info_train = data['X'], data['Info']

    print(f"X_train shape: {X_train.shape}")
    print(f"Info_train shape: {Info_train.shape}")

    return X_train, Info_train


def prepare_labels(Info: np.ndarray) -> np.ndarray:
    """
    Create labels from Info array using heuristic optimal allocation.

    The reward function in env.py is:
        reward = (rebate_bps/10000 * filled + punish * unfilled).sum() / log1p(qty)

    Where:
        - filled = min(alloc, capacity)
        - capacity = latest_vol * cap_window * fill_rate
        - cap_window = 1.0 + 0.6 * (1 - exp(-0.9 * duration))
        - unfilled = alloc - filled (punish is negative, so this reduces reward)

    Optimal strategy: allocate to venues where we can fill AND get good rebates,
    while avoiding venues where we'd have unfilled orders (penalty).

    The environment infers V from Info columns: V = (cols - 3) // 4
    Each venue has 4 columns: [fill_rate, rebate_bps, punish, latest_vol]
    """
    N = Info.shape[0]
    m = Info.shape[1]
    start = 3
    V = (m - start) // 4

    qty = Info[:, 0].astype(np.float32)        # shape: (N,)
    duration = Info[:, 1].astype(np.float32)   # shape: (N,)

    # Extract per-venue data
    cols = [start + 4 * j + k for j in range(V) for k in range(4)]
    block = Info[:, cols].astype(np.float32).reshape(N, V, 4)

    # Actual data order: [latest_vol, fill_rate, rebate_bps, punish]
    latest_vol = np.maximum(block[:, :, 0], 0.0)     # shape: (N, V)
    fill_rate = np.clip(block[:, :, 1], 0.0, 1.0)   # shape: (N, V)
    rebate_bps = block[:, :, 2]                      # shape: (N, V)
    punish = block[:, :, 3]                          # shape: (N, V) - negative values

    # Compute capacity exactly as in env.py
    cap_window = 1.0 + 0.6 * (1.0 - np.exp(-0.9 * duration))  # shape: (N,)
    capacity = latest_vol * cap_window[:, None] * fill_rate   # shape: (N, V)

    # Score each venue by expected reward contribution
    # If we allocate x to a venue:
    #   - If x <= capacity: reward = rebate_bps/10000 * x (positive)
    #   - If x > capacity: reward = rebate_bps/10000 * capacity + punish * (x - capacity)
    #                      The unfilled part gets punished (punish is negative)

    # Heuristic: prioritize venues with high (rebate * capacity) and low punish
    # We want to allocate proportionally to capacity, but weighted by rebate/punish ratio

    rebate_per_unit = rebate_bps / 10000.0  # Convert bps to decimal

    # Expected value per unit allocated (assuming we stay within capacity)
    # Higher rebate = better, lower (more negative) punish = worse if we exceed capacity
    # Score = capacity * rebate - (we don't want to exceed capacity, so weight by capacity)

    # Simple approach: allocate proportionally to capacity, weighted by rebate attractiveness
    # Venues with higher capacity AND higher rebates should get more allocation
    score = capacity * (rebate_per_unit - punish)  # punish is negative, so -punish is positive

    # Ensure non-negative scores
    score = np.maximum(score, 1e-8)

    # Normalize to get allocation probabilities
    labels = score / score.sum(axis=1, keepdims=True)

    return labels.astype(np.float32)


def train_mlp(
    data_file: str,
    output_dir: str = "runs",
    model_name: str = "mlp_example",
    hidden: Tuple[int, ...] = (256, 256),
    batch_size: int = 128,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    validation_split: float = 0.2,
    device: str = None,
    seed: int = 42,
) -> Dict:
    """
    Train a MLP model

    Args:
        data_file: Path to the .npz training data file
        output_dir: Directory to save model outputs
        model_name: Name for the saved model
        hidden: Hidden layer dimensions
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        validation_split: Fraction of training data to use for validation (0.0-1.0)
        device: Device to train on (cuda/cpu)
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing training logs
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    X_all, Info_all = load_data(data_file)

    # Prepare labels
    y_all = prepare_labels(Info_all)

    # Split data into train and validation sets
    n_samples = len(X_all)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    val_size = int(n_samples * validation_split)
    train_size = n_samples - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_val = X_all[val_indices]
    y_val = y_all[val_indices]

    print("\nData split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Create datasets and dataloaders
    train_dataset = SimpleDataset(X_train, y_train)
    val_dataset = SimpleDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Create model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = MLP(input_dim, output_dim, hidden=hidden).to(device)

    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    # Training logs
    log = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_time": [],
    }

    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience = 20  # Stop if no improvement for 20 epochs
    patience_counter = 0

    print("\nStarting training...")
    total_start = time.time()

    # Training loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        epoch_time = time.time() - epoch_start

        # Log metrics
        log["epoch"].append(epoch)
        log["train_loss"].append(avg_train_loss)
        log["val_loss"].append(avg_val_loss)
        log["train_time"].append(epoch_time)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[Epoch {epoch:4d}/{epochs}] "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f} | "
                f"Best: {best_val_loss:.6f} (ep {best_epoch}) | "
                f"Time: {epoch_time:.2f}s"
            )

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    total_time = time.time() - total_start
    print(f"\nTraining completed in {total_time:.2f}s")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)
        print(f"Restored best model from epoch {best_epoch}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save PyTorch model
    pt_path = os.path.join(output_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"Saved PyTorch model to {pt_path}")


    # Convert to ONNX
    print("\nConverting model to ONNX format...")
    model.eval()
    dummy_input = torch.randn(1, input_dim).to(device)
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=18,
        do_constant_folding=True,
    )
    print(f"Saved ONNX model to {onnx_path}")

    # Save training log
    log_path = os.path.join(output_dir, f"{model_name}_log.npz")
    np.savez(log_path, **log)
    print(f"Saved training log to {log_path}")

    return log


def upload_to_huggingface(
    folder_path: str,
    repo_id: str,
    token: str = None,
    commit_message: str = "Upload model",
):
    """
    Upload folder to Hugging Face Hub

    Args:
        folder_path: Path to the folder to upload
        repo_id: Hugging Face repository ID (e.g., "username/model-name")
        token: Hugging Face API token (or set HF_TOKEN environment variable)
        commit_message: Commit message for the upload
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "Error: huggingface_hub not installed. Install with: pip install huggingface_hub"
        )
        return

    print("\nUploading folder to Hugging Face...")
    print(f"Repository: {repo_id}")
    print(f"Folder path: {folder_path}")

    # Get token from environment if not provided
    if token is None:
        token = HF_TOKEN
        if token is None:
            print(
                "Error: No Hugging Face token provided. Set HF_TOKEN in .env file or pass token parameter."
            )
            return

    # Initialize API
    api = HfApi()

    try:
        # Create repo if it doesn't exist
        try:
            api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
            print(f"Repository {repo_id} ready")
        except Exception as e:
            print(f"Note: {e}")

        # Upload folder
        commit_info = api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            token=token,
            commit_message=commit_message,
        )
        commit_hash = commit_info.oid

        print(f"Successfully uploaded to https://huggingface.co/{repo_id}")
        return commit_hash

    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")


def submit_task(
    task_id: int, hg_repo_id: str, model_filename: str, revision: str
):
    payload = json.dumps(
        {
            "task_id": task_id,
            "data": {
                "hg_repo_id": hg_repo_id,
                "model_filename": model_filename,
                "revision": revision,
            },
        }
    )
    headers = {
        "flock-api-key": FLOCK_API_KEY,
        "Content-Type": "application/json",
    }
    response = requests.request(
        "POST",
        f"{FED_LEDGER_BASE_URL}/tasks/submit-result",
        headers=headers,
        data=payload,
        timeout=30,
    )
    if response.status_code != 200:
        raise Exception(f"Failed to submit task: {response.text}")
    return response.json()



if __name__ == "__main__":
    # Parse command line argument for training data file
    if len(sys.argv) < 2:
        print("Usage: python train.py <data_file>")
        print("Example: python train.py data/train.npz")
        sys.exit(1)

    data_file = sys.argv[1]
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Training data file not found: {data_file}")

    print("=" * 70)
    print("RL TRAINING AND VALIDATION EXAMPLE")
    print("=" * 70)
    print(f"Training data file: {data_file}")

    if TASK_ID is None:
        raise Exception("TASK_ID is not set in .env file")

    # STEP 1: TRAINER - Train the model
    print("\n[Step 1] Training model...")
    log = train_mlp(
        data_file=data_file,
        output_dir="runs",
        model_name="mlp_example",
        hidden=(256, 256),
        batch_size=128,
        epochs=100,
        learning_rate=1e-3,
        weight_decay=1e-5,
        validation_split=0.2,
        seed=42,
    )
    print("[Step 1] ✓ Training complete!")