"""
Train the neural network on MNIST + your collected handwriting (data/my_digits.npz).
Expects MNIST in data/ as CSV: first column = label (0-9), next 784 columns = pixels (0-255).
Combines both datasets, then trains and saves to models/ffnn_mnist_weights.npz.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from core.neural_network import NeuralNetwork


DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "my_digits.npz"
MODEL_PATH = PROJECT_ROOT / "models" / "ffnn_mnist_weights.npz"

# MNIST CSV: try these filenames in order (label in col 0, 784 pixels in cols 1:)
MNIST_CSV_NAMES = [
    "mnist_train.csv",
    "mnist_train_small (1).csv",
    "mnist_test.csv",
]


def one_hot_encode(Y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    n = Y.shape[0]
    Y_one_hot = np.zeros((n, num_classes))
    Y_one_hot[np.arange(n), Y.astype(int)] = 1
    return Y_one_hot


def load_mnist_from_csv():
    """Load MNIST from a CSV in data/: col 0 = label, cols 1-784 = pixels (0-255). Returns X (n,784), Y (n,)."""
    for name in MNIST_CSV_NAMES:
        path = DATA_DIR / name
        if path.exists():
            data = np.loadtxt(path, delimiter=",", dtype=np.float64)
            Y = data[:, 0].astype(np.int64)
            X = data[:, 1:]
            assert X.shape[1] == 784, f"Expected 784 pixel columns, got {X.shape[1]}"
            return X, Y
    return None, None


def main():
    # Load MNIST from your CSV in data/
    print("Loading MNIST from data/...")
    X_mnist, Y_mnist = load_mnist_from_csv()
    if X_mnist is None:
        print("  No MNIST CSV found. Put one of these in data/: " + ", ".join(MNIST_CSV_NAMES))
        print("  Format: first column = label (0-9), next 784 columns = pixels (0-255).")
        X_mnist = np.zeros((0, 784))
        Y_mnist = np.array([], dtype=np.int64)
    else:
        print(f"  MNIST: {X_mnist.shape[0]} samples")

    # Load your digits (optional but recommended)
    if DATA_PATH.exists():
        data = np.load(DATA_PATH)
        X_mine = data["X"]
        Y_mine = data["Y"]
        n_mine = X_mine.shape[0]
        print(f"  Your digits: {n_mine} samples")
        if X_mnist.shape[0] > 0:
            X = np.vstack([X_mnist, X_mine])
            Y = np.concatenate([Y_mnist, Y_mine])
        else:
            X, Y = X_mine, Y_mine
    else:
        print("  Your digits: none (data/my_digits.npz not found)")
        print("  To add your handwriting, run: python -m app.collect_data_app")
        X, Y = X_mnist, Y_mnist

    if X.shape[0] < 32:
        print(f"Need at least 32 samples; you have {X.shape[0]}. Add MNIST CSV or collect your digits.")
        sys.exit(1)

    Y_one_hot = one_hot_encode(Y)
    n_total = X.shape[0]
    print(f"Training on {n_total} samples total...")

    nn = NeuralNetwork(784, 16, 10)
    nn.train(X, Y_one_hot, epochs=200, learning_rate=0.1)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    nn.save(str(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}. Run the main app: python run_app.py")


if __name__ == "__main__":
    main()
