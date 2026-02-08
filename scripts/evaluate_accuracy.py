"""
Load the trained model and report accuracy on the MNIST test set (data/mnist_test.csv).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from core.neural_network import NeuralNetwork


DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "ffnn_mnist_weights.npz"
TEST_CSV = DATA_DIR / "mnist_test.csv"


def main():
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Train first: python scripts/train_on_my_data.py")
        sys.exit(1)
    if not TEST_CSV.exists():
        print(f"Test set not found at {TEST_CSV}. Put mnist_test.csv in data/.")
        sys.exit(1)

    print("Loading model...")
    nn = NeuralNetwork.load(str(MODEL_PATH))

    print(f"Loading test set from {TEST_CSV}...")
    data = np.loadtxt(TEST_CSV, delimiter=",", dtype=np.float64)
    Y = data[:, 0].astype(np.int64)
    X = data[:, 1:]
    n = X.shape[0]
    print(f"  {n} test samples")

    # Batch prediction: forward_pass returns (n, 10), argmax(axis=1) gives (n,) predicted labels
    probs = nn.forward_pass(X)
    Y_pred = np.argmax(probs, axis=1)
    acc = np.mean(Y_pred == Y)
    print(f"Test accuracy: {acc:.2%} ({np.sum(Y_pred == Y)} / {n} correct)")


if __name__ == "__main__":
    main()
