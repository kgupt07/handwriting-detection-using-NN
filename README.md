# Handwriting Digit Recognition

A small desktop app that uses a neural network (trained on MNIST) to recognize handwritten digits. Aimed at children: the app asks for a number, the child writes it, and the computer checks the answer.

## What it does

1. **Opens a window** with a prompt like “Write the number: 7”.
2. **You draw** a digit on the white canvas with the mouse.
3. **Click “Predict”** – your drawing is preprocessed to match MNIST format (28×28, white-on-black) and passed to the neural network.
4. **You get feedback**: “Correct! Well done!” or “You wrote X. Try writing 7!”.
5. **“New number”** gives another random digit to write; **“Clear”** clears the canvas.

## File structure

```
handwriting-detection-using-NN/
├── README.md
├── requirements.txt
├── run_app.py                # Run the main handwriting quiz app
├── core/                     # Shared library (NN + preprocessing)
│   ├── neural_network.py     # FFNN: load, predict, train
│   └── preprocessing.py      # Canvas image → MNIST-like 28×28
├── app/                      # GUI applications
│   ├── handwriting_app.py   # Main app: draw digit, get feedback
│   └── collect_data_app.py   # Collect your handwriting for training
├── scripts/                  # CLI: training and evaluation
│   ├── train_on_my_data.py  # Train on MNIST + your digits
│   └── evaluate_accuracy.py # Report test accuracy
├── data/                     # MNIST CSVs + your collected samples
├── models/                   # Saved weights (ffnn_mnist_weights.npz)
└── notebooks/
    └── ffnn.ipynb            # Training notebook (e.g. Colab)
```

- **`core/`** – Neural network and image preprocessing (used by app and scripts).
- **`app/`** – GUI apps: main quiz and data collection.
- **`scripts/`** – Train the model and evaluate on the test set.
- **`run_app.py`** – Main entry point to run the handwriting app.

## Setup

### 1. Python environment

```bash
cd handwriting-detection-using-NN
pip install -r requirements.txt
```

Uses **Python 3** with `numpy` and `opencv-python`. The GUI uses **tkinter** (included with Python on most installs).

### 2. Trained model (from Google Colab)

The app expects a saved model at `models/ffnn_mnist_weights.npz`. If you trained in Colab with the same `NeuralNetwork` class:

1. In Colab, after training, save the model:
   ```python
   neuralnetwork.save('ffnn_mnist_weights.npz')
   ```
2. Download `ffnn_mnist_weights.npz` from Colab (e.g. from the file browser).
3. Put it in the project’s **`models/`** folder so the path is:
   `handwriting-detection-using-NN/models/ffnn_mnist_weights.npz`.

Your Colab notebook must use the same architecture (e.g. 784 → 16 → 10) and the same `save()` format (e.g. `np.savez` with `w1`, `b1`, `w2`, `b2`, `input_size`, `hidden_size`, `output_size`) so that `NeuralNetwork.load()` in this repo works. The repo’s `core/neural_network.py` matches that.

## How to run

From the project root:

```bash
python run_app.py
```

Or run the app module directly:

```bash
python -m app.handwriting_app
```

A window opens: follow the “Write the number: X” prompt, draw, then click **Predict** to see if the network agrees. Use **New number** for the next digit and **Clear** to erase the canvas.

## For children / target audience

- The app **tells the child which digit to write** (e.g. “Write the number: 5”).
- The child **draws** that digit on the white area.
- **Predict** lets the computer “check” the answer and say **Correct!** or **You wrote X. Try writing 5!**
- **Clear** and **New number** allow another try or a new digit.

Preprocessing (in `core/preprocessing.py`) resizes and inverts the drawing to match MNIST (28×28, white digit on black), so the network gets input similar to its training data for better predictions.

## Train on your own handwriting

If the model often misrecognizes your writing, train it on your own digits:

1. **Collect data** – Run the data collection app. It will ask you to write 0, then 1, … up to 9, with 10 samples per digit (100 total). Draw each digit and click **Save & next**; when you’ve finished all digits, click **Finish & save**. Your samples are saved to `data/my_digits.npz`.

   ```bash
   python -m app.collect_data_app
   ```

2. **Train the model** – From the project root, run the training script. It loads `data/my_digits.npz`, trains the same neural network on your data, and overwrites `models/ffnn_mnist_weights.npz`.

   ```bash
   python scripts/train_on_my_data.py
   ```

3. **Use the app** – Run the main app; it will use the model trained on your handwriting.

   ```bash
   python run_app.py
   ```

You need at least 32 samples to train (e.g. 4 per digit). For better results, use the default 10 samples per digit (100 total) or edit `SAMPLES_PER_DIGIT` in `app/collect_data_app.py` to collect more.

To check accuracy on the MNIST test set: `python scripts/evaluate_accuracy.py` (requires `data/mnist_test.csv`).

## License / use

Use and modify as you like for learning and teaching.
