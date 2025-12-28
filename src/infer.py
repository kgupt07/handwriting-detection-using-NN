import numpy as np

from neural_network import NeuralNetwork  # or wherever your class lives

def load_model(weights_path):
    data = np.load(weights_path)

    NeuralNetwork_ = NeuralNetwork(
        int(data["input_size"]),
        int(data["hidden_size"]),
        int(data["output_size"])
    )

    NeuralNetwork_.w1 = data["w1"]
    NeuralNetwork_.b1 = data["b1"]
    NeuralNetwork_.w2 = data["w2"]
    NeuralNetwork_.b2 = data["b2"]

    return NeuralNetwork_