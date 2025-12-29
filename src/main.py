import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np
from neural_network import NeuralNetwork

# Load trained model
nn = NeuralNetwork.load("models/ffnn_mnist_weights.npz")

root = tk.Tk()
root.title("Digit Learning App")

canvas = tk.Canvas(root, width=280, height=280, bg="white")
canvas.pack()

label = tk.Label(root, text="Draw a digit (0–9)")
label.pack()

def draw(event):
    r = 8
    canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill="black")

canvas.bind("<B1-Motion>", draw)

def predict():
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    img = ImageGrab.grab().crop((x, y, x1, y1)).convert("L")
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.flatten()

    pred = nn.predict(img)
    label.config(text=f"Predicted: {pred}")

def clear():
    canvas.delete("all")
    label.config(text="Draw a digit (0–9)")

tk.Button(root, text="Predict", command=predict).pack()
tk.Button(root, text="Clear", command=clear).pack()

root.mainloop()
