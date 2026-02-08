"""
Handwriting digit recognition app for children.
Draw a digit, click Predict, and get feedback. Quiz mode: "Write the number X" then check.
Uses OpenCV for the drawing buffer and preprocessing (no PIL/Pillow).
"""

import sys
import random
from pathlib import Path

# Add project root so we can import core and load models
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
import tkinter as tk
from tkinter import font as tkfont

from core.neural_network import NeuralNetwork
from core.preprocessing import canvas_to_mnist


# Canvas size (will be resized to 28x28 for the model)
CANVAS_SIZE = 280
BRUSH_RADIUS = 10
MODEL_PATH = PROJECT_ROOT / "models" / "ffnn_mnist_weights.npz"


class HandwritingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Write a number!")
        self.root.resizable(False, False)
        self.root.configure(bg="#f0f4f8")

        # Load neural network
        if not MODEL_PATH.exists():
            self.root.destroy()
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Train in Colab, save weights with nn.save('ffnn_mnist_weights.npz'), "
                "download and place in the 'models' folder."
            )
        self.model = NeuralNetwork.load(str(MODEL_PATH))

        # Quiz: which digit should the user write?
        self.target_digit = random.randint(0, 9)
        # OpenCV/numpy drawing buffer (black on white): (H, W) grayscale, 255 = white
        self.drawing_image = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255

        self._build_ui()
        self._bind_drawing()

    def _build_ui(self):
        # Instruction (for children)
        instruction_font = tkfont.Font(family="Segoe UI", size=18, weight="bold")
        self.instruction_label = tk.Label(
            self.root,
            text=f"Write the number: {self.target_digit}",
            font=instruction_font,
            bg="#f0f4f8",
            fg="#1a237e",
        )
        self.instruction_label.pack(pady=(16, 8))

        # Canvas (white background, black stroke)
        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="white",
            highlightthickness=2,
            highlightbackground="#90a4ae",
        )
        self.canvas.pack(padx=20, pady=8)

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#f0f4f8")
        btn_frame.pack(pady=8)

        clear_btn = tk.Button(
            btn_frame,
            text="Clear",
            font=("Segoe UI", 12),
            command=self._clear,
            bg="#eceff1",
            fg="#37474f",
            relief="flat",
            padx=16,
            pady=6,
            cursor="hand2",
        )
        clear_btn.pack(side=tk.LEFT, padx=6)

        predict_btn = tk.Button(
            btn_frame,
            text="Predict",
            font=("Segoe UI", 12),
            command=self._predict,
            bg="#1a237e",
            fg="white",
            relief="flat",
            padx=16,
            pady=6,
            cursor="hand2",
        )
        predict_btn.pack(side=tk.LEFT, padx=6)

        # New number (next question in quiz mode)
        next_btn = tk.Button(
            btn_frame,
            text="New number",
            font=("Segoe UI", 12),
            command=self._new_number,
            bg="#7e57c2",
            fg="white",
            relief="flat",
            padx=16,
            pady=6,
            cursor="hand2",
        )
        next_btn.pack(side=tk.LEFT, padx=6)

        # Feedback label
        self.feedback_font = tkfont.Font(family="Segoe UI", size=16, weight="bold")
        self.feedback_label = tk.Label(
            self.root,
            text="",
            font=self.feedback_font,
            bg="#f0f4f8",
            fg="#2e7d32",
            wraplength=CANVAS_SIZE + 40,
        )
        self.feedback_label.pack(pady=(8, 20))

    def _bind_drawing(self):
        self.last_x = self.last_y = None

        def on_press(e):
            self.last_x, self.last_y = e.x, e.y
            self._draw_dot(e.x, e.y)

        def on_drag(e):
            if self.last_x is not None:
                self._draw_line(self.last_x, self.last_y, e.x, e.y)
                self.last_x, self.last_y = e.x, e.y

        def on_release(e):
            self.last_x = self.last_y = None

        self.canvas.bind("<Button-1>", on_press)
        self.canvas.bind("<B1-Motion>", on_drag)
        self.canvas.bind("<ButtonRelease-1>", on_release)

    def _draw_dot(self, x, y):
        r = BRUSH_RADIUS
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        # OpenCV: (x, y) is column, row; draw black (0) on buffer
        cv2.circle(self.drawing_image, (x, y), r, 0, -1)

    def _draw_line(self, x0, y0, x1, y1):
        r = BRUSH_RADIUS
        self.canvas.create_line(x0, y0, x1, y1, fill="black", width=r * 2, capstyle=tk.ROUND)
        cv2.line(
            self.drawing_image,
            (int(x0), int(y0)),
            (int(x1), int(y1)),
            0,
            thickness=r * 2,
            lineType=cv2.LINE_AA,
        )

    def _clear(self):
        self.canvas.delete("all")
        self.drawing_image = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255
        self.feedback_label.config(text="")

    def _new_number(self):
        self.target_digit = random.randint(0, 9)
        self.instruction_label.config(text=f"Write the number: {self.target_digit}")
        self._clear()

    def _predict(self):
        # Preprocess: canvas image -> MNIST-like (1, 784) using OpenCV
        X = canvas_to_mnist(self.drawing_image, (CANVAS_SIZE, CANVAS_SIZE))
        predicted = self.model.predict(X)

        # Quiz feedback: did they write the requested digit?
        if predicted == self.target_digit:
            self.feedback_label.config(text="Correct! Well done!", fg="#2e7d32")
        else:
            self.feedback_label.config(
                text=f"You wrote {predicted}. Try writing {self.target_digit}!",
                fg="#c62828",
            )

    def run(self):
        self.root.mainloop()


def main():
    app = HandwritingApp()
    app.run()


if __name__ == "__main__":
    main()
