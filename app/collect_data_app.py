"""
Data collection app: write digits 0–9 multiple times; each is saved with its label.
Use "Save & next" after each digit, then "Finish & save" when done. Run train_on_my_data.py after.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
import tkinter as tk
from tkinter import font as tkfont

from core.preprocessing import canvas_to_mnist


CANVAS_SIZE = 280
BRUSH_RADIUS = 10
SAMPLES_PER_DIGIT = 10  # how many times to write each digit 0–9
DATA_PATH = PROJECT_ROOT / "data" / "my_digits.npz"


class CollectDataApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Collect your handwriting")
        self.root.resizable(False, False)
        self.root.configure(bg="#f0f4f8")

        self.target_digit = 0
        self.sample_index = 0  # 0 .. SAMPLES_PER_DIGIT-1 for current digit
        self.X_list = []
        self.Y_list = []

        self.drawing_image = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255

        self._build_ui()
        self._bind_drawing()
        self._update_prompt()

    def _build_ui(self):
        instruction_font = tkfont.Font(family="Segoe UI", size=18, weight="bold")
        self.instruction_label = tk.Label(
            self.root,
            text="",
            font=instruction_font,
            bg="#f0f4f8",
            fg="#1a237e",
        )
        self.instruction_label.pack(pady=(16, 8))

        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="white",
            highlightthickness=2,
            highlightbackground="#90a4ae",
        )
        self.canvas.pack(padx=20, pady=8)

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

        save_btn = tk.Button(
            btn_frame,
            text="Save & next",
            font=("Segoe UI", 12),
            command=self._save_and_next,
            bg="#2e7d32",
            fg="white",
            relief="flat",
            padx=16,
            pady=6,
            cursor="hand2",
        )
        save_btn.pack(side=tk.LEFT, padx=6)

        finish_btn = tk.Button(
            btn_frame,
            text="Finish & save",
            font=("Segoe UI", 12),
            command=self._finish_and_save,
            bg="#1a237e",
            fg="white",
            relief="flat",
            padx=16,
            pady=6,
            cursor="hand2",
        )
        finish_btn.pack(side=tk.LEFT, padx=6)

        self.feedback_label = tk.Label(
            self.root,
            text="",
            font=tkfont.Font(family="Segoe UI", size=14),
            bg="#f0f4f8",
            fg="#37474f",
            wraplength=CANVAS_SIZE + 40,
        )
        self.feedback_label.pack(pady=(8, 20))

    def _update_prompt(self):
        if self.target_digit > 9:
            self.instruction_label.config(text="All digits done! Click 'Finish & save'.")
            self.feedback_label.config(
                text=f"Collected {len(self.X_list)} samples so far."
            )
            return
        self.instruction_label.config(
            text=f"Write the number: {self.target_digit}  ({self.sample_index + 1}/{SAMPLES_PER_DIGIT})"
        )
        self.feedback_label.config(text=f"Saved so far: {len(self.X_list)} samples.")

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
        self.feedback_label.config(text=f"Saved so far: {len(self.X_list)} samples.")

    def _save_and_next(self):
        # Preprocess and save one sample
        X_one = canvas_to_mnist(self.drawing_image, (CANVAS_SIZE, CANVAS_SIZE))
        self.X_list.append(X_one)
        self.Y_list.append(self.target_digit)

        self._clear()

        self.sample_index += 1
        if self.sample_index >= SAMPLES_PER_DIGIT:
            self.sample_index = 0
            self.target_digit += 1

        self._update_prompt()

    def _finish_and_save(self):
        if not self.X_list:
            self.feedback_label.config(text="No samples collected. Draw and use 'Save & next' first.")
            return
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        X = np.vstack(self.X_list)
        Y = np.array(self.Y_list, dtype=np.int64)
        np.savez(DATA_PATH, X=X, Y=Y)
        self.feedback_label.config(
            text=f"Saved {len(self.X_list)} samples to {DATA_PATH}. Run: python scripts/train_on_my_data.py",
            fg="#2e7d32",
        )
        self.root.after(2500, self.root.destroy)

    def run(self):
        self.root.mainloop()


def main():
    app = CollectDataApp()
    app.run()


if __name__ == "__main__":
    main()
