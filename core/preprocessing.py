"""
Preprocess canvas/drawn images to match MNIST format for the neural network.
MNIST: 28x28 grayscale, white digit on black background, pixel values 0-255.
Uses OpenCV for image operations.
"""

import numpy as np
import cv2


# MNIST dimensions
MNIST_SIZE = 28
MNIST_FLAT = 784


def canvas_to_mnist(image: np.ndarray, canvas_size: tuple = (280, 280)) -> np.ndarray:
    """
    Convert a canvas image (e.g. black stroke on white) to MNIST-like format.

    - Resizes to 28x28
    - Converts to grayscale (if needed)
    - Inverts so digit is white on black (like MNIST)
    - Centers the digit
    - Returns shape (1, 784) with values in [0, 255] to match training data

    Args:
        image: NumPy array (H, W) grayscale or (H, W, 3) BGR from OpenCV.
        canvas_size: Original canvas (width, height); used for consistency.

    Returns:
        Array of shape (1, 784), values in [0, 255].
    """
    # Grayscale if needed
    if image.ndim == 3:
        arr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        arr = np.asarray(image, dtype=np.uint8).copy()

    # Invert: MNIST is white digit on black background; canvas is black on white
    arr = cv2.bitwise_not(arr)

    # Resize to 28x28 (same as MNIST) using INTER_AREA for downscaling
    arr = cv2.resize(arr, (MNIST_SIZE, MNIST_SIZE), interpolation=cv2.INTER_AREA)

    # Center the digit in the 28x28 box
    arr = _center_digit(arr)

    # Flatten and scale to 0-255 (MNIST training used raw 0-255)
    flat = arr.astype(np.float64).flatten()
    flat = np.clip(flat, 0, 255)

    # Return (1, 784) for single-sample prediction
    return flat.reshape(1, MNIST_FLAT)


def _center_digit(img: np.ndarray) -> np.ndarray:
    """
    Center the digit in the image by finding the bounding box of non-zero pixels
    and shifting so the digit is centered in the 28x28 frame.
    """
    if img.size == 0:
        return img
    rows = np.any(img > 0, axis=1)
    cols = np.any(img > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    crop = img[rmin : rmax + 1, cmin : cmax + 1]
    out = np.zeros((MNIST_SIZE, MNIST_SIZE), dtype=img.dtype)
    y0 = (MNIST_SIZE - crop.shape[0]) // 2
    x0 = (MNIST_SIZE - crop.shape[1]) // 2
    y1, x1 = y0 + crop.shape[0], x0 + crop.shape[1]
    out[y0:y1, x0:x1] = crop
    return out
