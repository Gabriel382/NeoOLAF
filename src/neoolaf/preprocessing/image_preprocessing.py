from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def binarize_image(image: Image.Image) -> Image.Image:
    """Enhance contrast and binarize page image for cleaner OCR input."""
    img_np = np.array(image)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    gray = clahe.apply(gray)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(binary)


def remove_noise_artifacts(image: Image.Image) -> Image.Image:
    """Remove large blobs, tiny dots, and margin artifacts using connected components."""
    binary = np.array(image)

    inverted = cv2.bitwise_not(binary)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )

    h_img, w_img = binary.shape
    result = inverted.copy()

    for j in range(1, num_labels):
        area = stats[j, cv2.CC_STAT_AREA]
        x = stats[j, cv2.CC_STAT_LEFT]
        w, h = stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]
        aspect = w / (h + 1e-5)

        if area > 300000 and 0.5 < aspect < 2.0:
            result[labels == j] = 0
            continue

        if area < 50:
            result[labels == j] = 0
            continue

        in_left_margin = x < w_img * 0.05
        in_right_margin = (x + w) > w_img * 0.95
        if (in_left_margin or in_right_margin) and area < 5000 and w < 80:
            result[labels == j] = 0

    return Image.fromarray(cv2.bitwise_not(result))


def preprocess_page(page: Image.Image) -> Image.Image:
    """
    Run binarization and noise removal on a single page image.

    Args:
        page: PIL Image of a scanned page.

    Returns:
        Cleaned PIL Image ready for OCR.
    """
    page = binarize_image(page)
    page = remove_noise_artifacts(page)
    return page
