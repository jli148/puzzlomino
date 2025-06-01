import cv2
from cv2.typing import MatLike
import numpy as np

# default values determined experimentally
DEFAULT_BLUR = 32
DEFAULT_THRESH = 48


def preprocess(
    img: MatLike,
    blur: int = DEFAULT_BLUR,
    thresh: int = DEFAULT_THRESH,
) -> MatLike:
    blurred = cv2.blur(img, (blur, blur))
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    _, binarized = cv2.threshold(equalized, thresh, 255, cv2.THRESH_BINARY_INV)

    return binarized


class PuzzleContour:
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    LINE_THICKNESS = 10

    def __init__(self, contour: MatLike):
        self.contour = contour
        self.bounding_hull = cv2.convexHull(contour)

    def area(self) -> float:
        contour_area = cv2.contourArea(self.contour)
        bounding_area = cv2.contourArea(self.bounding_hull)

        return contour_area / bounding_area

    def overlay_on(self, img: MatLike) -> MatLike:
        canvas = np.copy(img)
        cv2.drawContours(canvas, [self.contour], 0, self.GREEN, self.LINE_THICKNESS)
        cv2.drawContours(canvas, [self.bounding_hull], 0, self.RED, self.LINE_THICKNESS)

        return canvas


def get_puzzle_contour(processed_img: MatLike) -> PuzzleContour:
    contours, _ = cv2.findContours(
        processed_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    biggest_contour = max(contours, key=cv2.contourArea)

    return PuzzleContour(biggest_contour)
