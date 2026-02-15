from __future__ import annotations

from typing import List, Optional, Tuple
import math

import cv2
import numpy as np

Point = Tuple[int, int]


def calcular_centroide(contour: np.ndarray) -> Optional[Point]:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None


def find_mark_centers(gray: np.ndarray, thresh_val: int, min_area: int) -> Tuple[List[Point], np.ndarray]:
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]

    centers: List[Point] = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            center = calcular_centroide(cnt)
            if center:
                centers.append(center)
    return centers, thresh


def pick_farthest_pair(points: List[Point]) -> Optional[Tuple[Point, Point, float]]:
    best = None
    best_d = -1.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = math.dist(points[i], points[j])
            if d > best_d:
                best_d = d
                best = (points[i], points[j], d)
    return best
