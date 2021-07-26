from typing import Any, List, Tuple, Union

import cv2
import numpy as np


def akaze(bimg: bytes, draw: bool = False) -> Tuple[Tuple[List, np.ndarray], Union[None, bytes]]:
    img = cv2.imdecode(np.frombuffer(bimg, np.uint8), cv2.IMREAD_COLOR)

    akaze = cv2.AKAZE_create()

    kp: List
    desc: np.ndarray
    kp, desc = akaze.detectAndCompute(img, None)

    if not draw:
        return (kp, desc), None

    kpimg = cv2.drawKeypoints(img, kp, None)
    buf: np.ndarray = cv2.imencode(".png", kpimg)[1]
    return (kp, desc), buf.tobytes()


def akaze_compe(
    bimg1: bytes, bimg2: bytes, thresh: float = 0.70, draw: bool = False
) -> Tuple[
    Tuple[Tuple[List, np.ndarray], Tuple[List, np.ndarray], List[Any]],
    Union[None, bytes],
]:
    img1 = cv2.imdecode(np.frombuffer(bimg1, np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(bimg2, np.uint8), cv2.IMREAD_COLOR)

    akaze = cv2.AKAZE_create()

    kp1: List
    desc1: np.ndarray
    kp2: List
    desc2: np.ndarray
    kp1, desc1 = akaze.detectAndCompute(img1, None)
    kp2, desc2 = akaze.detectAndCompute(img2, None)

    # TODO Consider using either way.
    k = 2
    if k >= 2:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = [a for a, b in bf.knnMatch(desc1, desc2, k) if a.distance < b.distance * thresh]
    else:  # k<= 1
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)

    if not draw:
        return ((kp1, desc1), (kp2, desc2), matches), None

    matches_img = cv2.drawMatchesKnn(
        img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    buf: np.ndarray = cv2.imencode(".png", matches_img)[1]
    return ((kp1, desc1), (kp2, desc2), matches), buf.tobytes()
