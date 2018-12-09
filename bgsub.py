import numpy as np
import cv2


class bgsub:
    def __init__(self, bgsegm):
        self.fgbg = bgsegm

    def get_bgsub_frame(self, img):
        stencil = np.zeros(img.shape).astype(img.dtype)
        imgNoBack = self.fgbg.apply(img)
        imgNoBack = cv2.morphologyEx(
            imgNoBack, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        imgNoBack = cv2.GaussianBlur(imgNoBack,(21,21),0)
        imgNoBack
        _, contours, _ = cv2.findContours(
            imgNoBack, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            cnt = contours[i]
            cv2.fillConvexPoly(stencil, cnt, [255, 255, 255])

        return cv2.bitwise_and(img, stencil)
