import cv2
import numpy as np

def apply_blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

image = cv2.imread('perro.jpg')
blurred_image = apply_blur(image)
cv2.imwrite('blurred_image.jpg', blurred_image)