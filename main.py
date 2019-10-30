import cv2 as cv
import numpy as np

img = cv.imread('./images/imagem1.jpg', 0)

blurred = cv.GaussianBlur(img, (3, 3), 0)
wide = cv.Canny(blurred, 10, 200)

cv.imwrite('result/img1.jpg', wide)
