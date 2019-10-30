import cv2 as cv
import numpy as np

img = cv.imread('./images/imagem1.jpg', 0)
img2 = cv.imread('./images/imagem2.jpg')
img3 = cv.imread('./images/imagem3.jpg')
img4 = cv.imread('./images/imagem4.jpg')
img5 = cv.imread('./images/imagem5.jpg')

blurred = cv.GaussianBlur(img, (3, 3), 0)
wide = cv.Canny(blurred, 10, 200)

cv.imwrite('result/img1.jpg', wide)

blurImg = cv.blur(img2,(10,10)) 

cv.imwrite('result/img2.jpg', blurImg)

dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)

cv.imwrite('result/img3.jpg', dst)
