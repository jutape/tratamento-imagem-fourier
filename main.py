import cv2 as cv
import numpy as np

def getEdge(filename):
    img = cv.imread(filename, 0)
    blurred = cv.GaussianBlur(img, (3, 3), 0)
    wide = cv.Canny(blurred, 10, 200)
    return wide

def imageBlur(filename):
    img = cv.imread(filename)
    blurImg = cv.blur(img,(20,20)) 
    return blurImg

def removeNoisePeriodic1(filename):
    img = cv.imread(filename,0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    col, row = img.shape
    col = int(col/2)
    row = int(row/2)
    fshift[14:20, 118:123] = 0
    fshift[35:52, 113:128] = 0
    fshift[90:110, 110:130] = 0
    fshift[row+90:row+110, 110:130] = 0
    fshift[row+155:row+170, 113:128] = 0
    fshift[row+180:row+186, 118:123] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def removeNoisePeriodic2(filename):
    img = cv.imread(filename,0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    col, row = fshift.shape
    col = int(col/2)
    row = int(row/2)
    fshift[0:col-20, row-1:row+1] = 0
    fshift[col+20:col+col, row-1:row+1] = 0
    fshift[col-100:col+120, row-102:row-98] = 0
    fshift[col-100:col+120, row+98:row+102] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def removeNoisePeriodic3(filename):
    img = cv.imread(filename,0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    col, row = fshift.shape
    i = 0
    while i < row:
        if(i != row/2):
            fshift[i][i] = 0
        i+=16

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

if __name__ == '__main__':
    cv.imwrite('./result/img1.jpg', getEdge('./images/imagem1.jpg'))
    cv.imwrite('./result/img2.jpg', imageBlur('./images/imagem2.jpg'))
    cv.imwrite('./result/img3.jpg', removeNoisePeriodic1('./images/noise1.png'))
    cv.imwrite('./result/img4.jpg', removeNoisePeriodic2('./images/noise2.jpg'))
    cv.imwrite('./result/img5.jpg', removeNoisePeriodic3('./images/noise3.jpg'))