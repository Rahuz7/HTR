import cv2
import numpy as np
import argparse
import os
import shutil

class Preprocessing:
    def __init__(self):
        pass

    def resize(self, img):
        width, length = img.shape[:2]
        factor = min(1, float(1024.0 / length))
        size = int(factor * length), int(factor * width)
        img_resized = cv2.resize(img, size)

        cv2.imwrite("./preprocessing_out/img_resized.png", img_resized)
        return(img_resized)


    def denoise(self, img):
        img_denoised = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        cv2.imwrite("./preprocessing_out/img_denoised.png", img_denoised)
        return(img_denoised)

    def binarize(self, img):
        while True:
            try:
                grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                binarized = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 32) #Increase the last argument (32) for better de-noising
                kernel_erode = np.ones((2,2),np.uint8)
                img_binarized = cv2.erode(binarized, kernel_erode, iterations = 1)

                cv2.imwrite("./preprocessing_out/binarized.png", img_binarized)
                return(img_binarized, grayscaled)
                break

            except:
                binarized = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 32) #Increase the last argument (32) for better de-noising
                kernel_erode = np.ones((2,2),np.uint8)
                img_binarized = cv2.erode(binarized, kernel_erode, iterations = 1)

                cv2.imwrite("./preprocessing_out/binarized.png", img_binarized)
                return(img_binarized, img)

    def rotate(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image color to grayscale
        gray = cv2.bitwise_not(gray) # change color between background and object
        thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # binary picture
        coords = np.column_stack(np.where(thresh > 0)) # where return value depending on condition (everything except black)
    # column stack take those value and convert 1D-array in 2D-array
        angle = cv2.minAreaRect(coords)[-1] # calculate angle with coords
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (height, width) = img.shape[:2]
        center = (width // 2,height // 2)
        m = cv2.getRotationMatrix2D(center,angle,1.0)
        (img_rotated) = cv2.warpAffine(img, m, (width, height),flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE) #Also needs research

        cv2.imwrite("./preprocessing_out/rotated.png", img_rotated)
        return(img_rotated)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image file")
    args = vars(ap.parse_args())
    img = cv2.imread(args["image"])

    while True:
        try:
            os.mkdir("./preprocessing_out")
            break
        except:
            shutil.rmtree("./preprocessing_out")

    p = Preprocessing()
    p.resize(img)
    p.denoise(img)
    p.binarize(img)
    p.rotate(img)
