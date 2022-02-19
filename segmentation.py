import cv2
import numpy as np
import argparse
import os # creer dossier
import shutil # supprime dossier
from matplotlib import pyplot as plt
from preprocessing import Preprocessing as p
import time

class TextSegmentation:

    def __init__(self, imgName):
        self.original_img = imgName

    #fonction qui enleve les espaces inutiles des images des mots
    def resizeWord(self, img):
        # img -- image des mots
        binary, grayscaled = p.binarize(self, img)
        erosion = cv2.erode(binary, None, iterations = 6)
        bitwise = cv2.bitwise_not(erosion)
        contours, hierarchy = cv2.findContours(bitwise, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, width, height = cv2.boundingRect(c)
            if width > 30 and height > 30:
                crop = img[y : y+height , x : x+width]
                return(crop)

# fonction de lissage d'une courbe #
    def smoothing(self, histogram, p = 20):
        # histogram -- liste de valeur de l'histogramme
        # p -- taille du noyau pour effectuer le lissage
        histogramSmoothing = []
        for i in range(p):
            histogramSmoothing.append(0)
        for i in range(p, len(histogram)-p):
            val = 0
            for j in range(2*p):
                val += histogram[i-p+j]
            histogramSmoothing.append(val/2/p)
        for i in range(p):
            histogramSmoothing.append(0)
        return(histogramSmoothing)

# fonction de création d'un histogramme horizontal #
    def horizontalHistogram(self, img):
        #img -- image principale
        heightMax, widthMax = img.shape[:2]
        histogram = []
        for height in range(heightMax):
            counter = 0
            for width in range(widthMax):
                if (img[height, width] == 0):
                    counter += 1 # compte le nombre de pixel noir par ligne
            histogram.append(counter)
        return(histogram)

# fonction de création d'un histogramme vertical #
    def verticalHistogram(self, img):
        # img -- image principale
        heightMax, widthMax = img.shape[:2]
        histogram = []
        for width in range(widthMax):
            counter = 0
            for height in range(heightMax):
                if (img[height, width] == 0):
                    counter += 1 # compte le nombre de pixel noir par colonne
            histogram.append(counter)
        return(histogram)

# fonction pour récupérer les pics inférieurs d'une courbe #
    def lowerPeak(self, histogram, list, widthMax, seuil) :
        # histogram -- liste des valeurs de l'histogramme
        # list -- liste de la largeur ou de la hauteur de l'image
        # widthMax -- nombre de pixel noir maximum
        # seuil -- limite à franchir pour déterminer les pics inférieurs (empeche la selection des faux pics inférieurs)
        bot = [0]
        progress = 0

        while True :
            # recherche pic supérieur
            upper = True
            min = 0
            while upper == True :
                if histogram[progress] >= min :
                    min = histogram[progress]
                    progress += 1
                else :
                    upper = False

            # recherche pic inférieur
            max = widthMax
            lower = True
            while lower == True :
                if histogram[progress] <= max :
                    max = histogram[progress]
                    progress += 1

                if histogram[progress] > max:
                    lower = False
                    if histogram[progress - 1] < seuil :
                        bot.append(list[progress])
                        progress += 1

                if progress == list[-1] :
                    bot.append(list[-1])
                    return(bot)

# fonction segmentation des mots #
    def wordSegmentation(self, img, counter):
        # img -- image des différentes lignes du texte
        # counter -- compteur pour le nom des images sauvegardées
        imgCopy = img.copy()
        imgBinary, imgGrayscaled = p.binarize(self, img) # binarisation
        heightMax, widthMax = self.original_img.shape[:2]

        # creer liste de la largeur de la ligne
        widthList = np.arange(widthMax)

        # histogramme vertical
        histogram = self.verticalHistogram(imgBinary)

        # lissage de l'histogramme
        histogramSmoothing = self.smoothing(histogram, p=7)

        # coordonnées de l'espace entre chaque mot
        words = self.lowerPeak(histogramSmoothing, widthList, widthMax, seuil=1)

        # parcourir les coordonnées de chaque mot pour les extraires
        for i in range(len(words)-1):
            cropImg = imgGrayscaled[0 : heightMax, words[i] : words[i+1]] # extraction des mots sur l'image en teinte de gris
            heightCropImg, widthCropImg = cropImg.shape[:2]
            if widthCropImg >= 10:
                cropWords = self.resizeWord(cropImg)
                cv2.imwrite('./out/{}.jpg'.format(counter), cropWords)
                counter += 1
        return(counter)

# fonction segmentation des lignes #
    def linesSegmentation(self):
        imgCopy = self.original_img.copy()
        heightMax, widthMax = self.original_img.shape[:2]
        imgBinary, imgGrayscaled = p.binarize(self, self.original_img) # binarisation

        # creer liste de la hauteur de l'image
        heightList = np.arange(heightMax)

        # creer histogramme horizontal
        histogram = self.horizontalHistogram(imgBinary)

        # lissage de l'histogramme
        histogramSmoothing = self.smoothing(histogram)
        moyenneHistogram = np.mean(histogramSmoothing)

        # coordonnées de l'espace entre chaque ligne
        lines = self.lowerPeak(histogramSmoothing, heightList, widthMax, moyenneHistogram)

        # creer dossier "./out"
        while True:
            try:
                os.mkdir("./out")
                break
            except:
                shutil.rmtree("./out")

        # compteur pour le nom des images
        counter = 1

        # parcourir les coordonnées de chaque ligne pour les extraires
        for i in range(len(lines)-1):
            cropImg = imgGrayscaled[lines[i] : lines[i+1], 0 : widthMax] # extraction des lignes sur l'image en teinte de gris
            heightCropImg, widthCropImg = cropImg.shape[:2]
            if heightCropImg >= 40:
                test = p.resize(self, cropImg)
                # cv2.imshow('line segmentation', test)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                counter = self.wordSegmentation(cropImg, counter)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image file")
    args = vars(ap.parse_args())

    start = time.time()
    segmentation = TextSegmentation(args["image"])
    segmentation.linesSegmentation()
    end = time.time()
    print(end - start) # calcul le temps que le programme met à s'éxecuter
