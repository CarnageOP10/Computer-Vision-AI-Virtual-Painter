import os

import cv2
import mediapipe as mp
import time

import numpy as np

import HandTrackingModule as htm

################
brushThickness = 15
eraserThickness = 50

folderPath = "header"
myList = os.listdir(folderPath)
print(myList)
overlaylist = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlaylist.append(image)
print(len(overlaylist))


header = overlaylist[0]
drawColor = (255,0,255)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0

imgCanvas = np.zeros((720,1280,3), np.uint8)

while True:
    # 1. import image
    success, img = cap.read()
    img = cv2.flip(img,1)

    # 2. find lm
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, False)

    if len(lmlist)!=0:
        # print(lmlist)
    # tip of index and middle
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

    # 3. which finger is up
        fingers = detector.fingersUp()
        print(fingers)

    # 4. if selectn mode:2 fingers up: select  index
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("selection mode")
            if y1<125:
                if 300<x1<350:
                    header = overlaylist[0]
                    drawColor = (255,0,255)
                elif 400<x1<500:
                    header = overlaylist[1]
                    drawColor = (255,0,0)
                elif 550<x1<620:
                    header = overlaylist[2]
                    drawColor = (0,255,0)
                elif 700<x1<720:
                    header = overlaylist[3]
                    drawColor = (0,0,0)

            cv2.rectangle(img, (x1,y1-25), (x2,y2+25), drawColor, cv2.FILLED)




    # 4. if selectn mode: index finger up: draw
        if fingers[1] and fingers[2] == False:
            xp, yp = 0, 0
            cv2.circle(img, (x1,y1),15, drawColor,cv2.FILLED)
            print("drawing mode")
            if xp == 0 and yp == 0:
                xp,yp = x1,y1

            if drawColor == (0,0,0):
                cv2.line(img,(xp,yp),(x1,y1), drawColor,eraserThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1), drawColor,eraserThickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1), drawColor,brushThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1), drawColor,brushThickness)

            xp,yp = x1,y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # setting header image
    img[0:125, 0:1280] = header
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow("Image", img)
    cv2.imshow("ImageCanvas", imgCanvas)

    cv2.waitKey(1)
