# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:10:03 2019

@author: balam
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.stats as stats
import math
import glob

trainingImages = "CroppedImages/"

cap = cv2.VideoCapture('detectbuoy.avi')

i = 0
while(cap.isOpened()):
    _, frame = cap.read()
    
    cv2.imshow("img", frame)
    
    if cv2.waitKey(500) & 0xFF == ord('s'):
        i +=1
        path = trainingImages + "Img_" + str(i) + ".png"
        print(path)
        cv2.imwrite(path,frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

