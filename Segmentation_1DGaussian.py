# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:10:03 2019

@author: balam
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
#import copy
import scipy.stats as stats
import math
import glob
import copy



def getEllipse(mask):
    processed = mask.astype(np.uint8)
    processed = cv2.blur(processed, (5, 5))
    ret, thresh = cv2.threshold(processed, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ellipses = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 250 and cv2.contourArea(cnt) < 4000:
            ellipses.append( cv2.fitEllipse(cnt))
    outEllipse = []
    for ell in ellipses:
        (x,y),(MA,ma),angle = ell
        if abs(MA/ma-1) <0.2:
            outEllipse.append(ell)
    return outEllipse

muRB = [];
varRB = [];

muGB = [];
varGB = [];

muYB = [];
varYB = [];

muW3 = [];
varW3 = [];
    
allRedChannel = []
allBlueChannel = []
allGreenChannel = []

redChannel = np.empty(1)
blueChannel = np.empty(1)
greenChannel = np.empty(1)

x1img = cv2.imread('TrainingImages/Red.png')
x2img = cv2.imread('TrainingImages/Green.png')
x3img = cv2.imread('TrainingImages/Yellow.png')
x4img = cv2.imread('TrainingImages/Water2.png')
roiR = np.nonzero(x1img)
roiG = np.nonzero(x2img)
roiY = np.nonzero(x3img)
roiW3 = np.nonzero(x4img)
for r, img in zip([roiR,roiG,roiY,roiW3],[x1img,x2img,x3img,x4img]):
    redChannel = np.hstack((redChannel,img[r[0],r[1],2]))
    greenChannel = np.hstack((greenChannel,img[r[0],r[1],1]))
    blueChannel = np.hstack((blueChannel,img[r[0],r[1],0]))

    allRedChannel.append(redChannel)
    allBlueChannel.append(blueChannel)
    allGreenChannel.append(greenChannel)

x = np.linspace(0,255, num=100, endpoint=True)

#redBuoy
print("Red Buoy")

# muRB holds blue, green and then red channel info

plt.figure(1)  
muRB.append(np.mean(allBlueChannel[0]))
varRB.append(np.var(allBlueChannel[0]))  

plt.subplot(421)
plt.hist(allBlueChannel[0],256,[0,256],alpha=0.5, color = 'b')
plt.subplot(422)
plt.plot(x, stats.norm.pdf(x, muRB[0], math.sqrt(varRB[0])),'b-')

muRB.append(np.mean(allGreenChannel[0]))
varRB.append(np.var(allGreenChannel[0]))
 
plt.subplot(423)
plt.hist(allGreenChannel[0],256,[0,256],alpha=0.5, color = 'g')
plt.subplot(424)
plt.plot(x, stats.norm.pdf(x, muRB[1], math.sqrt(varRB[1])),'g-')

muRB.append(np.mean(allRedChannel[0]))
varRB.append(np.var(allRedChannel[0]))

plt.subplot(425)
plt.hist(allRedChannel[0],256,[0,256],alpha=0.5, color = 'r')
plt.subplot(426)
plt.plot(x, stats.norm.pdf(x, muRB[2], math.sqrt(varRB[2])),'r-')

print("muRB: ", muRB)
print("varRB: ", varRB)

#greenBuoy
print("Green Buoy")
plt.figure(2)  
muGB.append(np.mean(allBlueChannel[1]))
varGB.append(np.var(allBlueChannel[1]))  

plt.subplot(421)
plt.hist(allBlueChannel[1],256,[0,256],alpha=0.5, color = 'b')
plt.subplot(422)
plt.plot(x, stats.norm.pdf(x, muGB[0], math.sqrt(varGB[0])),'b-')

muGB.append(np.mean(allGreenChannel[1]))
varGB.append(np.var(allGreenChannel[1]))
 
plt.subplot(423)
plt.hist(allGreenChannel[1],256,[0,256],alpha=0.5, color = 'g')
plt.subplot(424)
plt.plot(x, stats.norm.pdf(x, muGB[1], math.sqrt(varGB[1])),'g-')

muGB.append(np.mean(allRedChannel[1]))
varGB.append(np.var(allRedChannel[1]))

plt.subplot(425)
plt.hist(allRedChannel[1],256,[0,256],alpha=0.5, color = 'r')
plt.subplot(426)
plt.plot(x, stats.norm.pdf(x, muGB[2], math.sqrt(varGB[2])),'r-')

print("muGB: ", muGB)
print("varGB: ", varGB)

#yellowBuoy

print("Yellow Buoy")
plt.figure(3)  
muYB.append(np.mean(allBlueChannel[2]))
varYB.append(np.var(allBlueChannel[2]))  

plt.subplot(421)
plt.hist(allBlueChannel[2],256,[0,256],alpha=0.5, color = 'b')
plt.subplot(422)
plt.plot(x, stats.norm.pdf(x, muYB[0], math.sqrt(varYB[0])),'b-')

muYB.append(np.mean(allGreenChannel[2]))
varYB.append(np.var(allGreenChannel[2]))
 
plt.subplot(423)
plt.hist(allGreenChannel[2],256,[0,256],alpha=0.5, color = 'g')
plt.subplot(424)
plt.plot(x, stats.norm.pdf(x, muYB[1], math.sqrt(varYB[1])),'g-')

muYB.append(np.mean(allRedChannel[2]))
varYB.append(np.var(allRedChannel[2]))

plt.subplot(425)
plt.hist(allRedChannel[2],256,[0,256],alpha=0.5, color = 'r')
plt.subplot(426)
plt.plot(x, stats.norm.pdf(x, muYB[2], math.sqrt(varYB[2])),'r-')

muW3.append(np.mean(allBlueChannel[3]))
varW3.append(np.var(allBlueChannel[3]))

muW3.append(np.mean(allGreenChannel[3]))
varW3.append(np.var(allGreenChannel[3]))

muW3.append(np.mean(allRedChannel[3]))
varW3.append(np.var(allRedChannel[3]))

print("muYB: ", muYB)
print("varYB: ", varYB)

# comparing
plt.figure(4)
plt.plot(x, stats.norm.pdf(x, muRB[0], math.sqrt(varRB[0])),'b--')
plt.plot(x, stats.norm.pdf(x, muGB[0], math.sqrt(varGB[0])),'b+')
plt.plot(x, stats.norm.pdf(x, muYB[0], math.sqrt(varYB[0])),'b-')

plt.figure(5)
plt.plot(x, stats.norm.pdf(x, muRB[1], math.sqrt(varRB[1])),'g--')
plt.plot(x, stats.norm.pdf(x, muGB[1], math.sqrt(varGB[1])),'g+')
plt.plot(x, stats.norm.pdf(x, muYB[1], math.sqrt(varYB[1])),'g-')

plt.figure(6)
plt.plot(x, stats.norm.pdf(x, muRB[2], math.sqrt(varRB[2])),'r--')
plt.plot(x, stats.norm.pdf(x, muGB[2], math.sqrt(varGB[2])),'r+')
plt.plot(x, stats.norm.pdf(x, muYB[2], math.sqrt(varYB[2])),'r-')



#%%
out = cv2.VideoWriter('1D_Gaussian.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
def filterHighestPDF(prob,threshold):
    p = prob.reshape((prob.shape[0]*prob.shape[1],prob.shape[2]))
    q = np.multiply(p,p>threshold)
    b = np.multiply(q>0, np.equal(q, np.max(q, axis=-1, keepdims = True)))*255
    c = b.reshape((prob.shape[0],prob.shape[1],prob.shape[2]))
#    cv2.imshow("test",c)
    return c

testFrames = "Images/TestSet/Frames"
for file in glob.glob(f"{testFrames}/*.jpg"):
    frame = cv2.imread(file)
    ## Order of Probabilities - green, red
    # For redBuoy
    ProbRB = np.zeros((frame.shape[0],frame.shape[1],2))
    # Prob of green channel
    ProbRB[:,:,0] = stats.norm.pdf(frame[:,:,1], muRB[1], math.sqrt(varRB[1]))
    # Prob of red channel
    ProbRB[:,:,1] = stats.norm.pdf(frame[:,:,2], muRB[2], math.sqrt(varRB[2]))
    # Assuming probability of color independent
    ProbRB_final = np.zeros((frame.shape[0],frame.shape[1],1))
#    ProbRB_final = np.multiply(ProbRB[:,:,0],ProbRB[:,:,1])
#    ProbRB_final = ProbRB[:,:,0]+ProbRB[:,:,1]

    
    # For greenBuoy
    ProbGB = np.zeros((frame.shape[0],frame.shape[1],3))
    # Prob of green channel
    ProbGB[:,:,0] = stats.norm.pdf(frame[:,:,0], muGB[0], math.sqrt(varGB[0]))
    # Prob of red channel
    ProbGB[:,:,1] = stats.norm.pdf(frame[:,:,1], muGB[1], math.sqrt(varGB[1]))
    ProbGB[:,:,2] = stats.norm.pdf(frame[:,:,2], muGB[2], math.sqrt(varGB[2]))
    # Assuming probability of color independent
    ProbGB_final = np.zeros((frame.shape[0],frame.shape[1],1))
#    ProbGB_final = np.multiply(ProbGB[:,:,0],ProbGB[:,:,1])
#    ProbGB_final = ProbGB[:,:,0]+ProbGB[:,:,1]

    
    # For yellowBuoy
    ProbYB = np.zeros((frame.shape[0],frame.shape[1],2))
    # Prob of green channel
    ProbYB[:,:,0] = stats.norm.pdf(frame[:,:,1], muYB[1], math.sqrt(varYB[1]))
    # Prob of red channel
    ProbYB[:,:,1] = stats.norm.pdf(frame[:,:,2], muYB[2], math.sqrt(varYB[2]))
    # Assuming probability of color independent
    ProbYB_final = np.zeros((frame.shape[0],frame.shape[1],1))
#    ProbYB_final = np.multiply(ProbYB[:,:,0],ProbYB[:,:,1])
#    ProbYB_final = ProbYB[:,:,0]+ProbYB[:,:,1]
    
    # For Water
    ProbW3 = np.zeros((frame.shape[0],frame.shape[1],3))
    # Prob of green channel
    ProbW3[:,:,0] = stats.norm.pdf(frame[:,:,0], muW3[0], math.sqrt(varW3[0]))
    # Prob of red channel
    ProbW3[:,:,1] = stats.norm.pdf(frame[:,:,1], muW3[1], math.sqrt(varW3[1]))
    ProbW3[:,:,2] = stats.norm.pdf(frame[:,:,2], muW3[2], math.sqrt(varW3[2]))
    # Assuming probability of color independent
    ProbW3_final = np.zeros((frame.shape[0],frame.shape[1],1))
#    ProbYB_final = np.multiply(ProbYB[:,:,0],ProbYB[:,:,1])
#    ProbYB_final = ProbYB[:,:,0]+ProbYB[:,:,1]
    
    # best results with Multiply
    ProbRB_final = ProbRB[:,:,0]*ProbRB[:,:,1]
    ProbGB_final = ProbGB[:,:,0]*ProbGB[:,:,1]
    ProbYB_final = ProbYB[:,:,0]*ProbYB[:,:,1]
    ProbW3_final = ProbW3[:,:,0]* ProbW3[:,:,1]

    prob = np.zeros((frame.shape[0],frame.shape[1],4))
    
    prob[:,:,0] = ProbRB_final
    prob[:,:,1] = ProbGB_final
    prob[:,:,2] = ProbYB_final
    prob[:,:,3] = ProbW3_final 
    
    # best results with Multiply
    rgy = filterHighestPDF(prob,1e-5)
    
    
    redBuoySegement = cv2.bitwise_and(frame,frame,mask = rgy[:,:,0].astype(np.int8))
    greenBuoySegment = cv2.bitwise_and(frame,frame,mask = rgy[:,:,1].astype(np.int8))
    yellowBuoySegment = cv2.bitwise_and(frame,frame,mask = rgy[:,:,2].astype(np.int8))
    
#    cv2.imshow("redBw",redBuoySegement)
#    cv2.imshow("greenBw",greenBuoySegment)
#    cv2.imshow("YelowBw",yellowBuoySegment)
    
    ellipseR = getEllipse(rgy[:,:,0].astype(np.uint8))
    imageInputCp = copy.deepcopy(frame)
    for ell in ellipseR:
        cv2.ellipse(imageInputCp,ell,(0,0,255),3)
        
#    ellipseG = getEllipse(rgy[:,:,1].astype(np.uint8))
#    for ell in ellipseG:
#        cv2.ellipse(imageInputCp,ell,(0,255,0),3)
        
    ellipseY = getEllipse(rgy[:,:,2].astype(np.uint8))
    for ell in ellipseY:
        cv2.ellipse(imageInputCp,ell,(0,255,255),3)
        
    cv2.imshow("All",imageInputCp)
#    out.write(imageInputCp)
#    cv2.imshow("segment Red",temp )
#    cv2.imshow("segment Green", getConvexHull(greenBuoySegment))
#    cv2.imshow("segment Yellow", getConvexHull(yellowBuoySegment))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
                
cv2.waitKey(0)
cv2.destroyAllWindows()