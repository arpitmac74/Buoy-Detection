import cv2
import numpy as np
import matplotlib.pyplot as plt
#import copy
import scipy.stats as stats
import math
import glob
from scipy.stats import multivariate_normal
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
        if abs(MA/ma-1) <0.3:
            outEllipse.append(ell)
    return outEllipse

def filterHighestPDF(prob,threshold):
    p = prob.reshape((prob.shape[0]*prob.shape[1],prob.shape[2]))
    q = np.multiply(p,p>threshold)
    b = np.multiply(q>0, np.equal(q, np.max(q, axis=-1, keepdims = True)))*255
    c = b.reshape((prob.shape[0],prob.shape[1],prob.shape[2]))
#    cv2.imshow("test",c)
    return c

testFrames = "Images/TestSet/Frames"
for file in glob.glob(f"{testFrames}/*.jpg"):
    inputImage = cv2.imread(file)
    frame = np.zeros((inputImage.shape[0], inputImage.shape[1],3))
    frame[:,:,0] = inputImage[:,:,0]
    frame[:,:,1] = inputImage[:,:,1]
    frame[:,:,2] = inputImage[:,:,2]
    ## Order of Probabilities - green, red
    
    # For redBuoy1
    mean = np.array([ 81.79013503 , 141.08707121 , 253.25509747])
    cov = np.array([[256.35623751, 240.03100862,  -6.07034588],
       [240.03100862, 345.82123082, -11.25744928],
       [ -6.07034588, -11.25744928,   3.55587709]])
    ProbRB1 = multivariate_normal.pdf(frame, mean, cov)
    
    # For redBuoy2
    mean = np.array( [127.82256059 , 181.07202587 , 227.562899  ])
    cov = np.array([[ 807.28375841,  955.03284211,  -53.89916286],
       [ 955.03284211, 1322.46673621,   70.550918  ],
       [ -53.89916286,   70.550918  ,  483.35601928]])
    ProbRB2 = multivariate_normal.pdf(frame, mean, cov)
    
    # For redBuoy3
    mean = np.array( [118.93331116 , 203.23501071 , 239.5673932 ])
    cov = np.array([[ 483.64111335,  544.66624877, -144.76307453],
       [ 544.66624877,  745.64837677, -192.50690473],
       [-144.76307453, -192.50690473,   65.81344721]])
    ProbRB3 = multivariate_normal.pdf(frame, mean, cov)
    
    ProbRB = ProbRB1 + ProbRB2 + ProbRB3

    # For Green1 
    mean = np.array([112.05003011 , 183.18656764 , 103.53271839])
    cov = np.array([[ 98.18729895, 128.48175019, 111.23031125],
       [128.48175019, 372.47086917, 237.17047113],
       [111.23031125, 237.17047113, 230.78640153]])
    ProbGB1 = multivariate_normal.pdf(frame, mean, cov)

    # For Green2
    mean = np.array([125.22320558 , 229.46544678 , 142.17248589])
    cov = np.array([[ 83.42004155, 109.12603316, 133.04099339],
       [109.12603316, 181.75339967, 209.44426981],
       [133.04099339, 209.44426981, 280.21373779]])
    ProbGB2 = multivariate_normal.pdf(frame, mean, cov)
    
    # For Green3 
    mean = np.array([150.32076907 , 239.42616469 , 187.56685088])
    cov = np.array([[296.42463121, 109.06686387, 351.389052  ],
       [109.06686387, 138.29429843, 172.87515629],
       [351.389052  , 172.87515629, 653.94501523]])
    ProbGB3 = multivariate_normal.pdf(frame, mean, cov)
    
    ProbGB = ProbGB1 + ProbGB2 + ProbGB3

    # For yellowBuoy
    mean =  np.array([ 91.80577429 , 212.23157274 , 222.79794267])
    cov =np.array([[ 284.07052052,  100.38774394, -131.8461949 ],
       [ 100.38774394,  153.70950183,  110.31242709],
       [-131.8461949 ,  110.31242709,  354.51441495]])
    ProbYB = multivariate_normal.pdf(frame, mean, cov)
    
    
    # For Water1
    mean =  np.array([142.28733829 , 235.29220448 , 230.60909523])
    cov =np. array([[648.07214716,  -8.60508211, -32.46272246],
       [ -8.60508211,  41.25878113,  -9.85558692],
       [-32.46272246,  -9.85558692,  13.38085394]])
    ProbW1 = multivariate_normal.pdf(frame, mean, cov)
    
    # For Water2
    mean =  np.array([143.75613352 , 202.77981091 , 214.89774888])
    cov =np.array([[188.63396806, 266.05922201, 166.15702846],
       [266.05922201, 618.65997712, 483.76021496],
       [166.15702846, 483.76021496, 553.23286015]])
    ProbW2 = multivariate_normal.pdf(frame, mean, cov)
    
    # For Water3
    mean =  np.array([149.97339502 , 221.07885129 , 229.05863034])
    cov =np.array([[ 93.96794414, 101.92016779,  41.46286958],
       [101.92016779, 153.63250224,  53.94860116],
       [ 41.46286958,  53.94860116,  76.79155759]])
    ProbW3 = multivariate_normal.pdf(frame, mean, cov)
    
# =============================================================================
#     # best results with Add
#     ProbRB_final = ProbRB[:,:,0]+ProbRB[:,:,1]
#     ProbGB_final = ProbGB[:,:,0]+ProbGB[:,:,1]
#     ProbYB_final = ProbYB[:,:,0]+ProbYB[:,:,1]
# =============================================================================
    
    prob = np.zeros((frame.shape[0], frame.shape[1],4))
    
    prob[:,:,0] = ProbRB
    prob[:,:,1] = ProbGB
    prob[:,:,2] = ProbYB
#    prob[:,:,3] = ProbW1
#    prob[:,:,4] = ProbW2
    prob[:,:,3] = ProbW3*0.99
    
    
    # best results with Multiply
    rgy = filterHighestPDF(prob,1e-15) #-15
    

# =============================================================================
#     # best results with Add
#     rgy = filterHighestPDF(prob,0.01)
# =============================================================================
    
    
        
    redBuoySegement = cv2.bitwise_and(inputImage,inputImage,mask = rgy[:,:,0].astype(np.int8))
    greenBuoySegment = cv2.bitwise_and(inputImage,inputImage,mask = rgy[:,:,1].astype(np.int8))
    yellowBuoySegment = cv2.bitwise_and(inputImage,inputImage,mask = rgy[:,:,2].astype(np.int8))
#    W1Segment = cv2.bitwise_and(inputImage,inputImage,mask = rgy[:,:,3].astype(np.int8))
#    W2Segment = cv2.bitwise_and(inputImage,inputImage,mask = rgy[:,:,4].astype(np.int8))
    W3Segment = cv2.bitwise_and(inputImage,inputImage,mask = rgy[:,:,3].astype(np.int8))
#    cv2.imshow("redBw",redBuoySegement)
#    cv2.imshow("greenBw",greenBuoySegment)
#    cv2.imshow("YelowBw",yellowBuoySegment)
#    cv2.imshow("W1",W1Segment)
#    cv2.imshow("W2",W2Segment)
    
#    cv2.imshow("W3",W3Segment)
    
#    waterMask = cv2.add(rgy[:,:,5].astype(np.int8),
#                        rgy[:,:,3].astype(np.int8))
    waterMask = rgy[:,:,3].astype(np.int8)
#    waterMask = cv2.add(waterMask,rgy[:,:,4].astype(np.int8))
    waterMask = cv2.bitwise_not(waterMask)
    waterRemoved = cv2.bitwise_and(inputImage,inputImage,mask = waterMask)
    
#    cv2.imshow("WaterRemove",waterRemoved)
    
    # For redBuoy1
    mean = np.array([127.82256002 , 181.07202472 , 227.56289837])
    cov = np.array([[ 807.28375488,  955.03283225,  -53.899162  ],
       [ 955.03283225, 1322.46671736,   70.55092402],
       [ -53.899162  ,   70.55092402,  483.35604698]])
    ProbRB1 = multivariate_normal.pdf(waterRemoved, mean, cov)
    
    # For redBuoy2
    mean = np.array( [118.93331162 , 203.23501096 , 239.56739312])
    cov = np.array([[ 483.64112128,  544.66625081, -144.76307495],
       [ 544.66625081,  745.64837658, -192.5069043 ],
       [-144.76307495, -192.5069043 ,   65.81344725]])
    ProbRB2 = multivariate_normal.pdf(waterRemoved, mean, cov)
    
    # For redBuoy3
    mean = np.array([ 81.79013503 , 141.08707128 , 253.25509746])
    cov = np.array([[256.35623706, 240.03100881,  -6.07034587],
       [240.03100881, 345.82123242, -11.25744938],
       [ -6.07034587, -11.25744938,   3.5558771 ]])
    ProbRB3 = multivariate_normal.pdf(waterRemoved, mean, cov)
    
    PiRb = np.array([0.15828874, 0.38113269, 0.44139788])
    ProbRB = PiRb[0]*ProbRB1 + PiRb[1]*ProbRB2 + PiRb[2]*ProbRB3

    # For Green1 
    mean = np.array([110.15586103 , 177.988079  ,  97.8360865 ])
    cov = np.array([[ 82.84302567, 106.35540435,  74.22384909],
       [106.35540435, 306.33086617, 154.3897207 ],
       [ 74.22384909, 154.3897207 , 118.64202382]])
    ProbGB1 = multivariate_normal.pdf(waterRemoved, mean, cov)

    # For Green2
    mean = np.array([124.00448114 , 217.39861905 , 136.44552769])
    cov = np.array([[135.27527716, 132.43005772, 186.54968698],
       [132.43005772, 361.10595221, 281.7120668 ],
       [186.54968698, 281.7120668 , 375.55342302]])
    ProbGB2 = multivariate_normal.pdf(waterRemoved, mean, cov)
    
    # For Green3 
    mean = np.array( [152.97075593 , 244.63284543 , 194.2491698 ])
    cov = np.array([[269.37418864,  37.51788466, 286.85356749],
       [ 37.51788466,  38.57928137,  14.06820397],
       [286.85356749,  14.06820397, 491.56890665]])
    ProbGB3 = multivariate_normal.pdf(waterRemoved, mean, cov)
    
    PiGb = np.array([0.39978126, 0.38033716, 0.19886462])
    
    ProbGB = PiGb[0]*ProbGB1 + PiGb[1]*ProbGB2 + PiGb[2]*ProbGB3

    # For yellowBuoy1
    mean =  np.array([126.26019818 , 220.53188657 , 213.64088776])
    cov =np.array([[1309.02563182,  591.15305563,  206.03327097],
       [ 591.15305563,  456.16428276,  381.7748615 ],
       [ 206.03327097,  381.7748615 ,  656.20874119]])
    ProbYB1 = multivariate_normal.pdf(waterRemoved, mean, cov)
    
    # For yellowBuoy
    mean =  np.array([ 90.76317111 , 227.60531524 , 235.55071628])
    cov = np.array([[151.98932227,  97.19298225, -25.9326655 ],
       [ 97.19298225, 129.34471901, -25.94483159],
       [-25.9326655 , -25.94483159,  14.05785197]])
    ProbYB2 = multivariate_normal.pdf(waterRemoved, mean, cov)
    
    # For yellowBuoy
    mean =  np.array([142.1653417 , 239.86748964 , 228.92643902])
    cov =np.array([[743.10072833, -40.75966386, -33.70196675],
       [-40.75966386,   4.84762787,   1.89397787],
       [-33.70196675,   1.89397787,   7.39462613]])
    ProbYB3 = multivariate_normal.pdf(waterRemoved, mean, cov)
    
    PiYb = np.array([0.26255614, 0.2175131 , 0.50246477])
    
    ProbYB = PiYb[0]*ProbYB1 + PiYb[1]*ProbYB2 + PiYb[2]*ProbYB3
    
    prob = np.zeros((frame.shape[0], frame.shape[1],3))
    
    prob[:,:,0] = ProbRB
    prob[:,:,1] = ProbGB
    prob[:,:,2] = ProbYB
    
    rgy2 = filterHighestPDF(prob,1e-6) #-20
    
    redBuoySegement = cv2.bitwise_and(inputImage,inputImage,mask = rgy2[:,:,0].astype(np.int8))
    greenBuoySegment = cv2.bitwise_and(inputImage,inputImage,mask = rgy2[:,:,1].astype(np.int8))
    yellowBuoySegment = cv2.bitwise_and(inputImage,inputImage,mask = rgy2[:,:,2].astype(np.int8))

#    cv2.imshow("redBw",redBuoySegement)
#    cv2.imshow("greenBw",greenBuoySegment)
#    cv2.imshow("YelowBw",yellowBuoySegment)
    
    ellipseR = getEllipse(rgy2[:,:,0].astype(np.uint8))
    imageInputCp = copy.deepcopy(inputImage)
    for ell in ellipseR:
        cv2.ellipse(imageInputCp,ell,(0,0,255),5)
        
    ellipseG = getEllipse(rgy2[:,:,1].astype(np.uint8))
    for ell in ellipseG:
        cv2.ellipse(imageInputCp,ell,(0,255,0),5)
        
    ellipseY = getEllipse(rgy2[:,:,2].astype(np.uint8))
    for ell in ellipseY:
        cv2.ellipse(imageInputCp,ell,(0,255,255),5)
        
    cv2.imshow("All",imageInputCp)

#    cv2.imshow("segment Red",temp )
#    cv2.imshow("segment Green", getConvexHull(greenBuoySegment))
#    cv2.imshow("segment Yellow", getConvexHull(yellowBuoySegment))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
    if cv2.waitKey(10) & 0xFF == ord('s'):
        cv2.imwrite('waterMicham.png',waterRemoved)
                
cv2.waitKey(0)
cv2.destroyAllWindows()