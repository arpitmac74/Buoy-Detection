import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import copy
import math
from scipy.stats import multivariate_normal
import random
from mpl_toolkits.mplot3d import Axes3D
import cv2

N = 99     # Number of training samples
K = 3       # Number of groups
D = 3      # Number of dimensions

#x = np.random.randint(0, high = 256, size=(N, D))

#x1 = np.random.randint(20, high = 50, size=(N//3, D))
#x2 = np.random.randint(90, high = 120, size=(N//3, D))
#x3 = np.random.randint(200, high = 230, size=(N//3, D))

#x1 =np.random.normal(20, 10,size=(N//3, D))
#x2 =np.random.normal(60, 10,size=(N//3, D))
#x3 =np.random.normal(90, 10,size=(N//3, D))

x1img = cv2.imread('TrainingImages/Red.png')
x2img = cv2.imread('TrainingImages/Green.png')
x3img = cv2.imread('TrainingImages/Yellow.png')
x4img = cv2.imread('TrainingImages/Water1.png')
x5img = cv2.imread('TrainingImages/Water2.png')
x6img = cv2.imread('TrainingImages/Water3.png')
x7img = cv2.imread('TrainingImages/Water3_1.png')

x1xy = np.nonzero(x1img)
x2xy = np.nonzero(x2img)
x3xy = np.nonzero(x3img)
x4xy = np.nonzero(x4img)
x5xy = np.nonzero(x5img)
x6xy = np.nonzero(x6img)
x7xy = np.nonzero(x7img)

x1 = np.vstack([x1img[x1xy[0],x1xy[1],0] , x1img[x1xy[0],x1xy[1],1] , x1img[x1xy[0],x1xy[1],2]]).T
x2 = np.vstack([x2img[x2xy[0],x2xy[1],0] , x2img[x2xy[0],x2xy[1],1] , x2img[x2xy[0],x2xy[1],2]]).T
x3 = np.vstack([x3img[x3xy[0],x3xy[1],0] , x3img[x3xy[0],x3xy[1],1] , x3img[x3xy[0],x3xy[1],2]]).T
x4 = np.vstack([x4img[x4xy[0],x4xy[1],0] , x4img[x4xy[0],x4xy[1],1] , x4img[x4xy[0],x4xy[1],2]]).T
x5 = np.vstack([x5img[x5xy[0],x5xy[1],0] , x5img[x5xy[0],x5xy[1],1] , x5img[x5xy[0],x5xy[1],2]]).T
x6 = np.vstack([x6img[x6xy[0],x6xy[1],0] , x6img[x6xy[0],x6xy[1],1] , x6img[x6xy[0],x6xy[1],2]]).T
x7 = np.vstack([x7img[x7xy[0],x7xy[1],0] , x7img[x7xy[0],x7xy[1],1] , x6img[x7xy[0],x7xy[1],2]]).T

#x = np.vstack([x1,x2,x3,x4,x5,x6])
x = np.vstack([x7])

N = x.shape[0]

#%%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for c,m,x_ in zip(['r','g','y'],['o','^','*'],[x7,x7,x7]):
    plt.plot(x_[:,0], x_[:,1], x_[:,2], c=c, marker=m,alpha=0.5)
ax.set_xlabel('xAxis')
ax.set_ylabel('yAxis')
ax.set_zlabel('zAxis')
plt.show()

#%%
# initilizing gaussians
#seedMu = np.random.randint(1, high = 70, size = (K,D)).astype(np.float64) 
seedMu = np.asarray([[175,200,200],#[150,100,250],
                     [100,180,140], #[150,250,75]
                     [2500,240,240]]) #,
#                     [100,230,225],
#                     [140,160,225],
#                     [150,180,150]])
seedCov =  [(np.random.randint(100,200)*np.eye(D)) for i in range(K)]
regCov = 1e-6 * np.eye(D)
seedGauss = [multivariate_normal(mu, cov) for mu,cov in zip(seedMu,seedCov)]
seedPi = np.asarray([1/K for i in range(K)])

alpha = np.zeros((N,K))
pdf = np.zeros((N,K))

log_likelihoods = []

#%%

iterations = 300
for itera in range(iterations):  
    #%%
    # E Step
    
    # =============================================================================
    # for gauss,pi,i in zip(seedGauss, seedPi, range(K)):
    # #    dr = pi*gauss.pdf(x)
    #     alpha[:,i] = pi*gauss.pdf(x)
    # #    break
    # alpha = np.divide(alpha.T,
    #                   np.sum(alpha , axis = -1),
    #                   out=np.zeros_like(alpha.T),
    #                   where=np.sum(alpha , axis = -1)!=0).T
    # =============================================================================
    
    stop = False
    
    for i,gauss in enumerate(seedGauss):
       pdf[:,i] = gauss.pdf(x)
    
    pixPdf = np.dot(pdf, np.diag(seedPi))
    pixPdfSum = np.sum(pixPdf , axis = -1)
    
    alpha = np.divide(pixPdf.T,
                      pixPdfSum+1e-8).T 
    
#        alpha = np.divide(pixPdf.T,
#                      pixPdfSum,
#                      out=np.zeros_like(alpha.T),
#                      where=np.sum(alpha , axis = -1)!=0).T
    #%%
    # M step
    print(itera)
    alphaSum = np.sum(alpha.T , axis = -1)
    
    mu = np.divide((np.dot(alpha.T,x)).T, alphaSum+1e-8).T
    
    muTemp = []
    cov = []
    for i in range(K):
    #    muTemp = np.multiply(np.ones((N,D)),seedMu[:,i])
        #muTemp = np.dot(np.ones((N,D)),np.diag(seedMu[i,:]))
        xSubMu = x-seedMu[i, :]
        alphaTemp = np.multiply(alpha[:,i],np.ones((N,D)).T)
        #cov.append(np.dot((alphaTemp*xSubMu.T),xSubMu))
        cov.append(np.divide(np.dot(xSubMu.T, np.multiply(alphaTemp.T, xSubMu)), np.sum(alpha[:, i])+1e-8))
        cov[-1] += regCov
        
        
    
    pi = alphaSum/N
    
#    if (np.linalg.norm(mu - seedMu) < 1e-4):
#        stop = True
    print(seedMu)
    print(seedCov)
    
    seedMu = copy.deepcopy(mu)
    seedCov = copy.deepcopy(cov)
    seedPi = copy.deepcopy(pi)
    seedGauss =  [multivariate_normal(mu_, cov_) for mu_,cov_ in zip(seedMu,seedCov)]
    
    #%%
    #Log Likelihood
#    log_likelihoods.append(np.log(np.sum([k*multivariate_normal(mu[i],cov[j]).pdf(x) for k,i,j in zip(pi,range(len(mu)),range(len(cov)))])))
#    print (log_likelihoods)
    
    #%%
    
#    if(stop):
#        break
