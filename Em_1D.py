#For plotting
import matplotlib.pyplot as plt
#for matrix math
import numpy as np
#for normalization + probability density function computation
#from matplotlib import style
#style.use('fivethirtyeight')
from scipy.stats import norm
import copy
import math

y1 = np.random.normal(loc=-10, scale=3, size=100)

y2 = np.random.normal(loc=0, scale=2, size=100)

y3 = np.random.normal(loc=12, scale=4, size=100)

xAll = np.stack((y1,y2,y3)).flatten() # Combine the clusters to get the random datapoints from above
plt.figure(1)
#base = np.zeros_like(xAll)
#plt.plot(xAll,base,"r+")
plt.subplot(311)
base = np.zeros_like(y1)
plt.plot(y1,base,"r+")
plt.subplot(312)
plt.plot(y2,base,"b+")
plt.subplot(313)
plt.plot(y3,base,"g+")

plt.figure(7)
base = np.zeros_like(y1)
plt.plot(y1,base,"r+")
plt.plot(y2,base,"b+")
plt.plot(y3,base,"g+")

plt.figure(2)
x = np.linspace(-25,25, num=300, endpoint=True)
base = np.zeros_like(y1)
plt.plot(y1,base,"r+")
plt.plot(x,norm.pdf(x, -10, 3),"r-",linewidth=1)
base = np.zeros_like(y2)
plt.plot(y2,base,"b+")
plt.plot(x,norm.pdf(x, 0, 2),"b-",linewidth=1)
base = np.zeros_like(y3)
plt.plot(y3,base,"g+")
plt.plot(x,norm.pdf(x, 12, 4),"g-",linewidth=1)

xAll.sort()
#%%

# initialization
m = np.array([1/3,1/3,1/3])
pi = m / np.sum(m)

mu = np.array([-5,1,2]).astype(np.float64)
sd = np.array([5,3,1]).astype(np.float64)

plt.figure(4)
base = np.zeros_like(y1)
plt.plot(y1,base,"r+")
plt.plot(x,norm.pdf(x, mu[0], sd[0]),"y-",linewidth=1)
base = np.zeros_like(y2)
plt.plot(y2,base,"r+")
plt.plot(x,norm.pdf(x, mu[1], sd[1]),"y-",linewidth=1)
base = np.zeros_like(y3)
plt.plot(y3,base,"r+")
plt.plot(x,norm.pdf(x, mu[2], sd[2]),"y-",linewidth=1)

alpha = np.zeros((xAll.shape[0],3))
iterations = 1000
prevMu =0
for j in range(iterations):
    print("Convergence: ",j,",",math.sqrt(np.dot((mu - prevMu),(mu - prevMu))))
    if math.sqrt(np.dot((mu - prevMu),(mu - prevMu)))< 0.0000001:
        break
    
    prevMu = copy.deepcopy(mu)
    ## E- step
    for k,mu_,sd_,pi_ in zip(range(3),mu,sd,pi):
        alpha[:,k] = pi_*norm.pdf(xAll,mu_,sd_)
#        print("alphaSum", alpha[:,k].sum())
    alpha = np.divide(alpha.T, np.sum(alpha , axis = -1)).T

    ## M- step
    
    tempMu = np.zeros_like(mu)
    
    for i in range(3):
        den2 = alpha[:,i].sum()
        mu[i] = (alpha[:,i]*xAll).sum() / den2
        Num = (alpha[:,i]*(xAll-mu[i])**2).sum()
        
        sd[i] = math.sqrt(Num / den2)
        
        pi[i] = den2/ xAll.shape[0]  


plt.figure(5)
base = np.zeros_like(y1)
plt.plot(y1,base,"r+")
plt.plot(x,norm.pdf(x, mu[0], sd[0]),"y-",linewidth=1)
base = np.zeros_like(y2)
plt.plot(y2,base,"r+")
plt.plot(x,norm.pdf(x, mu[1], sd[1]),"y-",linewidth=1)
base = np.zeros_like(y3)
plt.plot(y3,base,"r+")
plt.plot(x,norm.pdf(x, mu[2], sd[2]),"y-",linewidth=1)
