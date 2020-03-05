# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:11:16 2019

@author: balam
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import copy
import math
import random
from mpl_toolkits.mplot3d import Axes3D

K = 3
N = 150
D = 3

seedPi = np.asarray([1/K for i in range(K)])
seedMu = np.asarray([random.randint(0,256) for i in range(K)]).astype(np.float64)
seedCov = [np.multiply(np.eye(D),
                       np.asarray([random.randint(0,256),
                                   random.randint(0,256),
                                   random.randint(0,256)])) for i in range(K)]

x1 = np.random.randint(0, high=256, size=(50,3)).astype(np.float64)
x2 = np.random.randint(0, high=256, size=(50,3)).astype(np.float64)
x3 = np.random.randint(0, high=256, size=(50,3)).astype(np.float64)

X = np.vstack((x1,x2,x3))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for c,m,x in zip(['r','g','y'],['o','^','*'],[x1,x2,x3]):
    ax.scatter(x[:,0], x[:,1], x[:,2], c=c, marker=m,alpha=0.5)
ax.set_xlabel('Red')
ax.set_ylabel('Blue')
ax.set_zlabel('Green')

