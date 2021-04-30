# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 23:19:55 2021

@author: Amanhandele
"""
import numpy as np
import numba
from numba import cuda
import time

x=np.ones(1000000)
x.shape=1000,1000
@numba.jit
def _smooth(x):

    out = np.empty_like(x)
    for i in range(1, x.shape[0] - 1):
        for j in range(1, x.shape[1] - 1):
            out[i,j] = (x[i-1, j-1] + x[i-1, j+0] + x[i-1, j+1] +
                        x[i+0, j-1] + x[i+0, j+0] + x[i+0, j+1] +
                        x[i+1, j-1] + x[i+1, j+0] + x[i+1, j+1])//9 

    return out
@cuda.jit
def smooth_gpu(x):
    out = np.empty_like(x)
    i, j = cuda.grid(2)
    n, m = x.shape
    if 1 <= i < n - 1 and 1 <= j < m - 1:
        out[i, j] = (x[i-1, j-1] + x[i-1, j] + x[i-1, j+1] +
                     x[i  , j-1] + x[i  , j] + x[i  , j+1] +
                     x[i+1, j-1] + x[i+1, j] + x[i+1, j+1]) // 9
    return out
print('CPU')
start=time.time()
_smooth(x)
print(time.time()-start)
print('GPU')
start=time.time()
smooth_gpu(x)
print(time.time()-start)