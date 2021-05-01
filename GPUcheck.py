# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 23:19:55 2021

@author: Amanhandele
"""
import numpy as np
import numba
from numba import cuda
import time
import math
cuda.select_device(0)
@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
       an_array[x, y] += 1
x=np.ones(1048576)
x.shape=1024,1024
threadsperblock = (16, 16)
blockspergrid_x = math.ceil(1024 / threadsperblock[0])
blockspergrid_y = math.ceil(1024 / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
increment_a_2D_array[blockspergrid, threadsperblock](x)

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
def smooth_gpu(x,out):

    i, j = cuda.grid(2)
    n, m = x.shape
    if 1 <= i < n - 1 and 1 <= j < m - 1:
        out[i,j]=( (x[i-1, j-1] + x[i-1, j] + x[i-1, j+1] +
                     x[i  , j-1] + x[i  , j] + x[i  , j+1] +
                     x[i+1, j-1] + x[i+1, j] + x[i+1, j+1]) // 9)

print('CPU')
start=time.time()
_smooth(x)
print(time.time()-start)
print('GPU')
start=time.time()
o=np.zeros(1048576)
o.shape=1024,1024
smooth_gpu[blockspergrid, threadsperblock](x,o)
print(time.time()-start)
