import numpy as np
import time
from numba import cuda, float32
import math
# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.


TPB = 40
@cuda.jit
def fast_matmul(A, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid


    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sA[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

a = [[], [], []]

size=7500
blocksPerGrid_x = math.ceil(2)
blockspergrid_y = math.ceil(2)
bpd = (blocksPerGrid_x, blockspergrid_y)
x0 = np.random.rand(size, size) * 10
x0.shape = size, size
x = x0
z=np.zeros(size*size)
z.shape=size,size
fast_matmul[bpd,TPB](x,z)


print ('Count\tCPU\tGPU\n')
Repeat=4
for size in range(10, 500, 1):
    t1=0.
    t2=0.
    for i in range (Repeat):
        x0=np.random.rand(size,size)*10
        x0.shape=size,size
        x=x0
        #x=np.array([[1,2,3],[3,4,5],[6,7,8]])
        start = time.time()
        x=np.matmul(x,x)
        t1_func = time.time() - start
        #print(x)



        x=x0
        z=np.zeros(size*size)
        z.shape=size,size
        fast_matmul[bpd,TPB](x,z)
        start = time.time()
        z=np.zeros(size*size)
        z.shape=size,size
        fast_matmul[bpd,TPB](x,z)
        t2_func = time.time() - start
        #print(x)
        t1 += t1_func/Repeat
        t2 += t2_func/Repeat
    a[0].append(size)
    a[1].append(t1)
    a[2].append(t2)
    print(str(size)+'\t'+str(t1)+'\t'+str(t2) + '\n')
a=np.array(a)
import matplotlib.pyplot as plt
plt.plot(a[0], a[1], '-b')
plt.plot(a[0], a[2], '-r')
plt.show()