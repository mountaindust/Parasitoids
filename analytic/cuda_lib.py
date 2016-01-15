import numpy as np

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.elementwise import ElementwiseKernel

def outer(A,B):
    '''Computes the outer product of two vectors on the GPU
    
    Args:
        A,B: two 1D numpy arrays
        
    Returns:
        2D numpy array with shape A.size, B.size'''
        
    # the outer product between two vectors
    outer_prod = ElementwiseKernel(
        "float *out, float *v1, float *v2, int v2_size",
        "out[i] = v1[i/v2_size]*v2[i%v2_size]", 
        "outer_prod")
        

    # GPUs typically are 32-bit
    A_gpu = gpuarray.to_gpu(A.astype(np.float32))
    B_gpu = gpuarray.to_gpu(B.astype(np.float32))

    C_gpu = gpuarray.empty(A.size*B.size, np.float32)

    outer_prod(C_gpu, A_gpu, B_gpu, B.size)
    
    return C_gpu.get().reshape(A.size,B.size)