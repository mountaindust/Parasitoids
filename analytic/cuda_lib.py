import numpy as np

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import skcuda.fft as fft

from pycuda.elementwise import ElementwiseKernel


def fftconv2(A,B):
    '''Computes the fft convolution of two sparse matrices on the GPU
    
    Args:
        A,B: two 2D numpy arrays
        
    Returns:
        2D numpy array with shape A.shape'''
    
    # each array and the output will be padded before fftconv2 returns
    mmid = (np.array(B.shape)/2).astype(int)
    pad_shape = A.shape + mmid
    
    # convolution output should be within an absolute tol of 1e-6
    #   using float32 and complex64?
    
    # if the output array plus the two input fft arrays are too big,
    #   abort mission! (4 bytes in 32-bit float)
    # cutoff at 1.65 GB (normal mem free = 1.76 GB)
    if pad_shape[0]*pad_shape[1]*4*5 > 1650000000: 
        raise MemoryError('Input arrays are too big for GPU convolution.')

    # Take fft of both input arrays on the GPU
    
    plan = fft.Plan(tuple(pad_shape), np.float32, np.complex64)
    AB_gpu = gpuarray.zeros(pad_shape,np.float32)
    # no idea if this will work... I think so?
    AB_gpu[:A.shape[0],:A.shape[1]] = A.toarray()
    A_hat_gpu = gpuarray.empty(pad_shape,np.complex64)
    fft.fft(AB_gpu,A_hat_gpu,plan)
    
    AB_gpu = gpuarray.zeros(pad_shape,np.float32)
    # make sure there is enough memory to continue
    assert cuda.mem_get_info()[0] > pad_shape[0]*pad_shape[1]*4*2
    B_lil = B.tolil()
    AB_gpu[:mmid[0]+1,:mmid[1]+1] = B_lil[mmid[0]:,mmid[1]:].toarray()
    AB_gpu[:mmid[0]+1,-mmid[1]:] = B_lil[mmid[0]:,:mmid[1]].toarray()
    AB_gpu[-mmid[0]:,-mmid[1]:] = B_lil[:mmid[0],:mmid[1]].toarray()
    AB_gpu[-mmid[0]:,:mmid[1]+1] = B_lil[:mmid[0],mmid[1]:].toarray()
    B_hat_gpu = gpuarray.empty(pad_shape,np.complex64)
    fft.fft(AB_gpu,B_hat_gpu,plan)

    AB_gpu.gpudata.free() #release the memory
    
    A_hat_gpu *= B_hat_gpu
    
    B_hat_gpu.gpudata.free()
    C_gpu = gpuarray.zeros(pad_shape,np.float32)
    
    plan = fft.Plan(tuple(pad_shape), np.complex64, np.float32)
    fft.ifft(A_hat_gpu,C_gpu,plan)
    
    return C_gpu.get()[:A.shape[0],:A.shape[1]]


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