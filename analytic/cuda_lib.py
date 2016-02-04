import numpy as np
from scipy import sparse

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import skcuda.fft as fft

# from pycuda.elementwise import ElementwiseKernel

class CudaSolve():
    
    def __init__(self,A,max_shape):
        '''Initialize CUDA solver with fft of solution after first day.
        
        Args:
            A: First day's spread, coo sparse matrix
            max_shape: Shape of transformed solution'''
        
        # determine the shape of the padded solution array
        mmid = (max_shape/2).astype(int)
        self.pad_shape = A.shape + mmid
        
        # CUDA plans
        self.fft_plan = fft.Plan(tuple(max_shape),np.float32,np.complex64)
        self.ifft_plan = fft.Plan(tuple(self.pad_shape),np.complex64,np.float32)
        
        # memory check!
        assert cuda.mem_get_info()[0] > self.pad_shape[0]*self.pad_shape[1]*(
            np.dtype(np.float32).itemsize + np.dtype(np.complex64).itemsize)
        
        # allocate temporary space on the gpu and send A there
        A_gpu = gpuarray.zeros(self.pad_shape,np.float32)
        A_gpu[:A.shape[0],:A.shape[1]] = A.toarray()
        
        # allocate persisting space on the gpu for current fft solution
        self.sol_hat_gpu = gpuarray.empty(self.pad_shape,np.complex64)
        
        # find fft
        fft.fft(A_gpu,self.sol_hat_gpu,self.fft_plan)
        
    def fftconv2(self,B):
        '''Update current fourier solution with filter B.
        
        The work to be done in here is padding B appropriately, shifting
        B so that the center is at B[0,0] with wrap-around.
        
        Args:
            B: 2D array'''
            
        # Get array shape information
        mmid = (np.arraay(B.shape)/2).astype(int)
        
        # memory check!
        assert cuda.mem_get_info()[0] > self.pad_shape[0]*self.pad_shape[1]*(
            np.dtype(np.float32).itemsize + np.dtype(np.complex64).itemsize)
        
        # allocate temporary space on the gpu and arrange B there appropriately
        B_gpu = gpuarray.zeros(self.pad_shape,np.float32)
        B_hat_gpu = gpuarray.empty(self.pad_shape,np.complex64)
        
        B_gpu[:mmid[0]+1,:mmid[1]+1] = B[mmid[0]:,mmid[1]:]
        B_gpu[:mmid[0]+1,-mmid[1]:] = B[mmid[0]:,:mmid[1]]
        B_gpu[-mmid[0]:,-mmid[1]:] = B[:mmid[0],:mmid[1]]
        B_gpu[-mmid[0]:,:mmid[1]+1] = B[:mmid[0],mmid[1]:]
        
        # fft and solution update
        fft.fft(B_gpu,B_hat_gpu,self.fft_plan)
        self.sol_hat_gpu *= B_hat_gpu
        
    def get_cursol(self,dom_len,negval=1e-6):
        '''Return the current solution (requires ifft) with small values removed
        
        Args:
            dom_len: domain length (number of cells) of returned solution
            
        Returns:
            coo matrix, current solution with shape (dom_len,dom_len)'''
        
        # memory check!
        assert cuda.mem_get_info()[0] > self.pad_shape[0]*self.pad_shape[1]*(
            np.dtype(np.float32).itemsize)
        
        # Assign temporary space for ifft
        cursol_gpu = gpuarray.empty(self.pad_shape,np.float32)
        
        fft.ifft(self.sol_hat_gpu,cursol_gpu,self.ifft_plan)
        
        # remove values less than negval
        cursol_gpu = gpuarray.if_positive(cursol_gpu-negval,cursol_gpu,0)
        
        # pull down current solution and return coo_matrix
        return sparse.coo_matrix(cursol_gpu[:dom_len,:dom_len].get())
        

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


# def outer(A,B):
    # '''Computes the outer product of two vectors on the GPU
    
    # Args:
        # A,B: two 1D numpy arrays
        
    # Returns:
        # 2D numpy array with shape A.size, B.size'''
        
    # # the outer product between two vectors
    # outer_prod = ElementwiseKernel(
        # "float *out, float *v1, float *v2, int v2_size",
        # "out[i] = v1[i/v2_size]*v2[i%v2_size]", 
        # "outer_prod")
        

    # # GPUs typically are 32-bit
    # A_gpu = gpuarray.to_gpu(A.astype(np.float32))
    # B_gpu = gpuarray.to_gpu(B.astype(np.float32))

    # C_gpu = gpuarray.empty(A.size*B.size, np.float32)

    # outer_prod(C_gpu, A_gpu, B_gpu, B.size)
    
    # return C_gpu.get().reshape(A.size,B.size)