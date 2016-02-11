import numpy as np
from scipy import sparse

import reikna.cluda as cluda
import reikna.fft as fft

api = cluda.cuda_api()
thr = api.Thread.create()

class CudaSolve():
    
    def __init__(self,A,max_shape):
        '''Initialize CUDA solver with fft of solution after first day.
        
        Args:
            A: First day's spread, coo sparse matrix
            max_shape: Shape of transformed solution'''
        
        # determine the shape of the padded solution array
        mmid = max_shape//2
        pads = A.shape + mmid
        self.pad_shape = (int(pads[0]),int(pads[1]))
        
        # memory check!
        assert api.cuda.mem_get_info()[0] > self.pad_shape[0]*self.pad_shape[1]*(
            np.dtype(np.float32).itemsize + np.dtype(np.complex64).itemsize)
        
        # pad A
        A_pad = np.zeros(self.pad_shape,dtype=np.complex64)
        A_pad[:A.shape[0],:A.shape[1]] = A.toarray().astype(np.float32)

        # allocate space on the gpu and send A there
        self.sol_hat_gpu = thr.to_device(A_pad)
        
        # create compiled fft procedure
        self.fft_proc = fft.FFT(self.sol_hat_gpu)
        self.fft_proc_c = self.fft_proc.compile(thr)
        
        # find fft, replacing the A_pad on gpu
        self.fft_proc_c(self.sol_hat_gpu,self.sol_hat_gpu,0)
        
    def fftconv2(self,B):
        '''Update current fourier solution with filter B.
        
        The work to be done in here is padding B appropriately, shifting
        B so that the center is at B[0,0] with wrap-around.
        
        Args:
            B: 2D array'''
            
        # Get array shape information
        mmid = np.array(B.shape).astype(int)//2
        
        # memory check!
        assert api.cuda.mem_get_info()[0] > self.pad_shape[0]*self.pad_shape[1]*(
            np.dtype(np.float32).itemsize + np.dtype(np.complex64).itemsize)
        
        # allocate temporary space on the gpu and arrange B there appropriately
        B_pad = np.zeros(self.pad_shape,np.complex64)
        B_pad[:mmid[0]+1,:mmid[1]+1] = B[mmid[0]:,mmid[1]:].astype(np.float32)
        B_pad[:mmid[0]+1,-mmid[1]:] = B[mmid[0]:,:mmid[1]].astype(np.float32)
        B_pad[-mmid[0]:,-mmid[1]:] = B[:mmid[0],:mmid[1]].astype(np.float32)
        B_pad[-mmid[0]:,:mmid[1]+1] = B[:mmid[0],mmid[1]:].astype(np.float32)

        B_gpu = thr.to_device(B_pad)
        
        # fft and solution update
        self.fft_proc_c(B_gpu,B_gpu,0)
        self.sol_hat_gpu *= B_gpu
        
    def get_cursol(self,dom_shape,negval=1e-6):
        '''Return the current solution (requires ifft) with small values removed
        
        Args:
            dom_len: domain length (number of cells) of returned solution
            
        Returns:
            coo matrix, current solution with shape (dom_len,dom_len)'''
        
        # memory check!
        assert api.cuda.mem_get_info()[0] > self.pad_shape[0]*self.pad_shape[1]*(
            np.dtype(np.float32).itemsize)
        
        # Assign temporary space for ifft and calculate
        cursol_gpu = thr.array(self.pad_shape,dtype=np.complex64)
        self.fft_proc_c(cursol_gpu,self.sol_hat_gpu,1)

	    # pull down current solution and return unpadded coo_matrix
        return sparse.coo_matrix(cursol_gpu.real[:dom_shape[0],:dom_shape[1]].get())

        #TODO: add method to return cursol with neg values zeroed out
