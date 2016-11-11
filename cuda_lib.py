import numpy as np
from scipy import sparse

import reikna.cluda as cluda
import reikna.fft as fft
from reikna.transformations import split_complex

api = cluda.cuda_api()
thr = api.Thread.create()

class CudaSolve():
    
    def __init__(self,A,max_shape):
        '''Initialize CUDA solver with fft of solution after first day.
        
        Args:
            A: First day's spread, sparse matrix
            max_shape: Shape of transformed solution'''
        
        # determine the shape of the padded solution array
        mmid = np.array(max_shape)//2
        pads = A.shape + mmid
        self.pad_shape = (int(pads[0]),int(pads[1]))
        
        # memory check!
        assert api.cuda.mem_get_info()[0] > self.pad_shape[0]*self.pad_shape[1]*(
            np.dtype(np.float32).itemsize + np.dtype(np.complex64).itemsize)
 
        # pad A
        A_pad = np.zeros(self.pad_shape,dtype=np.complex64)
        A_pad[:A.shape[0],:A.shape[1]] = A.toarray().astype(np.float32)

        # allocate space on the gpu and send A there
        self.sol_hat_gpu = api.gpuarray.to_gpu(A_pad)
        
        # create compiled fft procedure
        fft_proc = fft.FFT(self.sol_hat_gpu)
        self.fft_proc_c = fft_proc.compile(thr)
        # call signature is (sol_hat, sol, 0) or for ifft: (sol, sol_hat, 1)
        
        # create separate ifft procedure that splits real/imaginary parts
        splitc = split_complex(self.sol_hat_gpu)
        fft_proc.parameter.output.connect(
            splitc, splitc.input, sol_r=splitc.real, sol_i=splitc.imag)
        self.ifft_proc_c = fft_proc.compile(thr)
        # call signature is (sol_real,sol_imag,sol_hat,1)
        
        # find fft, replacing the A_pad on gpu
        self.fft_proc_c(self.sol_hat_gpu,self.sol_hat_gpu,0)
        
        
        
    def fftconv2(self,B,mem_print=True):
        '''Update current fourier solution with filter B.
        
        The work to be done in here is padding B appropriately, shifting
        B so that the center is at B[0,0] with wrap-around.
        
        Args:
            B: 2D slicable sparse matrix
            mem_print: bool. turn on/off GPU memory report'''
            
        # Get array shape information
        mmid = np.array(B.shape).astype(int)//2
        
        # memory check!
        assert api.cuda.mem_get_info()[0] > self.pad_shape[0]*self.pad_shape[1]*(
            np.dtype(np.float32).itemsize + np.dtype(np.complex64).itemsize)
        
         
        # allocate temporary space on the gpu and arrange B there appropriately
        B_pad = np.zeros(self.pad_shape,np.complex64)
        B_pad[:mmid[0]+1,:mmid[1]+1] = \
            B[mmid[0]:,mmid[1]:].astype(np.float32).toarray()
        B_pad[:mmid[0]+1,-mmid[1]:] = \
            B[mmid[0]:,:mmid[1]].astype(np.float32).toarray()
        B_pad[-mmid[0]:,-mmid[1]:] = \
            B[:mmid[0],:mmid[1]].astype(np.float32).toarray()
        B_pad[-mmid[0]:,:mmid[1]+1] = \
            B[:mmid[0],mmid[1]:].astype(np.float32).toarray()

        B_gpu = api.gpuarray.to_gpu(B_pad)
        if mem_print:
            print('GPU memory: free={0}, total={1}'.format(
                api.cuda.mem_get_info()[0],api.cuda.mem_get_info()[1]))
 
        # fft and solution update
        self.fft_proc_c(B_gpu,B_gpu,0)
        self.sol_hat_gpu *= B_gpu
        
        
        
    def get_cursol(self,dom_shape,negval=1e-8):
        '''Return the current solution (requires ifft) with small values removed
        
        Args:
            dom_shape: shape of returned solution
            
        Returns:
            coo matrix, current solution with shape (dom_len,dom_len)'''
        
        # memory check!
        assert api.cuda.mem_get_info()[0] > self.pad_shape[0]*self.pad_shape[1]*(
            np.dtype(np.float32).itemsize)*2
        
        # Assign temporary space for real and complex ifft and calculate
        cursol_gpu_r = api.gpuarray.empty(self.pad_shape,dtype=np.float32)
        cursol_gpu_i = api.gpuarray.empty(self.pad_shape,dtype=np.float32)
        self.ifft_proc_c(cursol_gpu_r,cursol_gpu_i,self.sol_hat_gpu,1)
        
        # Remove negligable values from reported real solution
        cursol_gpu_i.set(np.zeros(cursol_gpu_r.shape).astype(np.float32))
        cursol_gpu_red = api.gpuarray.if_positive(cursol_gpu_r>negval,
            cursol_gpu_r,cursol_gpu_i)
            
        # Check for non-zero entries outside of the domain shape
        # We cannot do a max over a slice (non-contiguous array restriction), so
        #   we must copy the padded part of cursol_gpu_red first
        maxval = max(api.gpuarray.max(
            cursol_gpu_red[dom_shape[0]:,dom_shape[1]:].copy()).get(),
            api.gpuarray.max(
            cursol_gpu_red[:dom_shape[0],dom_shape[1]:].copy()).get(),
            api.gpuarray.max(
            cursol_gpu_red[dom_shape[0]:,:dom_shape[1]].copy()).get())
        if maxval > 0:
            # zero out the padded part of the solution and re-fft
            print('Re-fft')
            self.sol_hat_gpu = api.gpuarray.zeros_like(self.sol_hat_gpu)
            self.sol_hat_gpu[:dom_shape[0],:dom_shape[1]] = \
                cursol_gpu_r[:dom_shape[0],:dom_shape[1]].copy().astype('complex64')
            self.fft_proc_c(self.sol_hat_gpu,self.sol_hat_gpu,0)

	    # pull down current solution and return unpadded coo_matrix
        return sparse.coo_matrix(
            cursol_gpu_red[:dom_shape[0],:dom_shape[1]].get())

        #TODO: add method to return cursol with neg values zeroed out

        
    def back_solve(self,prev_spread,dom_shape,negval=1e-8):    
        '''For each filter in prev_spread, convolute progressively in reverse order.
        The number of arrays returned will be equal to len(prev_spread).
        The last filter in prev_spread will be applied first, and the result
            returned (last). Then the next to last filter is applied to that 
            result to be returned next-to-last, etc.
        
        Args:
            prev_spread: list of slicable filters to apply (chronological order)
            dom_shape: shape of returned solution
            
        Returns:
            list of coo matrices, in order of wasp emerg., w/ shape dom_len^2'''
        
        assert api.cuda.mem_get_info()[0] > self.pad_shape[0]*self.pad_shape[1]*(
            np.dtype(np.float32).itemsize)*4 # must hold two cmplx fft matrices
        
        # store back solutions here in reverse chronological order
        bcksol = []
        
        # start with the current solution
        bcksol_hat_gpu = self.sol_hat_gpu
        
        for B in prev_spread[::-1]:
            # Get array shape information
            mmid = np.array(B.shape).astype(int)//2
            
            # allocate temporary space on the gpu and arrange B there appropriately
            B_pad = np.zeros(self.pad_shape,np.complex64)
            B_pad[:mmid[0]+1,:mmid[1]+1] = \
                B[mmid[0]:,mmid[1]:].astype(np.float32).toarray()
            B_pad[:mmid[0]+1,-mmid[1]:] = \
                B[mmid[0]:,:mmid[1]].astype(np.float32).toarray()
            B_pad[-mmid[0]:,-mmid[1]:] = \
                B[:mmid[0],:mmid[1]].astype(np.float32).toarray()
            B_pad[-mmid[0]:,:mmid[1]+1] = \
                B[:mmid[0],mmid[1]:].astype(np.float32).toarray()
            
            B_gpu = api.gpuarray.to_gpu(B_pad)
            
            # fft and backwards solution update
            self.fft_proc_c(B_gpu,B_gpu,0)
            bcksol_hat_gpu = B_gpu * bcksol_hat_gpu #this does not appear to
                                                    #   overwite self.sol_hat_gpu
            
            # ifft over B_gpu to free the space
            self.fft_proc_c(B_gpu,bcksol_hat_gpu,1)
            B_gpu = B_gpu.real
            
            # Remove negligable values from reported real solution
            sol_gpu = api.gpuarray.zeros_like(B_gpu)
            sol_gpu = api.gpuarray.if_positive(B_gpu>negval,
                B_gpu,sol_gpu) #this might not work because sol_gpu on both sides?
                
            # Check for non-zero entries outside of the domain shape
            # We cannot do a max over a slice (non-contiguous array restriction)
            #   We must copy the padded part of cursol_gpu_red first
            maxval = max(api.gpuarray.max(
                sol_gpu[dom_shape[0]:,dom_shape[1]:].copy()).get(),
                api.gpuarray.max(
                sol_gpu[:dom_shape[0],dom_shape[1]:].copy()).get(),
                api.gpuarray.max(
                sol_gpu[dom_shape[0]:,:dom_shape[1]].copy()).get())
            if maxval > 0:
                print('Re-fft')
                # zero out the padded part of the solution and re-fft
                bcksol_hat_gpu = api.gpuarray.zeros_like(bcksol_hat_gpu)
                bcksol_hat_gpu[:dom_shape[0],:dom_shape[1]] = \
                    B_gpu[:dom_shape[0],:dom_shape[1]].copy().astype('complex64')
                self.fft_proc_c(bcksol_hat_gpu,bcksol_hat_gpu,0)
                
            # pull down current solution and add unpadded coo_matrix to list
            bcksol.append(sparse.coo_matrix(
                sol_gpu[:dom_shape[0],:dom_shape[1]].get()))
                
        # return list in chronological order
        return bcksol[::-1]
