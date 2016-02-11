"""
Test suite for CalcSol, for use with py.test

Author: Christopher Strickland
"""

import pytest
import numpy as np
from scipy import sparse, signal, fftpack
import CalcSol as CS
import config

###############################################################################
#                                                                             #
#                              Test Fixtures                                  #  
#                                                                             #
###############################################################################

@pytest.fixture(scope="module")
def two_arrays():
    A = np.outer(range(10),range(1,11))
    B = np.outer(range(4,-1,-1),range(8,-1,-2))
    return (A,B)
    
############                    Decorators                ############

cuda_run = pytest.mark.skipif(not config.cuda,
    reason = 'need config.cuda == True')

###############################################################################
#                                                                             #
#                                   Tests                                     #
#                                                                             #
###############################################################################

def test_fftconv2(two_arrays):
    A,B = two_arrays
    A_hat = CS.fft2(sparse.coo_matrix(A),np.array(B.shape))
    CS.fftconv2(A_hat,B)
    #make sure something new is here
    assert not np.all(A_hat == CS.fft2(sparse.coo_matrix(A),np.array(B.shape))) 
    assert np.all(B == two_arrays[1]) #unchanged
    assert np.allclose(fftpack.ifft2(A_hat)[:A.shape[0],:A.shape[1]].real,
        signal.convolve2d(A,B,'same'))
    
def test_convolve_same(two_arrays):
    '''Test the full convolution sequence'''
    A,B = two_arrays
    # expand the arrays by something slightly larger than necessary
    fft_shape = np.array([A.shape[0]+6,A.shape[1]+6])
    A_hat = CS.fft2(sparse.coo_matrix(A),fft_shape)
    # update A_hat with the fft convolution
    CS.fftconv2(A_hat,B)
    C = CS.ifft2(A_hat,A.shape).toarray()
    
    assert np.allclose(C,signal.fftconvolve(A,B,'same'))
    assert np.all(A == two_arrays[0])
    assert np.all(B == two_arrays[1])
    
@cuda_run
def test_cuda_convolve(two_arrays):
    '''Test the full convolution sequence on the GPU'''
    import cuda_lib
    A,B = two_arrays
    max_shape = np.array(A.shape) + 6
    As = sparse.coo_matrix(A)
    cu_solver = cuda_lib.CudaSolve(As,max_shape)
    cu_solver.fftconv2(B)
    C = cu_solver.get_cursol(A.shape)
    
    assert np.allclose(C.toarray(),signal.fftconvolve(A,B,'same'))
    assert np.all(A == two_arrays[0])
    assert np.all(B == two_arrays[1])
