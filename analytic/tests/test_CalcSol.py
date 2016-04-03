"""
Test suite for CalcSol, for use with py.test

Author: Christopher Strickland
"""

import pytest
import numpy as np
from scipy import sparse, signal, fftpack
import CalcSol as CS
import globalvars

# see if the machine wants to try to run cuda
try:
    with open('config.txt', 'r') as f:
        for line in f:
            words = line.split()
            for n,word in enumerate(words):
                if word == '#': #comment
                    break
                elif word == '=':
                    arg = words[n-1]
                    val = words[n+1]
                    if arg == 'cuda':
                        if val == 'True':
                            globalvars.cuda = True
                        elif val == 'False':
                            globalvars.cuda = False
                        else:
                            globalvars.cuda = bool(val)
except:
    pass

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
    
@pytest.fixture(scope="module")
def many_arrays():
    # for now, we will pad these with zeros to avoid the periodic bndry effects
    Adata = np.outer(range(5),np.arange(.1,.6,.1))
    A = np.zeros((55,55))
    A[25:30,25:30] = Adata
    Bdata = np.outer(np.arange(0,2.5,0.5),np.ones(5))
    B = np.zeros((55,55))
    B[25:30,25:30] = Bdata
    Cdata = np.outer(range(5,0,-1),np.arange(.1,.6,.1))
    C = np.zeros((55,55))
    C[25:30,25:30] = Cdata
    Ddata = np.outer(np.arange(1,0,-.2),np.arange(0,2.5,0.5))
    D = np.zeros((55,55))
    D[25:30,25:30] = Ddata
    return (A,B,C,D)
    
############                    Decorators                ############

cuda_run = pytest.mark.skipif(not globalvars.cuda,
    reason = 'need globalvars.cuda == True')

###############################################################################
#                                                                             #
#                                   Tests                                     #
#                                                                             #
###############################################################################

def test_fftconv2(two_arrays):
    A,B = two_arrays
    A_hat = CS.fft2(sparse.coo_matrix(A),np.array(B.shape))
    CS.fftconv2(A_hat,sparse.csr_matrix(B))
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
    CS.fftconv2(A_hat,sparse.csr_matrix(B))
    C = CS.ifft2(A_hat,A.shape).toarray()
    assert not np.iscomplexobj(C)
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
    cu_solver.fftconv2(sparse.csr_matrix(B))
    C = cu_solver.get_cursol(A.shape)
    
    assert np.allclose(C.toarray(),signal.fftconvolve(A,B,'same'))
    assert np.all(A == two_arrays[0])
    assert np.all(B == two_arrays[1])

def test_back_solve(many_arrays):
    '''Test the backsolve against straightfoward convolution'''
    A,B,C,D = many_arrays
    # Each array has the same shape
    C_hat = CS.fft2(sparse.coo_matrix(C),A.shape)
    CS.fftconv2(C_hat,sparse.csr_matrix(D)) # overwrites C_hat
    bckCD = CS.back_solve([sparse.csr_matrix(A),sparse.csr_matrix(B)],
        C_hat,A.shape)
    
    B_hat = CS.fft2(sparse.coo_matrix(B),A.shape)
    CS.fftconv2(B_hat,sparse.csr_matrix(C))
    CS.fftconv2(B_hat,sparse.csr_matrix(D))
    BCD = CS.ifft2(B_hat,B.shape).toarray()
    
    A_hat = CS.fft2(sparse.coo_matrix(A),A.shape)
    CS.fftconv2(A_hat,sparse.csr_matrix(B))
    CS.fftconv2(A_hat,sparse.csr_matrix(C))
    CS.fftconv2(A_hat,sparse.csr_matrix(D))
    ABCD = CS.ifft2(A_hat,A.shape).toarray()

    # there's periodic boundary issues here that still need to be addressed...
    assert np.allclose(bckCD[1].toarray(),BCD)
    assert np.allclose(bckCD[0].toarray(),ABCD)
    
@cuda_run
def test_cuda_back_solve(many_arrays):
    '''Test the back_solve method in cuda_lib'''
    import cuda_lib
    A,B,C,D = many_arrays
    # Each array has the same shape
    cu_solver = cuda_lib.CudaSolve(sparse.coo_matrix(C),A.shape)
    cu_solver.fftconv2(sparse.csr_matrix(D))
    bckCD = cu_solver.back_solve([sparse.csr_matrix(A),sparse.csr_matrix(B)],
        A.shape)
        
    B_hat = CS.fft2(sparse.coo_matrix(B),A.shape)
    CS.fftconv2(B_hat,sparse.csr_matrix(C))
    CS.fftconv2(B_hat,sparse.csr_matrix(D))
    BCD = CS.ifft2(B_hat,B.shape).toarray()
    
    A_hat = CS.fft2(sparse.coo_matrix(A),A.shape)
    CS.fftconv2(A_hat,sparse.csr_matrix(B))
    CS.fftconv2(A_hat,sparse.csr_matrix(C))
    CS.fftconv2(A_hat,sparse.csr_matrix(D))
    ABCD = CS.ifft2(A_hat,A.shape).toarray()
    
    # there's periodic boundary issues here that still need to be addressed...
    # abs tolerance has to be jacked up here... it looks like roundoff errors
    #    in float32 build up quite a lot over several convolutions.
    #    Keep in mind these are bigger numbers we are convolution though, so
    #    fewer decimal points can be maintained compared to sim numbers < 1.
    assert np.allclose(bckCD[1].toarray(),BCD,atol=1e-04)
    assert np.allclose(bckCD[0].toarray(),ABCD,atol=1e-03)
