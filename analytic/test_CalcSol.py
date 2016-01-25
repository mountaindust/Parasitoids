"""
Test suite for CalcSol, for use with py.test

Author: Christopher Strickland
"""

import pytest
import numpy as np
from scipy import sparse, signal, fftpack
import CalcSol as CS

def test_fftconv2():
    A = np.outer(range(10),range(1,11))
    A_hat = fftpack.fft2(A)
    B = np.outer(range(4,-1,-1),range(8,-1,-2))
    CS.fftconv2(A_hat,B,False)
    assert not np.all(A_hat == fftpack.fft2(A)) #make sure something new is here
    assert np.all(B == np.outer(range(4,-1,-1),range(8,-1,-2))) #unchanged
    # need to test effects against signal.convolve2d
    
# def test_sconv2():
    # # Get some matrices
    # A = np.outer(range(10),range(1,11))
    # B = np.outer(range(4,-1,-1),range(8,-1,-2))
    # C = PM.sconv2(A,B)
    # assert sparse.issparse(C)
    # # check that the result matches signal.convolve2d
    # assert np.all(C == signal.convolve2d(A,B,'same'))
    # # make sure sparse matrices work as well
    # A_coo = sparse.coo_matrix(A)
    # B_coo = sparse.coo_matrix(B)
    # assert np.all(PM.sconv2(A_coo,B_coo) == signal.convolve2d(A,B,'same'))
    
# @slow
# def test_cuda_outer():
    # from cuda_lib import outer
    # A = np.outer(range(10),range(1,11))
    # B = np.outer(range(4,-1,-1),range(8,-1,-2))
    # C = outer(A,B)
    # assert np.all(np.outer(A,B) == C)