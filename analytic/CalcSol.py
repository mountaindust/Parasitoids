"""Module for calculating solution steps in a multi-day simulation.

Author: Christopher Strickland  
Email: cstrickland@samsi.info"""

import numpy as np
from scipy import sparse, fftpack
import config

if config.cuda:
    try:
        import cuda_lib
        NO_CUDA = False
    except ImportError:
        print('CUDA libraries not found. Running with NO_CUDA option.')
        config.cuda = False
        NO_CUDA = True
else:
    NO_CUDA = True

def fft2(A,Ashape):
    '''Return the fft of a coo sparse matrix signal A.
    
    Args:
        A: Coo sparse matrix
        Ashape: Shape of output array
        
    Returns:
        fft2 of A padded with zeros in the shape of Ashape'''
    mmid = (Ashape/2).astype(int)
    pad_shape = A.shape + mmid
    A_hat = np.zeros(pad_shape)
    A_hat[:A.shape[0],:A.shape[1]] = A.toarray()
    return fftpack.fft2(A_hat,overwrite_x=True) # test that A is unaltered!
    
    
    
def ifft2(A_hat,Ashape,CUDA_FLAG=(not NO_CUDA)):
    '''Return the ifft of A_hat truncated to Ashape as a coo matrix.
    
    This is the slowest function call.'''
    
    return sparse.coo_matrix(fftpack.ifft2(A_hat)[:Ashape[0],:Ashape[1]])
    
    
    
def fftconv2(A_hat,B,CUDA_FLAG=(not NO_CUDA)):
    '''Update A_hat as A_hat *= B_hat
    
    Args:
        A_hat: fft array
        B: 2D array
        
    Returns:
        fft array, A_hat times the fft of B.
    
    The work to be done in here is padding B appropriately, shifting
    B so that the center is at B[0,0] with wrap-around.'''
    
    mmid = (np.array(B.shape)/2).astype(int)
    pad_shape = A_hat.shape
    if CUDA_FLAG: #needs to be redone based on new function definition
        # return sparse.coo_matrix(cuda_lib.fftconv2(A,B))
        pass
    else:
        B_hat = np.zeros(pad_shape)
        B_hat[:mmid[0]+1,:mmid[1]+1] = B[mmid[0]:,mmid[1]:]
        B_hat[:mmid[0]+1,-mmid[1]:] = B[mmid[0]:,:mmid[1]]
        B_hat[-mmid[0]:,-mmid[1]:] = B[:mmid[0],:mmid[1]]
        B_hat[-mmid[0]:,:mmid[1]+1] = B[:mmid[0],mmid[1]:]
        B_hat = fftpack.fft2(B_hat)
        A_hat *= B_hat
        # return sparse.coo_matrix(
            # fftpack.ifft2(B_hat,overwrite_x=True)[:A.shape[0],:A.shape[1]].real)
            
            

def r_small_vals(A,negval=1e-6):
    '''Remove negligible values from the given coo sparse matrix and returns a
    coo matrix. This process significantly decreases the size of a solution, 
    saving storage and decreasing the time it takes to write to disk.
    
    A CUDA version might be warranted if really fast save time needed.'''
    
    A = A.todok()
    vallist = []
    rowlist = []
    collist = []
    for key, val in A.items():
        if val > 1e-6: # this is roundoff error territory for fft
            vallist.append(val)
            rowlist.append(key[0])
            collist.append(key[1])
    return sparse.coo_matrix((vallist,(rowlist,collist)),A.shape)
    
    
# def sconv2(A,B):
    # '''Return the sparse matrix convolution of the two inputs.
    # Return shape is given by A.shape and is a coo type sparse matrix.
    # This algorithm does not use fft, and is therefore going to be SLOW for
    # anything but the first convolution!
    
    # Credit for this algorithm: Bruno Luong'''
    # Ai,Aj,Avals = sparse.find(A) #Avals.size = 92001
    # Bi,Bj,Bvals = sparse.find(B) #Bvals.size = 3697
    
    # # these are freakishly enormous after the first conv...
    # # all sizes are 340,127,697 in length!!!
    # AI,BI = np.meshgrid(Ai,Bi,indexing='ij')
    # AJ,BJ = np.meshgrid(Aj,Bj,indexing='ij')
    
    # C = np.outer(Avals,Bvals)
    
    # ii = AI.flatten()+BI.flatten() - np.floor(B.shape[0]/2)
    # jj = AJ.flatten()+BJ.flatten() - np.floor(B.shape[1]/2)
    # b = np.logical_and.reduce((ii>=0, ii<A.shape[0], jj>=0, jj< A.shape[1]))
    
    # C_conv = sparse.coo_matrix((C.flatten()[b],(ii[b],jj[b])),A.shape)
    # return C_conv