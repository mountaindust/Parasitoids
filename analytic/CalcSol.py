"""Module for calculating solution steps in a multi-day simulation.

Author: Christopher Strickland  
Email: cstrickland@samsi.info"""

import sys
import numpy as np
from scipy import sparse, fftpack
import globalvars

def fft2(A,filt_shape):
    '''Return the fft of a coo sparse matrix signal A.
    
    Args:
        A: Coo sparse matrix
        filt_shape: Shape of filter largest array
        
    Returns:
        fft2 of A padded with zeros in the shape of filt_shape'''
    mmid = (filt_shape/2).astype(int)
    pad_shape = A.shape + mmid
    A_hat = np.zeros(pad_shape)
    A_hat[:A.shape[0],:A.shape[1]] = A.toarray()
    return fftpack.fft2(A_hat,overwrite_x=True) # test that A is unaltered!
    
    
    
def ifft2(A_hat,Ashape):
    '''Return the ifft of A_hat truncated to Ashape as a coo matrix.
    
    This is the slowest function call.'''
    
    return sparse.coo_matrix(fftpack.ifft2(A_hat)[:Ashape[0],:Ashape[1]].real)
    
    
    
def fftconv2(A_hat,B):
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
    B_hat = np.zeros(pad_shape)
    B_hat[:mmid[0]+1,:mmid[1]+1] = B[mmid[0]:,mmid[1]:]
    B_hat[:mmid[0]+1,-mmid[1]:] = B[mmid[0]:,:mmid[1]]
    B_hat[-mmid[0]:,-mmid[1]:] = B[:mmid[0],:mmid[1]]
    B_hat[-mmid[0]:,:mmid[1]+1] = B[:mmid[0],mmid[1]:]
    B_hat = fftpack.fft2(B_hat)
    A_hat *= B_hat
    # return sparse.coo_matrix(
        # fftpack.ifft2(B_hat,overwrite_x=True)[:A.shape[0],:A.shape[1]].real)
            
            

def r_small_vals(A,negval=1e-8):
    '''Remove negligible values from the given coo sparse matrix. 
    This process significantly decreases the size of a solution, 
    saving storage and decreasing the time it takes to write to disk.
    The sum of the removed values is added back to the origin to maintain a
    probability mass function.
    
    A CUDA version might be warranted if really fast save time needed.'''
    midpt = A.shape[0]//2 #assume domain is square
    
    mask = np.empty(A.data.shape,dtype=bool)
    for n,val in enumerate(A.data):
        if val < negval: # this should be roundoff error territory
            mask[n] = False
        else:
            mask[n] = True
    A_red = sparse.coo_matrix((A.data[mask],(A.row[mask],A.col[mask])),A.shape)
    # to get a pmf, add back the lost probability evenly
    A_red.data += (1-A_red.data.sum())/A_red.data.size
    return A_red
    
def get_solutions(modelsol,pmf_list,days,ndays,dom_len,max_shape):
    '''Find model solutions from a list of daily probability densities and given
    the distribution after the first day.
    
    Runs on GPU if globalvars.cuda is True and NO_CUDA is False.
    
    Args:
        modelsol: list of model solutions with the first day's already entered
        pmf_list: list of probability densities. len(pmf_list) == len(days)
        days: list of day dictionary keys, mostly for feedback
        ndays: number of days to run simulation
        dom_len: number of cells across one side of the domain
        max_shape: largest filter shape, based on largest in pmf_list
            
    Modifies:
        modelsol
    '''
    
    if globalvars.cuda:
        try:
            import cuda_lib
            NO_CUDA = False
        except ImportError:
            print('CUDA libraries not found. Running with NO_CUDA option.')
            globalvars.cuda = False
            NO_CUDA = True
        except Exception as e:
            print('Error encountered while importing CUDA:')
            print(str(e))
            globalvars.cuda = False
            NO_CUDA = True
    else:
        NO_CUDA = True
    
    if globalvars.cuda and not NO_CUDA:
        # go to GPU.
        print('Sending to GPU and finding fft of first day...')
        gpu_solver = cuda_lib.CudaSolve(modelsol[0],max_shape)
        # update and return solution for each day
        for n,day in enumerate(days[1:ndays]):
            print('Updating convolution for day {0}...'.format(day))
            gpu_solver.fftconv2(pmf_list[n+1].toarray())
            print('Finding ifft for day {0} and reducing...'.format(day))
            modelsol.append(gpu_solver.get_cursol([dom_len,dom_len]))
    else:
        print('Finding fft of first day...')
        cursol_hat = fft2(modelsol[0],max_shape)
        
        for n,day in enumerate(days[1:ndays]):
            print('Updating convolution for day {0}...'.format(day))
            # modifies cursol_hat
            fftconv2(cursol_hat,pmf_list[n+1].toarray())
            print('Finding ifft for day {0}...'.format(day))
            big_sol = ifft2(cursol_hat,[dom_len,dom_len])
            print('Reducing solution...')
            modelsol.append(r_small_vals(big_sol))
