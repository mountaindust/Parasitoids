"""Module for calculating solution steps in a multi-day simulation.

Author: Christopher Strickland  
Email: cstrickland@samsi.info"""

import sys
import numpy as np
from scipy import sparse, fftpack
import globalvars

def fft2(A,filt_shape):
    '''Return the fft of a sparse matrix signal A.
    
    Args:
        A: Coo sparse matrix
        filt_shape: Shape of filter largest array
        
    Returns:
        fft2 of A padded with zeros in the shape of filt_shape'''
    mmid = np.array(filt_shape)//2
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
        B: 2D array, shape must be odd
        
    Returns:
        fft array, A_hat times the fft of B.
    
    The work to be done in here is padding B appropriately, shifting
    B so that the center is at B[0,0] with wrap-around.'''
    
    mmid = np.array(B.shape)//2 #this fails if B.shape is even!
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
        
        
        
def back_solve(prev_spread,cursol_hat,dom_shape):
    '''For each filter in prev_spread, convolute progressively in reverse order.
        The number of arrays returned will be equal to len(prev_spread).
        The last filter in prev_spread will be applied first, and the result
            returned (last). Then the next to last filter is applied to that 
            result to be returned next-to-last, etc.
        
        Args:
            prev_spread: list of filters to apply (chronological order)
            cursol_hat: fft of current solution, calculated from last emerg day
            dom_shape: shape of returned solution
            
        Returns:
            list of coo matrices, in order of wasp emerg., w/ shape dom_len^2'''
            
    # store back solutions here in reverse chronological order
    bcksol = []
    bcksol_hat = np.array(cursol_hat)
    for B in prev_spread[::-1]:
        B = B.toarray()
        # Convolution
        mmid = np.array(B.shape)//2
        pad_shape = cursol_hat.shape
        B_hat = np.zeros(pad_shape)
        B_hat[:mmid[0]+1,:mmid[1]+1] = B[mmid[0]:,mmid[1]:]
        B_hat[:mmid[0]+1,-mmid[1]:] = B[mmid[0]:,:mmid[1]]
        B_hat[-mmid[0]:,-mmid[1]:] = B[:mmid[0],:mmid[1]]
        B_hat[-mmid[0]:,:mmid[1]+1] = B[:mmid[0],mmid[1]:]
        B_hat = fftpack.fft2(B_hat)
        bcksol_hat = B_hat * bcksol_hat
        
        # ifft and reduce
        bcksol.append(ifft2(bcksol_hat,dom_shape))
        
    # return list in emergence order
    return bcksol[::-1]
            

def r_small_vals(A,negval=1e-8):
    '''Remove negligible values from the given coo sparse matrix. 
    This process significantly decreases the size of a solution, 
    saving storage and decreasing the time it takes to write to disk.
    The sum of the removed values is added back to the origin to maintain a
    probability mass function.
    
    A CUDA version might be warranted if really fast save time needed.'''
    
    if not sparse.isspmatrix_coo(A):
        A = sparse.coo_matrix(A)
        
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
    
    Need boundary checking of solutions to prevent rollover in Fourier space
    
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
            print('Updating convolution for day {0} PR...'.format(n+2))
            gpu_solver.fftconv2(pmf_list[n+1].tocsr(),n==0)
            print('Finding ifft for day {0}...'.format(n+2))
            modelsol.append(r_small_vals(
                gpu_solver.get_cursol([dom_len,dom_len])))
    else:
        print('Finding fft of first day...')
        cursol_hat = fft2(modelsol[0],max_shape)
        
        for n,day in enumerate(days[1:ndays]):
            print('Updating convolution for day {0} PR...'.format(n+2))
            # modifies cursol_hat
            fftconv2(cursol_hat,pmf_list[n+1].tocsr())
            # get real solution
            print('Finding ifft for day {0} and reducing...'.format(n+2))
            modelsol.append(r_small_vals(ifft2(cursol_hat,[dom_len,dom_len])))
            
            

def get_populations(r_spread,pmf_list,days,ndays,dom_len,max_shape,
                    r_dur,r_number,dist):
    '''Find expected wasp densities from a list of daily probability densities
    and given the distribution after the last release day.
    
    Need boundary checking of solutions to prevent rollover in Fourier space
    
    Runs on GPU if globalvars.cuda is True and NO_CUDA is False.
    
    Args:
        r_spread: list of model probabilities for each release day
        pmf_list: list of probability densities. len(pmf_list) == len(days)
        days: list of day dictionary keys, mostly for feedback
        ndays: number of days to run simulation
        dom_len: number of cells across one side of the domain
        max_shape: largest filter shape, based on largest in pmf_list
        r_dur: duration of release, days (int)
        r_number: total number of wasps released, assume uniform release
        dist: emergence distribution during release
        
    Returns:
        popmodel: expected wasp population numbers on each day
    '''
    
    # holds probability solution for each release day, in order
    curmodelsol = [0 for ii in range(r_dur)] #holder for current solutions
    # holds population solution for each day
    popmodel = []
    
    # first day population spread is just via r_spread[0].
    # the rest is still at the origin.
    popmodel.append(r_small_vals(r_spread[0]).tocsr()*r_number*dist(1))
    popmodel[0][dom_len//2,dom_len//2] += r_number*(1- dist(1))
    
    curmodelsol[0] = r_spread[0].tocoo()
    
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
        print('Finding spread during release days on GPU...')
        # successive release day population spread
        for day in range(1,r_dur):
            gpu_solver = cuda_lib.CudaSolve(r_spread[day],max_shape)
            curmodelsol[day] = r_spread[day].tocoo()
            # back solve to get previous solutions
            curmodelsol[:day] = gpu_solver.back_solve(r_spread[:day],
                                                      [dom_len,dom_len])
            # get population spread
            popmodel.append(r_small_vals(np.sum(curmodelsol[d]*dist(d+1)
                for d in range(day+1))*r_number).tocsr())
            popmodel[-1][dom_len//2,dom_len//2] += (1-np.sum(
                dist(d+1) for d in range(day+1)))*r_number
        # update and return solutions for each day
        for n,day in enumerate(days[r_dur:ndays]):
            print('Updating convolution for day {0} PR...'.format(r_dur+n+1))
            # update current GPU solution based on last day of release
            gpu_solver.fftconv2(pmf_list[n+r_dur].tocsr(),n==0)
            print('Finding ifft for day {0}...'.format(r_dur+n+1))
            # get current GPU solution based on last day of release
            curmodelsol[-1] = gpu_solver.get_cursol([dom_len,dom_len])
            # get GPU solutions for previous release days
            curmodelsol[:-1] = gpu_solver.back_solve(r_spread[:-1],
                                                     [dom_len,dom_len])
            # get new population spread
            popmodel.append(r_small_vals(np.sum(curmodelsol[d]*dist(d+1)
                for d in range(r_dur))*r_number).tocsr())
            
    else: # no CUDA.
        print('Finding spread during release days...')
        # successive release day population spread
        for day in range(1,r_dur):
            cursol_hat = fft2(r_spread[day],max_shape)
            curmodelsol[day] = r_spread[day]
            # back solve to get previous solutions
            curmodelsol[:day] = back_solve(r_spread[:day],
                                            cursol_hat,[dom_len,dom_len])
            # get population spread
            popmodel.append(r_small_vals(np.sum(curmodelsol[d]*dist(d+1)
                for d in range(day+1))*r_number).tocsr())
            popmodel[-1][dom_len//2,dom_len//2] += (1-np.sum(
                dist(d+1) for d in range(day+1)))*r_number
        # update and return solutions for each day
        for n,day in enumerate(days[r_dur:ndays]):
            print('Updating convolution for day {0} PR...'.format(r_dur+n+1))
            # modifies cursol_hat
            fftconv2(cursol_hat,pmf_list[n+r_dur].toarray())
            print('Finding ifft for day {0}...'.format(r_dur+n+1))

            curmodelsol[-1] = ifft2(cursol_hat,[dom_len,dom_len])
            # get solutions for previous release days and reduce
            curmodelsol[:-1] = back_solve(r_spread[:-1],cursol_hat,
                                            [dom_len,dom_len])
            # get new population spread
            popmodel.append(r_small_vals(np.sum(curmodelsol[d]*dist(d+1)
                for d in range(r_dur))*r_number).tocsr())
            
    return popmodel