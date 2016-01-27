#! /usr/bin/env python3
'''
Routines for plotting the results of the model 
in a resolution sensitive way

Author: Christopher Strickland'''

import sys
import warnings
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import Run

clrmp = cm.get_cmap('viridis')
clrmp.set_bad('w') # Second arg is alpha. 
                       # Later we can use this to have a map show through!

def plot_all(modelsol,days,domain_info,mask_val=0.00001):
    '''Function for plotting the model solution
    
    Args:
        modelsol: list of daily solutions, coo sparse
        days: list of day identifiers
        domain_info: rad_dist, rad_res
        mask_val: values less then this value will not appear in plotting'''
    # The general idea here will be to comprehend resolutions in multiples
    #   of 1000. So, 1000x1000, 2000x2000, etc.
    # The first level will require no reduction.
    
    cell_dist = domain_info[0]/domain_info[1] #dist from one cell to 
                                              #neighbor cell (meters).
    
    # assume domain is square, probably odd.
    midpt = domain_info[1]
        
    plt.figure()
    for n,sol in enumerate(modelsol):
        #find the maximum distance from the origin
        rmax = max(np.fabs(sol.row-midpt).max(),np.fabs(sol.col-midpt).max())
        #construct xmesh and a masked solution array based on this
        rmax = min(rmax+5,midpt) # add a bit of frame space
        xmesh = np.linspace(-rmax*cell_dist-cell_dist/2,
            rmax*cell_dist+cell_dist/2,rmax*2+2)
        sol_fm = np.flipud(np.ma.masked_less(
            sol.toarray()[midpt-rmax:midpt+rmax+1,midpt-rmax:midpt+rmax+1],
            mask_val))
        plt.clf()
        plt.pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp)
        plt.axis([xmesh[0],xmesh[-1],xmesh[0],xmesh[-1]])
        plt.xlabel('West-East (meters)')
        plt.ylabel('North-South (meters)')
        plt.title('Parasitoid probability after day {0}'.format(days[n]))
        plt.colorbar()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if n != len(modelsol)-1:
                plt.pause(3.5)
            else:
                plt.pause(0.0001)
                plt.show()

            
            
def plot(sol,day,domain_info,mask_val=0.00001):
    '''Plot a solution for a single day
    
    Args:
        sol: day solution, coo sparse
        day: day identifier (for text identification)
        domain_info: rad_dist, rad_res
        mask_val: values less then this value will not appear in plotting'''

    cell_dist = domain_info[0]/domain_info[1] #dist from one cell to 
                                              #neighbor cell (meters).
    
    # assume domain is square, probably odd.
    midpt = domain_info[1]

    plt.ion()
    plt.figure()
    #find the maximum distance from the origin
    rmax = max(np.fabs(sol.row-midpt).max(),np.fabs(sol.col-midpt).max())
    #construct xmesh and a masked solution array based on this
    rmax = min(rmax+5,midpt) # add a bit of frame space
    xmesh = np.linspace(-rmax*cell_dist-cell_dist/2,
        rmax*cell_dist+cell_dist/2,rmax*2+2)
    sol_fm = np.flipud(np.ma.masked_less(
        sol.toarray()[midpt-rmax:midpt+rmax+1,midpt-rmax:midpt+rmax+1],
        mask_val))
    plt.pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp)
    plt.axis([xmesh[0],xmesh[-1],xmesh[0],xmesh[-1]])
    plt.xlabel('West-East (meters)')
    plt.ylabel('North-South (meters)')
    plt.title('Parasitoid probability after day {0}'.format(day))
    plt.colorbar()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.draw()
        plt.pause(0.0001)
    
def main(argv):
    '''Function for plotting a previous result.
    
    The first argument in the list should be the location of a simulation file.
    '''
    filename = argv[0]
    if filename.rstrip()[-5:] == '.json':
        filename = filename.rstrip[:-5]
    elif filename.rstrip()[-4:] == '.npz':
        filename = filename.rstrip[:-4]

    # load parameters
    params = Run.Params()
    params.file_read_chg(filename)

    dom_len = params.domain_info[1]*2 + 1

    # load data
    modelsol = {}
    with np.load(filename+'.npz') as npz_obj:
        days = npz_obj['days']
        for day in days:
            V = npz_obj[str(day)+'_data']
            I = npz_obj[str(day)+'_row']
            J = npz_obj[str(day)+'_col']
            modelsol[str(day)] = sparse.coo_matrix((V,(I,J)),
                                                    shape=(dom_len,dom_len))

    while True:
        val = input('Enter a day to plot or ? to see a list of plottable days.'+
                    ' Enter q to quit:')
        val = val.strip()
        if val == '?':
            print(*days)
        elif val.lower() == 'q' or val.lower() == 'quit':
            break
        else:
            try:
                plot(modelsol[val],val,params.domain_info)
            except KeyError:
                print('Day {0} not found.'.format(val))
    
if __name__ == "__main__":
    main(sys.argv[1:])