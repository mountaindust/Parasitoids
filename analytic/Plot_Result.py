#! /usr/bin/env python3
'''
Routines for plotting the results of the model 
in a resolution sensitive way

Author: Christopher Strickland'''

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import matplotlib.cm as cm


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
    xmesh = np.arange(-midpt*cell_dist-cell_dist/2,midpt*cell_dist+cell_dist/2 + 
        cell_dist/3,cell_dist)
        
    # assume that max aggregation occurs in the first solution, and use that
    #   to set the maximum of the color scale
    pmax = modelsol[0].data.max()
    # clrnorm = mcl.Normalize(0,pmax,clip=True)
    clrmp = cm.get_cmap('viridis')
    clrmp.set_bad('w') # Second arg is alpha. 
                       # Later we can use this to have a map show through!
        
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
        if n != len(modelsol)-1:
            plt.pause(4)
        else:
            plt.pause(0.0001)
            plt.show()

def main(argv):
    '''Function for plotting a previous result'''
    #argv is a list, first entry is the 'Plot_Result.py'
    pass
    
if __name__ == "__main__":
    main(sys.argv)