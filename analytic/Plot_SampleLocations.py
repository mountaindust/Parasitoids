#! /usr/bin/env python3

'''This module plots the locations at which data was collected based on the
information loaded in Bayes_MCMC.py. Requires Pillow, a Google static maps key,
and an internet connection to pull in satellite imagery. The static maps key
should be specified in config.txt.

Author: Christopher Strickland  
Email: cstrickland@samsi.info 
'''

__author__ = "Christopher Strickland"
__email__ = "cstrickland@samsi.info"
__status__ = "Development"
__version__ = "0.0"
__copyright__ = "Copyright 2015, Christopher Strickland"

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.path import Path
import matplotlib.patches as patches
from Run import Params
from Bayes_MCMC import LocInfo
from Plot_Result import get_satellite

def main():
    params = Params()
    # Set up location here with command line arguments in a list.
    params.cmd_line_chg(['--kalbar'])
    locinfo = LocInfo(params.dataset,params.coord,params.domain_info)
    dom_len = params.domain_info[1]*2 + 1
    # get satellite image of the full domain and add to plot
    sat_img = get_satellite(params.maps_key,params.maps_service,
        params.coord,params.domain_info[0])
    plot_limits = [-params.domain_info[0],params.domain_info[0],
                    -params.domain_info[0],params.domain_info[0]]
    ax = plt.axes()
    ax.axis(plot_limits)
    ax.imshow(sat_img,zorder=0,extent=plot_limits)
    # attach the emergence field paths to PathPatch and add to plot
    for poly in locinfo.field_polys.values():
        ax.add_patch(patches.PathPatch(poly,facecolor='none',edgecolor='r',lw=2))
        
    # we should probably plot the cells here too to make sure they match.
    # use a sparse matrix of ones in the sample location qand plot with
    #   pcolormesh using the clrmp.set_bad
    clrmp = cm.get_cmap('gray')
    #clrmp.set_bad(alpha=0)
    
    grayvals = np.zeros((dom_len,dom_len))
    for field in locinfo.field_cells.values():
        for r,c in field:
            grayvals[r,c] = 0.5
    cell_locs = np.flipud(np.ma.masked_values(grayvals,0))
    xmesh = np.linspace(-params.domain_info[0],params.domain_info[0],dom_len)
    ax.pcolormesh(xmesh,xmesh,cell_locs,cmap=clrmp,vmax=1,alpha=0.5,zorder=1)
    
    # now plot the grid cells
    grid_locs = np.zeros((dom_len,dom_len))
    for n,x in enumerate(locinfo.grid_cells[0,:]):
        if locinfo.grid_data['samples'][n] == 90: # this bit is data specific
            grid_locs[locinfo.grid_cells[0,n],locinfo.grid_cells[1,n]] = 0.01
        else:
            grid_locs[locinfo.grid_cells[0,n],locinfo.grid_cells[1,n]] = 1
    grid_locs = np.flipud(np.ma.masked_values(grid_locs,0))
    ax.pcolormesh(xmesh,xmesh,grid_locs,cmap=cm.get_cmap('cool'),vmax=1,zorder=2)
    ax.set_xlabel('West-East (meters)')
    ax.set_ylabel('North-South (meters)')
    ax.set_title('Parasitoid collection site locations')
    plt.show()
    
    
    
if __name__ == "__main__":
    main()