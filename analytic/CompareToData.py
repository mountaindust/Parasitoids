#! /usr/bin/env python3
'''This module is for comparing model results to data and generating
publication quality figures.
'''

import numpy as np
from scipy import sparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import Run
from Plot_Result import get_satellite, r_small_vals

base_clrmp = cm.get_cmap('viridis')
# alter this colormap with alpha values
dict_list = []
for x in zip(*base_clrmp.colors):
    dict_list.append(tuple((n/(len(x)-1),val,val) for n,val in enumerate(x)))
cdict = {'red': dict_list[0], 'green': dict_list[1], 'blue': dict_list[2]}
cdict['alpha'] = ((0.0,0.65,0.3),
                  (1.0,0.65,0.3))
plt.register_cmap(name='alpha_viridis',data=cdict)
clrmp = cm.get_cmap('alpha_viridis')
clrmp.set_bad('w',alpha=0)


def main(modelsol,params,locinfo,bw=None):
    '''Compare model results to data, as contained in locinfo
    
    Args:
        modelsol: list of daily solutions, sparse
        params: Params object from Run.py
        locinfo: LocInfo object from Data_Import.py
        bw: set this to something not None for b/w
        '''
    # Assume here that an emerging parasitoid (in the data) came from the 
    #   average incubation time. This is both easier to explain and easier
    #   to implement.
        
    import Bayes_funcs
    avg_incubation = Bayes_funcs.max_incubation_time - \
        np.floor(Bayes_funcs.incubation_time.size/2)
    
    # if multiple collections were made for emergence, just compare results for
    #   the first one.
    sent_col_num = 0
    grid_col_num = 0
    
    ##### Gather sentinal fields data #####
    dframe = locinfo.sent_DataFrames[0]
    collection_date = locinfo.collection_datesPR[0].days
    # get dates emergence was observed
    obs_datesPR = dframe['datePR'].map(lambda t: t.days).unique()
    # walk these dates backward the average incubation period, cutting off
    #   at day 0. It is assumed that collection happened in the morning, and
    #   emergence observations occur at the end of each day. Thus, the last day
    #   oviposition was possible was the day BEFORE the collection day. 
    #   Since model results are given at the end of each day PR (same as 
    #   emergence observations), one day of incubation corresponds directly to 
    #   one model day. We assume incubation is longer than zero days in all cases.
    ovi_datesPR = np.maximum(obs_datesPR - avg_incubation,
                             np.zeros(obs_datesPR.size)).astype('int')
    # set up presence array - rows are fields, columns are presence dates
    sent_ovi_array = np.zeros((len(locinfo.sent_ids),ovi_datesPR[-1]+1))
    for n,obs_date in enumerate(obs_datesPR):
        sent_ovi_array[:,ovi_datesPR[n]] = \
            dframe[dframe['datePR'] == obs_date]['E_total'].values
    # cut the oviposition array at the collection date
    if sent_ovi_array.shape(1) > collection_date:
        sent_ovi_array = sent_ovi_array[:,:collection_date]
    if sent_ovi_array.shape(1) < collection_date:
        sent_ovi_array = np.pad(sent_ovi_array,((0,0),(0,
            collection_date - sent_ovi_array.shape(1))),'constant')
    # now sent_ovi_array can be directy compared to the density of wasps in each
    #   field on the same day PR
    
    ##### Calculate the density of wasps in each field on each day #####
    # Get the size of each cell in m**2
    cell_dist = params.domain_info[0]/params.domain_info[1]
    cell_size = (cell_dist)**2
    field_sizes_m = {}
    for key,val in locinfo.field_sizes:
        field_sizes_m[key] = val*cell_size
    # Collect the number of wasps in each field on each day up to 
    #   collection_date and calculate the wasps' density
    model_field_densities = np.zeros((len(locinfo.sent_ids),collection_date))
    for day in range(collection_date):
        for n,field_id in enumerate(locinfo.sent_ids):
            field_total = modelsol[day][locinfo.field_cells[field_id][:,0],
                                    locinfo.field_cells[field_id][:,1]].sum()
            model_field_densities[day,n] = field_total/field_sizes_m
            
    ##### Plot comparison #####
    # The first two columns can be stills of the model at 3, 6, 9 and 
    #   19 days PR, corresponding to data collection days. The last column
    #   should be two 3D histograms with the above data.
    plot_days = [2,5,8,18] # 0 = 1 day PR
    subplots = [231,232,234,235]
    sp3d = [233,236]
    
    # assume domain is square, probably odd.
    midpt = params.domain_info[1]
    #Establish a miminum for plotting based 0.00001 of the maximum
    mask_val = min(10**(np.floor(np.log10(sol.data.max()))-3),1)
    
    fig = plt.figure()
    ## Plot model result maps ##
    for ii in range(4):
        sol = modelsol[plot_days[ii]]
        ax = fig.add_subplot(subplots[ii])
        #remove all values that are too small to be plotted.
        sol_red = r_small_vals(sol,mask_val)
        #find the maximum distance from the origin
        rmax = max(np.fabs(sol_red.row-midpt).max(),np.fabs(sol_red.col-midpt).max())
        #construct xmesh and a masked solution array based on this
        rmax = int(min(rmax+5,midpt)) # add a bit of frame space
        xmesh = np.linspace(-rmax*cell_dist-cell_dist/2,
            rmax*cell_dist+cell_dist/2,rmax*2+2)
        sol_fm = np.flipud(np.ma.masked_less(
            sol_red.toarray()[midpt-rmax:midpt+rmax+1,midpt-rmax:midpt+rmax+1],
            mask_val))
        plot_limits = [xmesh[0],xmesh[-1],xmesh[0],xmesh[-1]]
        ax.axis(plot_limits)
        #find the max value excluding the middle area
        midpt2 = sol_fm.shape[0]//2
        sol_mid = np.array(sol_fm[midpt2-4:midpt2+5,midpt2-4:midpt2+5])
        sol_fm[midpt2-4:midpt2+5,midpt2-4:midpt2+5] = np.ma.masked #mask the middle
        sprd_max = np.max(sol_fm) #find max
        sol_fm[midpt2-4:midpt2+5,midpt2-4:midpt2+5] = sol_mid #replace values
        #get satellite image
        sat_img = get_satellite(params.maps_key,params.maps_service,
            params.coord,xmesh[-1])
        if sat_img is None:
            if bw is None: #color
                ax.pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp,vmax=sprd_max,alpha=1)             
            else: #black and white
                ax.pcolormesh(xmesh,xmesh,sol_fm,cmap=plt.get_cmap('gray'),
                            vmax=sprd_max,alpha=1)
        else:
            if bw is None: #color
                ax.imshow(sat_img,zorder=0,extent=plot_limits)
                ax.pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp,vmax=sprd_max,zorder=1)
            else: #black and white
                sat_img = sat_img.convert('L') #B/W satellite image w/ dither
                ax.imshow(sat_img,zorder=0,cmap=plt.get_cmap('gray'),
                            extent=plot_limits)
                ax.pcolormesh(xmesh,xmesh,sol_fm,cmap=plt.get_cmap('gray'),
                            vmax=sprd_max,zorder=1,alpha=0.65)
        # sentinel field locations
        if bw is None: #color
            for poly in locinfo.field_polys.values():
                ax.add_patch(patches.PathPatch(poly,facecolor='none',
                             edgecolor='r',lw=2,zorder=2))
        else: #black and white
            for poly in locinfo.field_polys.values():
                ax.add_patch(patches.PathPatch(poly,facecolor='none',
                             edgecolor='k',lw=2,zorder=2))
        # axis labels
        ax.xlabel('West-East (meters)')
        ax.ylabel('North-South (meters)')
        #report the value at the origin
        oval = sol_fm.flat[sol_fm.size//2]
        oval = 0.0 if oval is np.ma.masked else oval
        ax.text(0.95,0.95,'Origin: {:.3}'.format(oval),color='w',
            ha='right',va='center',transform=ax.transAxes,fontsize=16)
        res = int(params.domain_info[0]//params.domain_info[1])
        ax.text(0.01,0.05,'{0}x{0} m resolution'.format(res),color='w',
            ha='left',va='center',transform=ax.transAxes,fontsize=12)
        cbar = ax.colorbar()
        cbar.solids.set_edgecolor("face")
        cbar.set_label('Wasps per cell')
        ## Finish plotting model maps ##
    
    # Lay out histograms relative to distance from release
    zcoord = [300,650,750,2000,2900,3500]
    
    ## Plot sentinel field data and model ##    
    for subplt in sp3d:
        ax = fig.add_subplot(sp3d,projection='3d')
        if subplt == sp3d[0]:
            # put the data on top
            for z in zcoord:
                ax.bar(range(collection_date),sent_ovi_array,zs=z,ydir='x',
                    alpha=0.8)
            ax.set_zlabel('Adult counts')
        else:
            # model densities
            for n,z in enumerate(zcoord):
                ax.bar(range(collection_date),model_field_densities[:,n],
                    zs=z,ydir='x',alpha=0.8)
            ax.set_zlabel('Modeled adult densities')
        ax.set_xlabel('Fields')
        ax.set_ylabel('Days PR')
        # re-tick x-axis
        ax.xticks([300,650,750,2000,2900,3500],('B','C','D','E','F','G'))
        
    plt.show()
        
        