#! /usr/bin/env python3
'''This module is for comparing model results to data and generating
publication quality figures.
'''

import argparse
import numpy as np
from scipy import sparse, stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
import Run
from Plot_Result import get_satellite, r_small_vals
from Data_Import import LocInfo
import Bayes_funcs

##### argparse #####
parser = argparse.ArgumentParser()
# require a model output filename
parser.add_argument("filename",
    help="path to model output that should be compared to data")
parser.add_argument("--bw", help="plot in black/white", action="store_true")
parser.add_argument("-b","--banner",help="plot banner figure",
    action="store_true")
parser.add_argument("-a","--assess",help="assess model fit to grid obs",
    action="store_true")


##### Colormap stuff #####
qcmap = cm.get_cmap('Accent') #qualitative colormap
base_clrmp = cm.get_cmap('viridis') # nice colors
# alter this colormap with alpha values
dict_list = []
for x in zip(*base_clrmp.colors):
    dict_list.append(tuple((n/(len(x)-1),val,val) for n,val in enumerate(x)))
cdict = {'red': dict_list[0], 'green': dict_list[1], 'blue': dict_list[2]}
cdict['alpha'] = ((0.0,0.65,0.3),
                  (1.0,0.65,0.3))
plt.register_cmap(name='alpha_viridis',data=cdict)
clrmp = cm.get_cmap('alpha_viridis')
clrmp.set_bad('w',alpha=0) # colormap for showing parasitoid spread



def main(modelsol,params,locinfo,bw=False):
    '''Compare model results to data, as contained in locinfo
    TODO: This function should also spit out R**2 values for the model densities
    at grid points compared to observed adult counts on the three days this data
    was collected.

    Args:
        modelsol: list of daily solutions, sparse
        params: Params object from Run.py
        locinfo: LocInfo object from Data_Import.py
        bw: set this to something not None for b/w
        '''

    # Compare both release field and sentinel fields
    allfields_ids = [locinfo.releasefield_id]
    allfields_ids.extend(locinfo.sent_ids)

    # if multiple collections were made for emergence, just compare results for
    #   the first one.
    sent_col_num = 0
    grid_col_num = 0

    ##### Gather sentinal fields data #####
    dframe = locinfo.sent_DataFrames[0]
    collection_date = locinfo.collection_datesPR[0].days
    # get dates emergence was observed
    obs_dates_TD = dframe['datePR'].unique() # timedelta objects
    # ...and in integer form
    obs_datesPR = dframe['datePR'].map(lambda t: t.days).unique()

    # for release field, add up grid observations
    dframe_rel = locinfo.release_DataFrames[0]
    obs_rel_dates_TD = dframe_rel['datePR'].unique()
    obs_rel_datesPR = dframe_rel['datePR'].map(lambda t: t.days).unique()

    # set up emergence array - rows are fields, columns are emergence dates
    obs_emerg_array = np.zeros((len(allfields_ids),
                    max(obs_datesPR[-1],obs_rel_datesPR[-1])-collection_date+1))
    # first row is for release field
    for n,obs_date in enumerate(obs_rel_dates_TD):
        obs_emerg_array[0,obs_rel_datesPR[n]-collection_date] = \
            dframe_rel[dframe_rel['datePR'] == obs_date]['E_total'].sum()
    # the rest are for the sentinel fields
    for n,obs_date in enumerate(obs_dates_TD):
        obs_emerg_array[1:,obs_datesPR[n]-collection_date] = \
            dframe[dframe['datePR'] == obs_date]['E_total'].values
    # now obs_emerg_array can be directy compared to a projection of emergence
    #   in each field on the same day PR

    ##### Calculate the density of wasps in each field on each day #####
    # Get the size of each cell in m**2
    cell_dist = params.domain_info[0]/params.domain_info[1]
    cell_size = (cell_dist)**2
    field_sizes_m = {}
    for key,val in locinfo.field_sizes.items():
        field_sizes_m[key] = val*cell_size
    # Collect the number of wasps in each field on each day up to
    #   collection_date and calculate the wasps' density
    model_field_densities = np.zeros((len(allfields_ids),collection_date))
    for day in range(collection_date):
        for n,field_id in enumerate(allfields_ids):
            field_total = modelsol[day][locinfo.field_cells[field_id][:,0],
                                    locinfo.field_cells[field_id][:,1]].sum()
            model_field_densities[n,day] = field_total/field_sizes_m[field_id]
    # Now for each day, project forward to emergence using the function info
    #   specified in Bayes_funcs.
    proj_emerg_densities = np.zeros((len(allfields_ids),
                           collection_date+Bayes_funcs.max_incubation_time))
    max_incubation_time = Bayes_funcs.max_incubation_time
    min_incubation_time = max_incubation_time-len(Bayes_funcs.incubation_time)+1
    for day in range(collection_date):
        proj_emerg_densities[:,day+min_incubation_time:day+max_incubation_time+1]\
            += np.outer(model_field_densities[:,day],Bayes_funcs.incubation_time)
    # cut everything before the collection date
    proj_emerg_densities = proj_emerg_densities[:,collection_date:]
    # check that proj_emerg_densities.shape[1]>=obs_emerg_array.shape[1] and pad
    #   the latter one if necessary so that they are the same size.
    #   The idea here: we can project probability of emergence beyond what we
    #   happened to observe, but we should be projecting at least as far as obs
    assert proj_emerg_densities.shape[1] >= obs_emerg_array.shape[1], \
        'Emergence projection should cover the observations!'
    if proj_emerg_densities.shape[1] > obs_emerg_array.shape[1]:
        obs_emerg_array = np.pad(obs_emerg_array,
            ((0,0),(0,proj_emerg_densities.shape[1]-obs_emerg_array.shape[1])),
            'constant')


    ##### Plot comparison #####
    # The first two columns can be stills of the model at 3, 6, 9 and
    #   19 days PR, corresponding to data collection days. The last column
    #   should be two 3D histograms with the above data.
    plot_days = [1,4,7,17] # 0 = 2 day PR b/c model starts 1 day late and
                           #    counts by end-of-days
    locinfo_dates = []
    for date in locinfo.grid_obs_datesPR:
        locinfo_dates.append(date.days-1)
    assert plot_days[:3] == locinfo_dates, 'Incorrect plot days!'

    subplots = [231,234,232,235]
    sp3d = [233,236]
    labels = ['a)','b)','c)','d)','e)','f)']

    # assume domain is square, probably odd.
    midpt = params.domain_info[1]

    ax1 = []
    fig = plt.figure(figsize=(16,9),dpi=100)
    ## Plot model result maps ##
    for ii in range(4):
        sol = modelsol[plot_days[ii]]
        ax1.append(fig.add_subplot(subplots[ii]))
        #Establish a miminum for plotting based 0.00001 of the maximum
        mask_val = min(10**(np.floor(np.log10(sol.data.max()))-3),1)
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
        ax1[ii].axis(plot_limits)
        if xmesh[-1]>=6000:
            ax1[ii].set_xticks(np.arange(-6000,6001,3000),minor=False)
            ax1[ii].set_yticks(np.arange(-6000,6001,3000),minor=False)
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
            if bw is False: #color
                pc = ax1[ii].pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp,vmin=mask_val,
                                   vmax=sprd_max,alpha=1)
            else: #black and white
                pc = ax1[ii].pcolormesh(xmesh,xmesh,sol_fm,cmap=plt.get_cmap('gray'),
                                   vmin=mask_val,vmax=sprd_max,alpha=1)
        else:
            if bw is False: #color
                ax1[ii].imshow(sat_img,zorder=0,extent=plot_limits)
                pc = ax1[ii].pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp,vmin=mask_val,
                                   vmax=sprd_max,zorder=1)
            else: #black and white
                sat_img = sat_img.convert('L') #B/W satellite image w/ dither
                ax1[ii].imshow(sat_img,zorder=0,cmap=plt.get_cmap('gray'),
                            extent=plot_limits)
                pc = ax1[ii].pcolormesh(xmesh,xmesh,sol_fm,cmap=plt.get_cmap('gray'),
                                   vmin=mask_val,vmax=sprd_max,zorder=1,
                                   alpha=0.65)
        # sentinel field locations
        if bw is False: #color
            for poly in locinfo.field_polys.values():
                ax1[ii].add_patch(patches.PathPatch(poly,facecolor='none',
                             edgecolor='r',lw=2,zorder=2))
        else: #black and white
            for poly in locinfo.field_polys.values():
                ax1[ii].add_patch(patches.PathPatch(poly,facecolor='none',
                             edgecolor='k',lw=2,zorder=2))
        # axis labels
        ax1[ii].set_xlabel('West-East (meters)',fontsize=16)
        ax1[ii].set_ylabel('North-South (meters)',fontsize=16)
        # report the day PR
        ax1[ii].text(0.98,0.95,'{} days PR'.format(plot_days[ii]+2),color='w',
            ha='right',va='center',transform=ax1[ii].transAxes,fontsize=18)
        #report the value at the origin
        # oval = sol_fm.flat[sol_fm.size//2]
        # oval = 0.0 if oval is np.ma.masked else oval
        # ax1[ii].text(0.95,0.95,'Origin: {:.3}'.format(oval),color='w',
        #     ha='right',va='center',transform=ax1[ii].transAxes,fontsize=16)
        res = int(params.domain_info[0]//params.domain_info[1])
        ax1[ii].text(0.01,0.07,'{0}x{0} m cells'.format(res),color='w',
            ha='left',va='center',transform=ax1[ii].transAxes,fontsize=16)
        # colorbars
        left = [0.25,0.59]
        bottom = [0.61,0.115]
        l = left[int(ii/2)]
        b = bottom[ii%2]
        cax = fig.add_axes([l,b,0.05,0.02],zorder=10)
        cbar = fig.colorbar(pc,cax=cax,ticks=[mask_val,sprd_max],
            orientation='horizontal')
        if sprd_max < 10:
            cbar.set_ticklabels([mask_val,'{:.1f}+'.format(sprd_max)])
        else:
            cbar.set_ticklabels([mask_val,'{}+'.format(int(sprd_max))])
        cbytick_obj = plt.getp(cbar.ax.axes, 'xticklabels')
        plt.setp(cbytick_obj, color='w',fontsize=14)
        # cbar = fig.colorbar(pc,cax=ax)
        # cbar.solids.set_edgecolor("face")
        # cbar.set_label('Wasps per cell')

        # label plots
        ax1[ii].text(0.01,0.95,labels[ii],color='w',ha='left',va='center',
                transform=ax1[ii].transAxes,fontsize=18)
        # change tick label sizes
        ax1[ii].xaxis.set_tick_params(labelsize=14)
        ax1[ii].yaxis.set_tick_params(labelsize=14)
        ## Finish plotting model maps ##

    # Lay out histograms relative to distance from release
    zcoord = [0,300,550,850,2000,2900,3500]
    majorLocator = MultipleLocator(4)
    minorLocator = MultipleLocator(2)

    ## Plot sentinel field data and model ##
    emerg_dates = np.arange(collection_date,
                            collection_date+proj_emerg_densities.shape[1])
    ax2 = []
    for ii,subplt in enumerate(sp3d):
        ax2.append(fig.add_subplot(subplt,projection='3d'))
        if subplt == sp3d[0]:
            # put the data on the top plot
            color_list = np.linspace(0.95,0.05,len(zcoord)) # color setup
            for n,z in enumerate(zcoord):
                ax2[ii].bar(emerg_dates,obs_emerg_array[n,:],
                    zs=z,zdir='x',color=qcmap(color_list[n]),alpha=0.7)
            ax2[ii].set_zlabel('Emergence observations',fontsize=16)
        else:
            # model densities
            for n,z in enumerate(zcoord):
                ax2[ii].bar(emerg_dates,proj_emerg_densities[n,:]*100,
                    zs=z,zdir='x',color=qcmap(color_list[n]),alpha=0.7)
            ax2[ii].set_zlabel('\n'+r'Projected emergence/100m$^2$',fontsize=16)
        ax2[ii].set_ylim(emerg_dates[0],emerg_dates[-1])
        ax2[ii].set_xlabel('Fields',fontsize=16)
        ax2[ii].set_ylabel('Days PR',fontsize=16)
        # re-tick x-axis
        ax2[ii].set_xticks(zcoord)
        ax2[ii].set_xticklabels(allfields_ids,fontsize=14)
        # re-tick y-axis
        #ax.yaxis.set_major_locator(majorLocator)
        #ax.yaxis.set_minor_locator(minorLocator)
        # set label sizes
        ax2[ii].yaxis.set_tick_params(labelsize=14)
        ax2[ii].zaxis.set_tick_params(labelsize=14)

        # label plots
        ax2[ii].text2D(0.01,0.95,labels[ii+4],color='k',ha='left',va='center',
                transform=ax2[ii].transAxes,fontsize=18)

    plt.tight_layout(pad=0.25)
    # adjust size and position of the 3D plots
    for ii in range(len(sp3d)):
        pos = list(ax2[ii].get_position().bounds) #[left,bottom,width,height]
        pos[2] = pos[2]*0.9
        ax2[ii].set_position(pos)
    # shift over plots c) and d)
    for ii in range(2,4):
        pos1 = ax1[ii].get_position()
        pos2 = [pos1.x0+0.01, pos1.y0, pos1.width, pos1.height]
        ax1[ii].set_position(pos2)
    plt.show()



def assess_fit(modelsol,params,locinfo,bw=False):
    '''Compare model results to observation data, as contained in locinfo,
    and return plots and statistics assessing the model's fit to data

    Args:
        modelsol: list of daily solutions, sparse
        params: Params object from Run.py
        locinfo: LocInfo object from Data_Import.py
        bw: set this to something not None for b/w
    '''

    # Assess the model at data collection days.
    obs_days = []
    for date in locinfo.grid_obs_datesPR:
        obs_days.append(date.days-1)

    # Collect model pop numbers on the grid for each observation day
    grid_counts = Bayes_funcs.popdensity_grid(modelsol,locinfo)

    # Separate out model and data for different collection efforts
    # Assume that the same effort distribution was used each collection day
    efforts, counts = np.unique(locinfo.grid_samples[:,0],return_counts=True)
    data = []
    model = []
    for eff,cnt in zip(efforts,counts):
        this_data = np.zeros((cnt,len(obs_days)))
        this_model = np.zeros((cnt,len(obs_days)))
        n = 0
        for ii,this_effort in enumerate(locinfo.grid_samples[:,0]):
            if this_effort == eff:
                this_data[n,:] = locinfo.grid_obs[ii,:]
                this_model[n,:] = grid_counts[ii,:]
                n += 1
        data.append(this_data)
        model.append(this_model)

    #### Generate 3d plots showing the model density surface and the data ####
    # Pull out grid domain info for plotting
    xmax = np.fabs(locinfo.grid_data[['xcoord','ycoord']].values[:,0]).max()
    ymax = np.fabs(locinfo.grid_data[['xcoord','ycoord']].values[:,1]).max()
    # We assume this is more or less centered around the origin. Add padding.
    xmax *= 1.2
    ymax *= 1.2
    # Model resolution and center
    res = params.domain_info[0]/params.domain_info[1]
    center = params.domain_info[1]
    # Cell extents in x and y direction
    xcellrad = np.ceil(xmax/res)
    ycellrad = np.ceil(ymax/res)
    # Meshes
    xmesh = np.arange(0,xmax+res,res)
    xmesh = np.concatenate((-xmesh[:0:-1],xmesh))
    ymesh = np.arange(0,ymax+res,res)
    ymesh = np.concatenate((-ymesh[:0:-1],ymesh))
    ymesh = ymesh[::-1]
    xmeshgrid, ymeshgrid = np.meshgrid(xmesh,ymesh)
    ngridpoints = locinfo.grid_data.shape[0]

    # bar colors
    c_nums = np.linspace(0,1,locinfo.grid_obs_DataFrame['obs_count'].max()*5*2+2)
    c_nums = c_nums[1:-1]

    # surface default color
    default_cmap = cm.get_cmap('Oranges')
    default_clr = default_cmap(0.45)

    # find grid boundary cells
    bndry_cells = np.zeros_like(xmeshgrid)
    for x, y in locinfo.grid_boundary.T:
        ii = np.argmin(np.abs(ymesh - y))
        jj = np.argmin(np.abs(xmesh - x))
        bndry_cells[ii,jj] += 1

    # grid boundary color
    bndry_clr = default_cmap(0.2)

    # plot labels
    labels = ['a)','b)','c)']

    all_xcoord = locinfo.grid_data['xcoord'].values
    all_ycoord = locinfo.grid_data['ycoord'].values
    fig = plt.figure(figsize=(16,6),dpi=100)
    for day, date in enumerate(locinfo.grid_obs_datesPR):
        # get the non-zero observations on this day
        date_rows = locinfo.grid_obs_DataFrame['datePR'] == date

        ax = fig.add_subplot(1,len(obs_days),day+1,projection='3d')
        model_grid = modelsol[day][center-ycellrad:center+ycellrad+1,
                                   center-xcellrad:center+xcellrad+1].toarray()
        # the middle point (and possibly some adjacent points) will be far
        #   larger than other locations. clip these.
        clipval = 50
        for ii in range(model_grid.shape[0]):
            for jj in range(model_grid.shape[1]):
                if model_grid[ii,jj] > clipval:
                    model_grid[ii,jj] = clipval
        # Now scale so that the bar heights will be visible. Scale is currently
        #   wasps/25 m**2, make this wasps/10 m**2. This will set the clip to
        #   8 wasps/10 m**2 (50*10**2/25**2)
        model_grid /= 6.25

        # bars with no height
        ax.bar3d(all_xcoord, all_ycoord, np.zeros(ngridpoints), res, res, 0)
        ### bars with height ###
        xcoords = locinfo.grid_obs_DataFrame[date_rows]['xcoord'].values
        ycoords = locinfo.grid_obs_DataFrame[date_rows]['ycoord'].values
        scaling = np.zeros_like(xcoords)
        # get sampling effort for these coordinates
        n = 0
        for xcoord, ycoord in zip(xcoords,ycoords):
            scaling[n] = locinfo.grid_data[(locinfo.grid_data['xcoord']==xcoord)
                         & (locinfo.grid_data['ycoord']==ycoord)]['samples']
            if scaling[n] == 270:
                scaling[n] = 10/9
            else:
                scaling[n] = 10
            n += 1
        # color bars according to height
        clr_list = []
        for n,obs in enumerate(locinfo.grid_obs_DataFrame[date_rows]['obs_count']):
            clr_list.append(c_nums[int(obs*scaling[n]*2-1)])

        ax.bar3d(xcoords,ycoords,
                 np.zeros(locinfo.grid_obs_DataFrame[date_rows].shape[0]),
                 res, res,
                 locinfo.grid_obs_DataFrame[date_rows]['obs_count'].values*scaling*3/10,
                 color=base_clrmp(clr_list),label='Data')

        # color the facets like the bars, using a color not in viridis where
        #   there is no bar.
        facet_clrs = np.empty_like(xmeshgrid,dtype=object)
        for ii,x in enumerate(xmesh):
            for jj,y in enumerate(ymesh):
                xlow = x-res/2
                xhigh = x+res/2
                ylow = y-res/2
                yhigh = y+res/2
                for row in locinfo.grid_obs_DataFrame[date_rows].iterrows():
                    if xlow <= row[1]['xcoord'] < xhigh and\
                            ylow <= row[1]['ycoord'] < yhigh and\
                            facet_clrs[jj,ii] is None:
                        facet_clrs[jj,ii] = base_clrmp(c_nums[
                            row[1]['obs_count']-1])
                if facet_clrs[jj,ii] is None:
                    # check for grid boundary, color it different
                    if bndry_cells[jj,ii] > 0:
                        facet_clrs[jj,ii] = bndry_clr
                if facet_clrs[jj,ii] is None:
                    # assign default color
                    facet_clrs[jj,ii] = default_clr

        ax.plot_surface(xmeshgrid,ymeshgrid,model_grid,facecolors=facet_clrs,
                        rstride=1,cstride=1,alpha=0.35,shade=True,label='Model')
        ax.set_xlabel('West-East (meters)',fontsize=16)
        ax.set_ylabel('South-North (meters)',fontsize=16)
        ax.set_zlabel(r'num/10 m$^2$ model & observed',fontsize=16)
        # set view
        ax.view_init(24,-41)
        # add label
        ax.text2D(0.05,0.85,labels[day],color='k',ha='left',va='center',
                transform=ax.transAxes,fontsize=18)
        if day == 1:
            ax.set_title('Model vs.\n parasitoid observation data',fontsize=21)

    plt.tight_layout(pad=1.5)
    plt.show()



def banner(modelsol,params,locinfo,bw=False):
    '''Compare model results to data, as contained in locinfo, but give a
    simplified banner plot, e.g. for a research statement.

    Args:
        modelsol: list of daily solutions, sparse
        params: Params object from Run.py
        locinfo: LocInfo object from Data_Import.py
        bw: set this to something not None for b/w
    '''

    # Compare both release field and sentinel fields
    allfields_ids = [locinfo.releasefield_id]
    allfields_ids.extend(locinfo.sent_ids)

    # if multiple collections were made for emergence, just compare results for
    #   the first one.
    sent_col_num = 0
    grid_col_num = 0

    ##### Gather sentinal fields data #####
    dframe = locinfo.sent_DataFrames[0]
    collection_date = locinfo.collection_datesPR[0].days
    # get dates emergence was observed
    obs_dates_TD = dframe['datePR'].unique() # timedelta objects
    # ...and in integer form
    obs_datesPR = dframe['datePR'].map(lambda t: t.days).unique()

    # for release field, add up grid observations
    dframe_rel = locinfo.release_DataFrames[0]
    obs_rel_dates_TD = dframe_rel['datePR'].unique()
    obs_rel_datesPR = dframe_rel['datePR'].map(lambda t: t.days).unique()

    # set up emergence array - rows are fields, columns are emergence dates
    obs_emerg_array = np.zeros((len(allfields_ids),
                    max(obs_datesPR[-1],obs_rel_datesPR[-1])-collection_date+1))
    # first row is for release field
    for n,obs_date in enumerate(obs_rel_dates_TD):
        obs_emerg_array[0,obs_rel_datesPR[n]-collection_date] = \
            dframe_rel[dframe_rel['datePR'] == obs_date]['E_total'].sum()
    # the rest are for the sentinel fields
    for n,obs_date in enumerate(obs_dates_TD):
        obs_emerg_array[1:,obs_datesPR[n]-collection_date] = \
            dframe[dframe['datePR'] == obs_date]['E_total'].values
    # now obs_emerg_array can be directy compared to a projection of emergence
    #   in each field on the same day PR

    ##### Calculate the density of wasps in each field on each day #####
    # Get the size of each cell in m**2
    cell_dist = params.domain_info[0]/params.domain_info[1]
    cell_size = (cell_dist)**2
    field_sizes_m = {}
    for key,val in locinfo.field_sizes.items():
        field_sizes_m[key] = val*cell_size
    # Collect the number of wasps in each field on each day up to
    #   collection_date and calculate the wasps' density
    model_field_densities = np.zeros((len(allfields_ids),collection_date))
    for day in range(collection_date):
        for n,field_id in enumerate(allfields_ids):
            field_total = modelsol[day][locinfo.field_cells[field_id][:,0],
                                    locinfo.field_cells[field_id][:,1]].sum()
            model_field_densities[n,day] = field_total/field_sizes_m[field_id]
    # Now for each day, project forward to emergence using the function info
    #   specified in Bayes_funcs.
    proj_emerg_densities = np.zeros((len(allfields_ids),
                           collection_date+Bayes_funcs.max_incubation_time))
    max_incubation_time = Bayes_funcs.max_incubation_time
    min_incubation_time = max_incubation_time-len(Bayes_funcs.incubation_time)+1
    for day in range(collection_date):
        proj_emerg_densities[:,day+min_incubation_time:day+max_incubation_time+1]\
            += np.outer(model_field_densities[:,day],Bayes_funcs.incubation_time)
    # cut everything before the collection date
    proj_emerg_densities = proj_emerg_densities[:,collection_date:]
    # check that proj_emerg_densities.shape[1]>=obs_emerg_array.shape[1] and pad
    #   the latter one if necessary so that they are the same size.
    #   The idea here: we can project probability of emergence beyond what we
    #   happened to observe, but we should be projecting at least as far as obs
    assert proj_emerg_densities.shape[1] >= obs_emerg_array.shape[1], \
        'Emergence projection should cover the observations!'
    if proj_emerg_densities.shape[1] > obs_emerg_array.shape[1]:
        obs_emerg_array = np.pad(obs_emerg_array,
            ((0,0),(0,proj_emerg_densities.shape[1]-obs_emerg_array.shape[1])),
            'constant')


    ##### Plot comparison #####
    # The first two columns can be stills of the model at 3, 6, 9 and
    #   19 days PR, corresponding to data collection days. The last column
    #   should be two 3D histograms with the above data.
    plot_days = [1,4,7] # 0 = 2 day PR b/c model starts 1 day late and
                           #    counts by end-of-days
    subplots = [141,142,143]
    sp3d = [144]
    labels = ['a)','b)','c)','d)']

    # assume domain is square, probably odd.
    midpt = params.domain_info[1]

    ax1 = []
    fig = plt.figure(figsize=(18,4.5),dpi=100)
    ## Plot model result maps ##
    for ii in range(3):
        sol = modelsol[plot_days[ii]]
        ax1.append(fig.add_subplot(subplots[ii]))
        #Establish a miminum for plotting based 0.00001 of the maximum
        mask_val = min(10**(np.floor(np.log10(sol.data.max()))-3),1)
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
        ax1[ii].axis(plot_limits)
        if xmesh[-1]>=6000:
            ax1[ii].set_xticks(np.arange(-6000,6001,3000),minor=False)
            ax1[ii].set_yticks(np.arange(-6000,6001,3000),minor=False)
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
            if bw is False: #color
                pc = ax1[ii].pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp,vmin=mask_val,
                                   vmax=sprd_max,alpha=1)
            else: #black and white
                pc = ax1[ii].pcolormesh(xmesh,xmesh,sol_fm,cmap=plt.get_cmap('gray'),
                                   vmin=mask_val,vmax=sprd_max,alpha=1)
        else:
            if bw is False: #color
                ax1[ii].imshow(sat_img,zorder=0,extent=plot_limits)
                pc = ax1[ii].pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp,vmin=mask_val,
                                   vmax=sprd_max,zorder=1)
            else: #black and white
                sat_img = sat_img.convert('L') #B/W satellite image w/ dither
                ax1[ii].imshow(sat_img,zorder=0,cmap=plt.get_cmap('gray'),
                            extent=plot_limits)
                pc = ax1[ii].pcolormesh(xmesh,xmesh,sol_fm,cmap=plt.get_cmap('gray'),
                                   vmin=mask_val,vmax=sprd_max,zorder=1,
                                   alpha=0.65)
        # sentinel field locations
        if bw is False: #color
            for poly in locinfo.field_polys.values():
                ax1[ii].add_patch(patches.PathPatch(poly,facecolor='none',
                             edgecolor='r',lw=2,zorder=2))
        else: #black and white
            for poly in locinfo.field_polys.values():
                ax1[ii].add_patch(patches.PathPatch(poly,facecolor='none',
                             edgecolor='k',lw=2,zorder=2))
        # axis labels
        ax1[ii].set_xlabel('West-East (meters)',fontsize=16)
        ax1[ii].set_ylabel('North-South (meters)',fontsize=16)
        # report the day PR
        ax1[ii].text(0.98,0.95,'{} days PR'.format(plot_days[ii]+2),color='w',
            ha='right',va='center',transform=ax1[ii].transAxes,fontsize=18)
        #report the value at the origin
        # oval = sol_fm.flat[sol_fm.size//2]
        # oval = 0.0 if oval is np.ma.masked else oval
        # ax1[ii].text(0.95,0.95,'Origin: {:.3}'.format(oval),color='w',
        #     ha='right',va='center',transform=ax1[ii].transAxes,fontsize=16)
        res = int(params.domain_info[0]//params.domain_info[1])
        ax1[ii].text(0.01,0.07,'{0}x{0} m cells'.format(res),color='w',
            ha='left',va='center',transform=ax1[ii].transAxes,fontsize=16)
        # colorbars
        left = [0.182,0.441,0.7]
        bottom = 0.23
        l = left[ii]
        b = bottom
        cax = fig.add_axes([l,b,0.05,0.025],zorder=10)
        cbar = fig.colorbar(pc,cax=cax,ticks=[mask_val,sprd_max],
            orientation='horizontal')
        if sprd_max < 10:
            cbar.set_ticklabels([mask_val,'{:.1f}+'.format(sprd_max)])
        else:
            cbar.set_ticklabels([mask_val,'{}+'.format(int(sprd_max))])
        cbytick_obj = plt.getp(cbar.ax.axes, 'xticklabels')
        plt.setp(cbytick_obj, color='w',fontsize=14)
        # cbar = fig.colorbar(pc,cax=ax)
        # cbar.solids.set_edgecolor("face")
        # cbar.set_label('Wasps per cell')

        # label plots
        ax1[ii].text(0.01,0.95,labels[ii],color='w',ha='left',va='center',
                transform=ax1[ii].transAxes,fontsize=18)
        # change tick label sizes
        ax1[ii].xaxis.set_tick_params(labelsize=14)
        ax1[ii].yaxis.set_tick_params(labelsize=14)
        ## Finish plotting model maps ##

    # Lay out histograms relative to distance from release
    zcoord = [0,300,550,850,2000,2900,3500]
    majorLocator = MultipleLocator(4)
    minorLocator = MultipleLocator(2)

    ## Plot sentinel field data and model ##
    emerg_dates = np.arange(collection_date,
                            collection_date+proj_emerg_densities.shape[1])
    ax2 = []
    for ii,subplt in enumerate(sp3d):
        ax2.append(fig.add_subplot(subplt,projection='3d'))
        color_list = np.linspace(0.95,0.05,len(zcoord)) # color setup
        # model densities
        for n,z in enumerate(zcoord):
            ax2[ii].bar(emerg_dates,proj_emerg_densities[n,:]*100,
                zs=z,zdir='x',color=qcmap(color_list[n]),alpha=0.7)
        ax2[ii].set_zlabel('\n'+r'Projected emergence/100m$^2$',fontsize=16)
        ax2[ii].set_ylim(emerg_dates[0],emerg_dates[-1])
        ax2[ii].set_xlabel('Fields',fontsize=16)
        ax2[ii].set_ylabel('Days PR',fontsize=16)
        # re-tick x-axis
        ax2[ii].set_xticks(zcoord)
        ax2[ii].set_xticklabels(allfields_ids,fontsize=14)
        # re-tick y-axis
        #ax.yaxis.set_major_locator(majorLocator)
        #ax.yaxis.set_minor_locator(minorLocator)
        # set label sizes
        ax2[ii].yaxis.set_tick_params(labelsize=14)
        ax2[ii].zaxis.set_tick_params(labelsize=14)

        # label plots
        ax2[ii].text2D(0.01,0.95,labels[ii+3],color='k',ha='left',va='center',
                transform=ax2[ii].transAxes,fontsize=18)

    plt.tight_layout(pad=0.3)
    # adjust size and position of the 3D plots
    for ii in range(len(sp3d)):
        pos = list(ax2[ii].get_position().bounds) #[left,bottom,width,height]
        pos[0] = pos[0] - 0.019
        pos[1] = pos[1] + 0.085
        pos[3] = pos[3]*0.817
        ax2[ii].set_position(pos)
    # shift over plots b) and c)
    for ii in range(1,3):
        pos1 = ax1[ii].get_position()
        pos2 = [pos1.x0+0.008+(ii-1)*0.008, pos1.y0, pos1.width, pos1.height]
        ax1[ii].set_position(pos2)
    plt.show()



if __name__ == "__main__":
    args = parser.parse_args()
    if args.filename.rstrip()[-5:] == '.json':
        args.filename = args.filename[:-5]
    elif args.filename.rstrip()[-4:] == '.npz':
        args.filename = args.filename[:-4]

    # load parameters
    params = Run.Params()
    params.file_read_chg(args.filename)
    dom_len = params.domain_info[1]*2 + 1

    # load model result
    modelsol = []
    with np.load(args.filename+'.npz') as npz_obj:
        days = npz_obj['days']
        # some code here to make loading robust to both COO and CSR.
        CSR = False
        for day in days:
            V = npz_obj[str(day)+'_data']
            if CSR:
                indices = npz_obj[str(day)+'_ind']
                indptr = npz_obj[str(day)+'_indptr']
                modelsol.append(sparse.csr_matrix((V,indices,indptr),
                                                    shape=(dom_len,dom_len)))
            else:
                try:
                    I = npz_obj[str(day)+'_row']
                    J = npz_obj[str(day)+'_col']
                    modelsol.append(sparse.coo_matrix((V,(I,J)),
                                                    shape=(dom_len,dom_len)))
                except KeyError:
                    CSR = True
                    indices = npz_obj[str(day)+'_ind']
                    indptr = npz_obj[str(day)+'_indptr']
                    modelsol.append(sparse.csr_matrix((V,indices,indptr),
                                                    shape=(dom_len,dom_len)))

    # load data
    try:
        locinfo = LocInfo(params.dataset,params.coord,params.domain_info)
    except:
        print('Could not load the datasets for this location.')
        print(params.dataset)
        raise

    # call main
    if args.banner:
        banner(modelsol,params,locinfo,args.bw)
    elif args.assess:
        assess_fit(modelsol,params,locinfo,args.bw)
    else:
        main(modelsol,params,locinfo,args.bw)