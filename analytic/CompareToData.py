#! /usr/bin/env python3
'''This module is for comparing model results to data and generating
publication quality figures.
'''

import argparse
import numpy as np
from scipy import sparse
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


##### Colormap stuff #####
qcmap = cm.get_cmap('Dark2') #qualitative colormap
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
    subplots = [231,234,232,235]
    sp3d = [233,236]
    labels = ['a)','b)','c)','d)','e)','f)']
    
    # assume domain is square, probably odd.
    midpt = params.domain_info[1]
    
    fig = plt.figure(figsize=(16,9),dpi=100)
    ## Plot model result maps ##
    for ii in range(4):
        sol = modelsol[plot_days[ii]]
        ax = fig.add_subplot(subplots[ii])
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
            if bw is False: #color
                pc = ax.pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp,vmin=mask_val,
                                   vmax=sprd_max,alpha=1)             
            else: #black and white
                pc = ax.pcolormesh(xmesh,xmesh,sol_fm,cmap=plt.get_cmap('gray'),
                                   vmin=mask_val,vmax=sprd_max,alpha=1)
        else:
            if bw is False: #color
                ax.imshow(sat_img,zorder=0,extent=plot_limits)
                pc = ax.pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp,vmin=mask_val,
                                   vmax=sprd_max,zorder=1)
            else: #black and white
                sat_img = sat_img.convert('L') #B/W satellite image w/ dither
                ax.imshow(sat_img,zorder=0,cmap=plt.get_cmap('gray'),
                            extent=plot_limits)
                pc = ax.pcolormesh(xmesh,xmesh,sol_fm,cmap=plt.get_cmap('gray'),
                                   vmin=mask_val,vmax=sprd_max,zorder=1,
                                   alpha=0.65)
        # sentinel field locations
        if bw is False: #color
            for poly in locinfo.field_polys.values():
                ax.add_patch(patches.PathPatch(poly,facecolor='none',
                             edgecolor='r',lw=2,zorder=2))
        else: #black and white
            for poly in locinfo.field_polys.values():
                ax.add_patch(patches.PathPatch(poly,facecolor='none',
                             edgecolor='k',lw=2,zorder=2))
        # axis labels
        ax.set_xlabel('West-East (meters)')
        ax.set_ylabel('North-South (meters)')
        # report the day PR
        ax.text(0.98,0.95,'{} days PR'.format(plot_days[ii]+2),color='w',
            ha='right',va='center',transform=ax.transAxes,fontsize=16)
        #report the value at the origin
        # oval = sol_fm.flat[sol_fm.size//2]
        # oval = 0.0 if oval is np.ma.masked else oval
        # ax.text(0.95,0.95,'Origin: {:.3}'.format(oval),color='w',
        #     ha='right',va='center',transform=ax.transAxes,fontsize=16)
        res = int(params.domain_info[0]//params.domain_info[1])
        ax.text(0.01,0.07,'{0}x{0} m cells'.format(res),color='w',
            ha='left',va='center',transform=ax.transAxes,fontsize=12)
        # colorbars
        left = [0.24,0.55]
        bottom = [0.61,0.12]
        l = left[int(ii/2)]
        b = bottom[ii%2]
        cax = fig.add_axes([l,b,0.05,0.02],zorder=10)
        cbar = fig.colorbar(pc,cax=cax,ticks=[mask_val,int(sprd_max)],
            orientation='horizontal')
        cbar.set_ticklabels([mask_val,'{}+'.format(int(sprd_max))])
        cbytick_obj = plt.getp(cbar.ax.axes, 'xticklabels')
        plt.setp(cbytick_obj, color='w')
        # cbar = fig.colorbar(pc,cax=ax)
        # cbar.solids.set_edgecolor("face")
        # cbar.set_label('Wasps per cell')
        
        # label plots
        ax.text(0.01,0.95,labels[ii],color='k',ha='left',va='center',
                transform=ax.transAxes,fontsize=16)
        ## Finish plotting model maps ##
    
    # Lay out histograms relative to distance from release
    zcoord = [0,300,550,850,2000,2900,3500]
    majorLocator = MultipleLocator(4)
    minorLocator = MultipleLocator(2)
    
    ## Plot sentinel field data and model ##
    emerg_dates = np.arange(collection_date,
                            collection_date+proj_emerg_densities.shape[1])
    for ii,subplt in enumerate(sp3d):
        ax = fig.add_subplot(subplt,projection='3d')
        if subplt == sp3d[0]:
            # put the data on the top plot
            color_list = np.linspace(0.95,0.05,len(zcoord)) # color setup
            for n,z in enumerate(zcoord):
                ax.bar(emerg_dates,obs_emerg_array[n,:],
                    zs=z,zdir='x',color=qcmap(color_list[n]),alpha=0.7)
            ax.set_zlabel('Emergence observations')
        else:
            # model densities
            for n,z in enumerate(zcoord):
                ax.bar(emerg_dates,proj_emerg_densities[n,:]*100,
                    zs=z,zdir='x',color=qcmap(color_list[n]),alpha=0.7)
            ax.set_zlabel('\n'+r'Projected emergence/100m$^2$')
        ax.set_xlabel('Fields')
        ax.set_ylabel('Days PR')
        # re-tick x-axis
        ax.set_xticks(zcoord)
        ax.set_xticklabels(allfields_ids)
        # re-tick y-axis
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_minor_locator(minorLocator)
        
        # label plots
        ax.text2D(0.01,0.95,labels[ii+4],color='k',ha='left',va='center',
                transform=ax.transAxes,fontsize=16)
        
        # adjust size of the 3D plots
        pos = list(ax.get_position().bounds) #[left,bottom,right,top]
        pos[2] = pos[2]*1.2
        ax.set_position(pos)
        
    plt.tight_layout()
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
    main(modelsol,params,locinfo,args.bw)