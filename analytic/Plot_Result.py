#! /usr/bin/env python3
'''
Routines for plotting the results of the model 
in a resolution sensitive way

Author: Christopher Strickland'''

import sys, io
import warnings
import math
import urllib.parse, urllib.request
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.patches as patches
try:
    from PIL import Image
    NO_PILLOW = False
except:
    NO_PILLOW = True
import Run

PILLOW_MSG = False

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


def r_small_vals(A,negval):
    '''Remove negligible values from the given matrix. 
    This process significantly decreases the size of a solution and gives an
    accurate plot domain. Return a sparse coo matrix.'''
    if not sparse.isspmatrix_coo(A):
        A = sparse.coo_matrix(A)
    
    midpt = A.shape[0]//2 #assume domain is square
    
    mask = np.empty(A.data.shape,dtype=bool)
    for n,val in enumerate(A.data):
        if val < negval:
            mask[n] = False
        else:
            mask[n] = True
    return sparse.coo_matrix((A.data[mask],(A.row[mask],A.col[mask])),A.shape)
    

    
def latlong_trans(lat,lon,brng,dist):
    '''Translate the lat/long coordinates by baring and distance.
    
    Args:
        lat: Latitude
        lon: Longitude
        brng: Bearing in radians, clockwise from the north
        dist: distance in meters
        
    Returns:
        lat2: new latitude
        lon2: new longitude'''
        
    R = 6378100 #Radius of the Earth in meters at equator

    lat1 = math.radians(lat) #Current lat point converted to radians
    lon1 = math.radians(lon) #Current long point converted to radians

    lat2 = math.asin( math.sin(lat1)*math.cos(dist/R) +
         math.cos(lat1)*math.sin(dist/R)*math.cos(brng))

    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(dist/R)*math.cos(lat1),
                 math.cos(dist/R)-math.sin(lat1)*math.sin(lat2))

    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    return (lat2,lon2)
    
    
    
def resfunc(lat,zoom):
    '''Get the ground resolution in meters per pixel at a given latitude/zoom'''
    
    return (math.cos(lat*math.pi/180)*2*math.pi*6378137)/(256*2**zoom)
    

    
def get_satellite(key,service,center,dist):
    '''Get Bing satellite image for plot area
    
    Args:
        key: Bing maps key
        center: lat/long coordinates of release point, tuple
        dist: distance from release to side of domain, in meters
        
    Returns:
        numpy array of image that can be seen with plt.imshow'''
        
    global PILLOW_MSG
    if NO_PILLOW:
        if not PILLOW_MSG:
            print('Note: Python package "Pillow" not found. Continuing...')
            PILLOW_MSG = True
        return None
    
    if key is None or center is None or service is None:
        return None
    
    lat,long = center
    
    # get zoom level so that we are within the services specified resolution
    #   dist is only half the domain size!
    zoom = 4
    if service == 'Google':
        # for Google, we need to be between 320 and 640 pixels**2. This is for
        #   display area purposes - the returned figure is twice that resolution
        while not dist/320 < resfunc(lat,zoom) <= dist/160:
            zoom += 1
    else:
        # for Bing, we need to be between 400 and 800 pixels**2. This will
        #   be the actual resolution of the image we get.
        while not dist/400 < resfunc(lat,zoom) <= dist/200:
            zoom += 1
        
    # get the pixel dimensions to request
    pixel_len = int(round((dist*2+1)/resfunc(lat,zoom)))
    
    # collect parameters for maps API
    if service == 'Bing':
        urlparams = urllib.parse.urlencode({
            'mapSize': '{0:d},{0:d}'.format(pixel_len),
            'format': 'jpeg',
            'key': key})  
        url = 'http://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/'+\
            '{0:03.6f}%2C{1:03.6f}'.format(lat,long)+\
            '/{0:d}?'.format(zoom)+urlparams
    
    elif service == 'Google':
        urlparams = urllib.parse.urlencode({
            'center': '{0:03.6f},{1:03.6f}'.format(lat,long),
            'zoom': '{0:d}'.format(zoom),
            'size': '{0:d}x{0:d}'.format(pixel_len),
            'scale': '2', #twice as many pixels, same converage area
            'format': 'jpeg',
            'maptype': 'satellite',
            'key': key
            })
        url = 'https://maps.googleapis.com/maps/api/staticmap?'+urlparams
    
    else:
        print('Unknown maps service. Continuing without satellite imagery...')
        return None
    
    try:
        f = urllib.request.urlopen(url)
        im = Image.open(io.BytesIO(f.read()))
    except urllib.error.URLError as e:
        print('Could not retrieve arial image from url.')
        print(e.reason)
        print('Continuing without satellite imagery...')
        return None
        
    # matplotlib can plot a pillow Image object directly
    return im
    
    
    
def plot_all(modelsol,params,locinfo=None):
    '''Function for plotting the model solution
    
    Args:
        modelsol: list of daily solutions, sparse
        params: Params object from Run.py
        locinfo: if provided, is used to plot field polygons'''
    
    domain_info = params.domain_info
    cell_dist = domain_info[0]/domain_info[1] #dist from one cell to 
                                              #neighbor cell (meters).
    
    # assume domain is square, probably odd.
    midpt = domain_info[1]
        
    plt.figure()
    for n,sol in enumerate(modelsol):
        #Establish a miminum for plotting based 0.00001 of the maximum
        mask_val = min(10**(np.floor(np.log10(sol.data.max()))-3),1)
        #remove all values that are too small to be plotted.
        sol_red = r_small_vals(sol,mask_val)
        #find the maximum distance from the origin that will be plotted
        rmax = max(np.fabs(sol_red.row-midpt).max(),
            np.fabs(sol_red.col-midpt).max())
        #construct xmesh and a masked solution array based on this
        rmax = int(min(rmax+5,midpt)) # add a bit of frame space
        xmesh = np.linspace(-rmax*cell_dist-cell_dist/2,
            rmax*cell_dist+cell_dist/2,rmax*2+2)
        sol_fm = np.flipud(np.ma.masked_less(
            sol_red.toarray()[midpt-rmax:midpt+rmax+1,midpt-rmax:midpt+rmax+1],
            mask_val))
        plot_limits = [xmesh[0],xmesh[-1],xmesh[0],xmesh[-1]]
        plt.clf()
        ax = plt.axes()
        plt.axis(plot_limits)
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
            plt.pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp,vmax=sprd_max,alpha=1)
        else:
            plt.imshow(sat_img,zorder=0,extent=plot_limits)
            plt.pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp,vmax=sprd_max,zorder=1)
        if locinfo is not None:
            for poly in locinfo.field_polys.values():
                ax.add_patch(patches.PathPatch(poly,facecolor='none',
                             edgecolor='r',lw=2,zorder=2))
        
        plt.xlabel('West-East (meters)')
        plt.ylabel('North-South (meters)')
        plt.title('Parasitoid spread {0} day(s) post release'.format(n+1))
        oval = sol_fm.flat[sol_fm.size//2]
        oval = 0.0 if oval is np.ma.masked else oval
        plt.text(0.95,0.95,'Origin: {:.3}'.format(oval),color='w',
            ha='right',va='center',transform=ax.transAxes,fontsize=16)
        res = int(params.domain_info[0]//params.domain_info[1])
        plt.text(0.01,0.05,'{0}x{0} m cells'.format(res),color='w',
        ha='left',va='center',transform=ax.transAxes,fontsize=12)
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")
        cbar.set_label('Wasps per cell')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if n != len(modelsol)-1:
                plt.pause(0.85)
            else:
                plt.pause(0.0001)
                plt.show()

            
            
def plot(sol,day,params,saveonly=None,locinfo=None):
    '''Plot a solution for a single day
    
    Args:
        sol: day solution, sparse
        day: day identifier (for text identification)
        params: Params object from Run.py
        saveonly: (string) if not None, don't plot - save to location in saveonly
        locinfo: if provided, is used to plot field polygons
        '''

    bw = None
    if saveonly is not None:
        outname = saveonly+'_'+str(day)
        format = 'png'
        dpi = 300
        out_chg = input('Filename and/or .ext [{}]:'.format(outname+'.'+format))
        if out_chg != '':
            try:
                file, format = out_chg.strip().rsplit(sep='.',maxsplit=1)
                if file != '':
                    outname = file
            except ValueError:
                outname = out_chg.strip()
        dpi_chg = input('dpi (figsize={}) [{}]:'.format(
            mpl.rcParams['figure.figsize'],dpi))
        if dpi_chg != '':
            dpi = int(dpi_chg.strip())
        bw_chg = input('B/W? y/[n]:')
        if bw_chg.strip().lower() == 'y' or bw_chg.strip().lower() == 'yes':
            bw = True
        
    domain_info = params.domain_info
    cell_dist = domain_info[0]/domain_info[1] #dist from one cell to 
                                              #neighbor cell (meters).
    
    # assume domain is square, probably odd.
    midpt = domain_info[1]
    #Establish a miminum for plotting based 0.00001 of the maximum
    mask_val = min(10**(np.floor(np.log10(sol.data.max()))-3),1)
    if saveonly is None:
        plt.ion()
    else:
        plt.ioff()
    plt.figure()
    ax = plt.axes()
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
    plt.axis(plot_limits)
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
            plt.pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp,vmax=sprd_max,alpha=1)             
        else: #black and white
            plt.pcolormesh(xmesh,xmesh,sol_fm,cmap=plt.get_cmap('gray'),
                        vmax=sprd_max,alpha=1)
    else:
        if bw is None: #color
            plt.imshow(sat_img,zorder=0,extent=plot_limits)
            plt.pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp,vmax=sprd_max,zorder=1)
        else: #black and white
            sat_img = sat_img.convert('L') #B/W satellite image w/ dither
            plt.imshow(sat_img,zorder=0,cmap=plt.get_cmap('gray'),
                        extent=plot_limits)
            plt.pcolormesh(xmesh,xmesh,sol_fm,cmap=plt.get_cmap('gray'),
                        vmax=sprd_max,zorder=1,alpha=0.65)
    # sentinel field locations
    if locinfo is not None:
        if bw is None: #color
            for poly in locinfo.field_polys.values():
                ax.add_patch(patches.PathPatch(poly,facecolor='none',
                             edgecolor='r',lw=2,zorder=2))
        else: #black and white
            for poly in locinfo.field_polys.values():
                ax.add_patch(patches.PathPatch(poly,facecolor='none',
                             edgecolor='k',lw=2,zorder=2))      
    plt.xlabel('West-East (meters)')
    plt.ylabel('North-South (meters)')
    plt.title('Parasitoid spread {0} day(s) post release'.format(day))
    #report the value at the origin
    oval = sol_fm.flat[sol_fm.size//2]
    oval = 0.0 if oval is np.ma.masked else oval
    plt.text(0.95,0.95,'Origin: {:.3}'.format(oval),color='w',
        ha='right',va='center',transform=ax.transAxes,fontsize=16)
    res = int(params.domain_info[0]//params.domain_info[1])
    plt.text(0.01,0.05,'{0}x{0} m cells'.format(res),color='w',
        ha='left',va='center',transform=ax.transAxes,fontsize=12)
    cbar = plt.colorbar()
    cbar.solids.set_edgecolor("face")
    cbar.set_label('Wasps per cell')
    if saveonly is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.draw()
            plt.pause(0.0001)
    else:
        plt.savefig(outname+'.'+format,dpi=dpi,format=format)
        plt.close()
        print('...Figure saved to {}.'.format(outname+'.'+format))
        print('----------------Model result visualizations----------------')
        
        
        
def create_mp4(modelsol,params,filename,locinfo=None):
    '''Create and save an mp4 video of all the plots.
    The saved file name/location will be based on filename.
    
    Args:
        modelsol: list of daily solutions, sparse
        params: Params object from Run.py
        filename: location to save mp4
        locinfo: if provided, is used to plot field polygons'''
    
    print('Creating spread model video',end="")
    sys.stdout.flush()
    domain_info = params.domain_info
    cell_dist = domain_info[0]/domain_info[1] #dist from one cell to 
                                              #neighbor cell (meters).
    
    # assume domain is square, probably odd.
    midpt = domain_info[1]
    
    fig = plt.figure()
    ax = plt.axes()
    ax.axis([-400,400,-400,400])
    ax.set_xlabel('West-East (meters)')
    ax.set_ylabel('North-South (meters)')
    ax.set_title('Parasitoid spread')
    # try to get a satellite image to see if it will work
    sat_img = get_satellite(params.maps_key,params.maps_service,params.coord,500)
    if sat_img is None:
        pcl = ax.pcolormesh([],cmap=clrmp,zorder=1,alpha=1)
        SAT = False
    else:
        pcl = ax.pcolormesh([-400,0],[-400,0],[[0,0],[0,0]],cmap=clrmp,zorder=1)
        ax.imshow([[]],zorder=0)
        #plt.imshow(sat_img,zorder=0,extent=plot_limits)
        SAT = True
    cbar = plt.colorbar(pcl)
    cbar.set_label('Wasps per cell')
        
    def animate(nsol):
        n,sol = nsol
        #remove just the pcolormesh and satellite image from before
        for col in ax.collections:
            col.remove()
        #also remove the text from before
        ntexts = len(ax.texts)
        for ii in range(ntexts): 
            ax.texts[0].remove()
        #Establish a miminum for plotting based 0.00001 of the maximum
        mask_val = min(10**(np.floor(np.log10(sol.data.max()))-3),1)
        #remove all values that are too small to be plotted.
        sol_red = r_small_vals(sol,mask_val)
        #find the maximum distance from the origin
        rmax = max(np.fabs(sol_red.row-midpt).max(),
                   np.fabs(sol_red.col-midpt).max())
        #construct xmesh and a masked solution array based on this
        rmax = int(min(rmax+5,midpt)) # add a bit of frame space
        xmesh = np.linspace(-rmax*cell_dist-cell_dist/2,
            rmax*cell_dist+cell_dist/2,rmax*2+2)
        sol_fm = np.flipud(np.ma.masked_less(
            sol_red.toarray()[midpt-rmax:midpt+rmax+1,midpt-rmax:midpt+rmax+1],
            mask_val))
        #find the max value excluding the middle area
        midpt2 = sol_fm.shape[0]//2
        sol_mid = np.array(sol_fm[midpt2-4:midpt2+5,midpt2-4:midpt2+5])
        sol_fm[midpt2-4:midpt2+5,midpt2-4:midpt2+5] = np.ma.masked #mask the middle
        sprd_max = np.max(sol_fm) #find max
        sol_fm[midpt2-4:midpt2+5,midpt2-4:midpt2+5] = sol_mid #replace values
        plot_limits = [xmesh[0],xmesh[-1],xmesh[0],xmesh[-1]]
        ax.axis(plot_limits)
        ax.set_title('Parasitoid spread {0} day(s) post release'.format(n))
        if locinfo is not None:
            for poly in locinfo.field_polys.values():
                ax.add_patch(patches.PathPatch(poly,facecolor='none',
                             edgecolor='r',lw=2,zorder=2))
        if SAT:
            sat_img = get_satellite(params.maps_key,params.maps_service,
                params.coord,xmesh[-1])
            ax.imshow(sat_img,zorder=0,extent=plot_limits)
            pcl = ax.pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp,
                vmax=sprd_max,zorder=1)
        else:
            pcl = ax.pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp,vmax=sprd_max,
                zorder=1,alpha=1)
        oval = sol_fm.flat[sol_fm.size//2]
        oval = 0.0 if oval is np.ma.masked else oval
        ax.text(0.95,0.95,'Origin: {:.3}'.format(oval),color='w',
            ha='right',va='center',transform=ax.transAxes,fontsize=16)
        res = int(params.domain_info[0]//params.domain_info[1])
        plt.text(0.01,0.05,'{0}x{0} m cells'.format(res),color='w',
            ha='left',va='center',transform=ax.transAxes,fontsize=12)
        cbar.mappable = pcl
        cbar.update_bruteforce(pcl)
        cbar.solids.set_edgecolor("face")
        print('.',end="")
        sys.stdout.flush()
    
    # if we pass modelsol as is, the first and last frames won't appear...
    #   it seems that maybe they are there and gone so fast that they never
    #   appear. Let's not only duplicate them, put pause a little longer on them.

    def animGen():
        # Creates an animation generator
        for n,result in enumerate(modelsol):
            if n == 0 or n == len(modelsol)-1:
                # pause a bit on first and last result
                for ii in range(3):
                    yield (n+1,result)
            else:
                yield (n+1, result)

    # create animation
    framegen = animGen()
    anim = animation.FuncAnimation(fig,animate,frames=framegen,
            blit=False,interval=850)
    anim.save(filename+'.mp4',dpi=140,bitrate=500)
    print('\n...Video saved to {0}.'.format(filename+'.mp4'))
    
    
    
def main(argv):
    '''Function for plotting a previous result.
    
    The first argument in the list should be the location of a simulation file.
    '''
    try:
        filename = argv[0]
    except IndexError:
        print('Please pass the filename you wish to load as an argument, e.g.:')
        print('python Plot_Result.py output/filename')
        return
    if filename.rstrip()[-5:] == '.json':
        filename = filename[:-5]
    elif filename.rstrip()[-4:] == '.npz':
        filename = filename[:-4]

    # load parameters
    params = Run.Params()
    params.file_read_chg(filename)

    dom_len = params.domain_info[1]*2 + 1

    # load data
    modelsol = []
    with np.load(filename+'.npz') as npz_obj:
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

    LOCINFO_LOADED = False
    print('----------------Model result visualizations----------------')
    while True:
        val = input('Enter a day number to plot, '+
                'or "all" to plot all.\n'+
                '"save" or "s" and then a number will save that day to file.\n'+
                '? will provide a list of plottable day numbers.\n'+
                '"vid" will output a video (requires FFmpeg or menconder).\n'+
                '"fields" will load data for plotting sentinel field outlines.\n'+
                'Or enter q to quit:')
        val = val.strip()
        if val == '':
            continue
        elif val == '?':
            print(*list(range(1,len(days)+1)))
        elif val.lower() == 'q' or val.lower() == 'quit':
            break
        elif val.lower() == 'a' or val.lower() == 'all':
            # plot_all wants a list of values. pass a view into ordered dict
            if LOCINFO_LOADED:
                plot_all(modelsol,params,locinfo=locinfo)
            else:
                plot_all(modelsol,params)
        elif val.lower() == 'vid':
            if LOCINFO_LOADED:
                create_mp4(modelsol,params,filename,locinfo=locinfo)
            else:
                create_mp4(modelsol,params,filename)
        elif val.lower() == 'fields':
            try:
                from Data_Import import LocInfo
                locinfo = LocInfo(params.dataset,params.coord,params.domain_info)
                LOCINFO_LOADED = True
                print('Sentinel field locations loaded.\n')
            except:
                print('Could not load sentinel field data.')
                print(sys.exc_info()[0])
                continue
        elif val[0] == 's':
            try:
                if val[:4] == 'save':
                    val = int(val[4:].strip())
                else:
                    val = int(val[1:].strip())
            except ValueError:
                print('Could not convert {} to an integer.'.format(val))
                continue
            if LOCINFO_LOADED:
                plot(modelsol[val-1],val,params,saveonly=filename,locinfo=locinfo)
            else:
                plot(modelsol[val-1],val,params,saveonly=filename)
        else:
            try:
                if LOCINFO_LOADED:
                    plot(modelsol[int(val)-1],val,params,locinfo=locinfo)
                else:
                    plot(modelsol[int(val)-1],val,params)
            except KeyError:
                print('Day {0} not found.'.format(val))
            except ValueError:
                print('Input {} not understood.'.format(val))
                continue
    
if __name__ == "__main__":
    main(sys.argv[1:])