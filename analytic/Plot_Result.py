#! /usr/bin/env python3
'''
Routines for plotting the results of the model 
in a resolution sensitive way

Author: Christopher Strickland'''

import sys, io
import warnings
import math
from collections import OrderedDict
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
from PIL import Image
import urllib.parse, urllib.request
import Run

base_clrmp = cm.get_cmap('viridis')
# alter this colormap with alpha values
dict_list = []
for x in zip(*base_clrmp.colors):
    dict_list.append(tuple((n/(len(x)-1),val,val) for n,val in enumerate(x)))
cdict = {'red': dict_list[0], 'green': dict_list[1], 'blue': dict_list[2]}
cdict['alpha'] = ((0.0,0.65,0.65),
                  (1.0,0.65,0.65))
plt.register_cmap(name='alpha_viridis',data=cdict)
clrmp = cm.get_cmap('alpha_viridis')
clrmp.set_bad('w',alpha=0)

def r_small_vals(A,negval):
    '''Remove negligible values from the given coo sparse matrix. 
    This process significantly decreases the size of a solution and gives an
    accurate plot domain.'''
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
    

    
def get_satellite(key,center,dist):
    '''Get Bing satellite image for plot area
    
    Args:
        key: Bing maps key
        center: lat/long coordinates of release point, tuple
        dist: distance from release to side of domain, in meters
        
    Returns:
        numpy array of image that can be seen with plt.imshow'''
        
    if key is None or center is None:
        return None
    
    lat,long = center
    
    # get zoom level so that we have between 400 and 800 pixels**2 resolution
    zoom = 4
    while not dist/400 < resfunc(lat,zoom) <= dist/200:
        zoom += 1
        
    # get the pixel dimensions to request
    pixel_len = int(round((dist*2+1)/resfunc(lat,zoom)))
    
    urlparams = urllib.parse.urlencode({
        'mapSize': '{0:d},{0:d}'.format(pixel_len),
        'format': 'jpeg',
        'key': key})
        
    url = 'http://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/'+\
        '{0:03.6f}%2C{1:03.6f}'.format(lat,long)+\
        '/{0:d}?'.format(zoom)+urlparams
    
    try:
        f = urllib.request.urlopen(url)
        im = Image.open(io.BytesIO(f.read()))
    except urllib.error.URLError as e:
        print('Could not retrieve arial image from url.')
        print(e.reason)
        print('Continuing without satellite imagery...')
        return None
        
    # convert to numpy array and return
    return np.array(im.getdata(),np.uint8).reshape(im.size[1],im.size[0],3)
    
    
    
def plot_all(modelsol,days,params,mask_val=0.00001):
    '''Function for plotting the model solution
    
    Args:
        modelsol: list of daily solutions, coo sparse
        days: list of day identifiers
        domain_info: rad_dist, rad_res
        mask_val: values less then this value will not appear in plotting'''
    
    domain_info = params.domain_info
    cell_dist = domain_info[0]/domain_info[1] #dist from one cell to 
                                              #neighbor cell (meters).
    
    # assume domain is square, probably odd.
    midpt = domain_info[1]
        
    plt.figure()
    for n,sol in enumerate(modelsol):
        #remove all values that are too small to be plotted.
        sol_red = r_small_vals(sol,mask_val)
        #find the maximum distance from the origin that will be plotted
        rmax = max(np.fabs(sol_red.row-midpt).max(),
            np.fabs(sol_red.col-midpt).max())
        #construct xmesh and a masked solution array based on this
        rmax = min(rmax+5,midpt) # add a bit of frame space
        xmesh = np.linspace(-rmax*cell_dist-cell_dist/2,
            rmax*cell_dist+cell_dist/2,rmax*2+2)
        sol_fm = np.flipud(np.ma.masked_less(
            sol_red.toarray()[midpt-rmax:midpt+rmax+1,midpt-rmax:midpt+rmax+1],
            mask_val))
        plot_limits = [xmesh[0],xmesh[-1],xmesh[0],xmesh[-1]]
        plt.clf()
        plt.axis(plot_limits)
        sat_img = get_satellite(params.maps_key,params.coord,xmesh[-1])
        if sat_img is None:
            plt.pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp,alpha=1)
        else:
            plt.imshow(sat_img,zorder=0,extent=plot_limits)
            plt.pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp,zorder=1)
        
        plt.xlabel('West-East (meters)')
        plt.ylabel('North-South (meters)')
        plt.title('Parasitoid spread {0} day(s) post release'.format(days[n]))
        plt.colorbar()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if n != len(modelsol)-1:
                plt.pause(3.5)
            else:
                plt.pause(0.0001)
                plt.show()

            
            
def plot(sol,day,params,mask_val=0.00001):
    '''Plot a solution for a single day
    
    Args:
        sol: day solution, coo sparse
        day: day identifier (for text identification)
        domain_info: rad_dist, rad_res
        mask_val: values less then this value will not appear in plotting'''

    domain_info = params.domain_info
    cell_dist = domain_info[0]/domain_info[1] #dist from one cell to 
                                              #neighbor cell (meters).
    
    # assume domain is square, probably odd.
    midpt = domain_info[1]

    plt.ion()
    plt.figure()
    #remove all values that are too small to be plotted.
    sol_red = r_small_vals(sol,mask_val)
    #find the maximum distance from the origin
    rmax = max(np.fabs(sol_red.row-midpt).max(),np.fabs(sol_red.col-midpt).max())
    #construct xmesh and a masked solution array based on this
    rmax = min(rmax+5,midpt) # add a bit of frame space
    xmesh = np.linspace(-rmax*cell_dist-cell_dist/2,
        rmax*cell_dist+cell_dist/2,rmax*2+2)
    sol_fm = np.flipud(np.ma.masked_less(
        sol_red.toarray()[midpt-rmax:midpt+rmax+1,midpt-rmax:midpt+rmax+1],
        mask_val))
    plot_limits = [xmesh[0],xmesh[-1],xmesh[0],xmesh[-1]]
    plt.axis(plot_limits)    
    sat_img = get_satellite(params.maps_key,params.coord,xmesh[-1])
    if sat_img is None:
        plt.pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp,zorder=1,alpha=1)
    else:
        plt.imshow(sat_img,zorder=0,extent=plot_limits)
        plt.pcolormesh(xmesh,xmesh,sol_fm,cmap=clrmp,zorder=1)
    plt.xlabel('West-East (meters)')
    plt.ylabel('North-South (meters)')
    plt.title('Parasitoid spread {0} day(s) post release'.format(day))
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
        filename = filename[:-5]
    elif filename.rstrip()[-4:] == '.npz':
        filename = filename[:-4]

    # load parameters
    params = Run.Params()
    params.file_read_chg(filename)

    dom_len = params.domain_info[1]*2 + 1

    # load data
    modelsol = OrderedDict() # we use dict here for convenience
    with np.load(filename+'.npz') as npz_obj:
        days = npz_obj['days']
        for day in days:
            V = npz_obj[str(day)+'_data']
            I = npz_obj[str(day)+'_row']
            J = npz_obj[str(day)+'_col']
            modelsol[str(day)] = sparse.coo_matrix((V,(I,J)),
                                                    shape=(dom_len,dom_len))

    while True:
        val = input('Enter day number to plot, '+
                    '? to see a list of plottable days, '+
                    'or "all" to plot all.'+
                    ' Enter q to quit:')
        val = val.strip()
        if val == '?':
            print(*days)
        elif val.lower() == 'q' or val.lower() == 'quit':
            break
        elif val.lower() == 'a' or val.lower() == 'all':
            # plot_all wants a list of values. pass a view into ordered dict
            plot_all(modelsol.values(),days,params)    
        else:
            try:
                plot(modelsol[val],val,params)
            except KeyError:
                print('Day {0} not found.'.format(val))
    
if __name__ == "__main__":
    main(sys.argv[1:])