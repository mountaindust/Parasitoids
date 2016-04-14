#! /usr/bin/env python3

'''This module uses PyMC to fit parameters to the model via Bayesian inference.

Author: Christopher Strickland  
Email: cstrickland@samsi.info 
'''

__author__ = "Christopher Strickland"
__email__ = "cstrickland@samsi.info"
__status__ = "Development"
__version__ = "0.1"
__copyright__ = "Copyright 2015, Christopher Strickland"

import sys, os, time, math
from io import StringIO
from multiprocessing import Pool
import numpy as np
from scipy import sparse
from matplotlib.path import Path
import pymc as pm
import globalvars
from Run import Params
import ParasitoidModel as PM
from CalcSol import get_solutions


location = 'Kalbar'

params = Params()
# Set parameters specific to Bayesian runs
params.PLOT = False
params.OUTPUT = False
params.ndays = -1 # locinfo.get_landscape_sample_datesPR()[-1]?

# This sends a message to CalcSol on whether or not to use CUDA
if params.CUDA:
    globalvars.cuda = True
else:
    globalvars.cuda = False
# get wind data and day labels
wind_data,days = PM.get_wind_data(*params.get_wind_params())



###############################################################################
#                                                                             #
#                              Data Import                                    #
#                                                                             #
###############################################################################

class LocInfo():
    '''Class def to hold all info on experimental location and resulting data'''
    
    def __init__(self,location,release_latlong,domain_info):
        '''
        Args: location: required string giving the location name.
                All data files must be stored in ./data with the proper 
                naming convention
              domain_info: Run.Params.domain_info
              release_latlong: lat/long coord of the release poin.'''
        
        # Import sentinal field locations
        self.field_polys = get_fields(location+'fields.txt',release_latlong)
        # Convert to cell indices
        self.field_cells = get_field_cells(self.field_polys,domain_info)
                
        # Import release field grid
        
        # Here import all emergence data
        

        pass
        
    def get_landscape_sample_datesPR(self):
        pass
        
    def get_landscape_emergence_data(self):
        '''2D array of dates and locations'''
        pass
        
    def get_sampling_effort(self):
        '''2D array of dates and locations'''
        pass



def get_fields(filename,center):
    '''This function reads in polygon data from a file which describs boundaries
    of the fields that make up each collection site. The polygon data is in the
    form of lists of vertices, the locations of which are given in x,y
    coordinates away from the release point. This function then returns a list
    of matplotlib Path objects which allow point testing for inclusion.
    
    Args:
        filename: file name to open
        center: lat/long coord of the release point'''
    
    def latlong_tocoord(center,lat,long):
        '''Translate a lat/long coordinate into an (x,y) coordinate pair where
        center is the origin.
        
        Args:
            center: subscritable lat/long location of the origin
            lat: latitude to translate
            long: longitude to translate
            
        Returns:
            (x,y): x,y coordinate from center, in meters'''
            
        R = 6378100 #Radius of the Earth in meters at equator
        
        o_lat = math.radians(center[0]) #origin lat in radians
        o_long = math.radians(center[1]) #origin long in radians
        lat = math.radians(lat)
        long = math.radians(long)
        
        # # Haversine formula
        # a = math.sin((lat-o_lat)/2)**2 + math.cos(lat)*math.cos(o_lat)*\
            # math.sin((long-o_long)/2)**2
        # c = 2*math.atan2(math.sqrt(a),math.sqrt(1-a))
        # dist = R*c
        
        # # Bearing
        # bear = math.atan2(math.sin(long-o_long)*math.cos(lat),
            # math.cos(o_lat)*math.sin(lat)-math.sin(o_lat)*math.cos(lat)*
            # math.cos(long-o_long))
        
        # # Approximation based on straight line from bearing
        # x = dist*math.cos(bear)
        # y = dist*math.sin(bear)
        
        # Equirectangular approximation
        x = R*(long-o_long)*math.cos((o_lat+lat)/2)
        y = R*(lat-o_lat)
        
        return (x,y)
    
    polys = []
    with open(filename,'r') as f:
        verts = []
        codes = []
        for line in f:
            if line.strip() == '':
                # end of current polygon. convert into Path object.
                verts.append((0.,0.)) # ignored
                codes.append(Path.CLOSEPOLY)
                polys.append(Path(verts,codes))
                verts = []
                codes = []
            else:
                # deal with possible comments
                c_ind = line.find('#')
                if c_ind >= 0:
                    line = line[:c_ind]
                # parse the data
                vals = line.split(',')
                verts.append(latlong_tocoord(
                            center,float(vals[0]),float(vals[1])))
                if len(codes) == 0:
                    codes.append(Path.MOVETO)
                else:
                    codes.append(Path.LINETO)
    # Convert the last polygon into a Path object if not already done
    if len(verts) != 0:
        verts.append((0.,0.)) # ignored
        codes.append(Path.CLOSEPOLY)
        polys.append(Path(verts,codes))
        
    return polys
    
 
    
def get_field_cells(polys,domain_info):
    '''Get a list of lists of cell indices that represent each field.
    
    Args:
        polys: List of Path objects representing each field
        domain_info: (dist (m), cells) from release point to side of domain
                        (as in the Run.Params class)
                        
    Returns:
        fields: list with each sublist representing a field of cells'''
    
    fields = []
    res = domain_info[0]/domain_info[1] #cell resolution
    # construct a list of all x,y coords (in meters) for the center of each cell
    centers = [(x,y) for x in range(domain_info[1],-domain_info[1]-1,-1)*res
                        for y in range(-domain_info[1],domain_info[1]+1)*res]
    for poly in polys:
        fields.append(np.argwhere(
            poly.contains_points(centers).reshape(
            domain_info[1]*2+1,domain_info[1]*2+1)))
                
    return fields




###############################################################################
#                                                                             #
#                              PYMC Setup                                     #
#                                                                             #
###############################################################################
locinfo = LocInfo(location,params.domain_info)
    
    
    
@pm.deterministic
def params_obj(params=params,g_aw=g_aw,g_bw=g_bw,f_a1=f_a1,f_b1=f_b1,
    f_a2=f_a2,f_b2=f_b2,sig_x=sig_x,sig_y=sig_y,corr=corr,lam=lam,mu_r=mu_r):
    '''Return altered parameter object to be passed in to simulation'''
    
    # g wind function parameters
    params.g_params = (g_aw,g_bw)
    # f time of day function parameters
    params.f_params = (f_a1,f_b1,f_a2,f_b2)
    # Diffusion coefficients
    params.Dparams = (sig_x,sig_y,corr)
    # Probability of any flight during the day under ideal circumstances
    params.lam = lam
    
    # TRY BOTH - VARYING mu_r OR n_periods
    # scaling flight advection to wind advection
    params.mu_r = mu_r
    # number of time periods (based on interp_num) in one flight
    #params.n_periods = n_periods # if interp_num = 30, this is # of minutes
    
    return params
    
    
    
@pm.deterministic
def run_model(params=params_obj,wind_data=wind_data,days=days):
    '''This function acts as an interface between PyMC and the model.
    It provides the model based on stochastic parameters in params.
    '''
    
    ### PHASE ONE ###
    # First, get spread probability for each day as a coo sparse matrix
    pmf_list = []
    max_shape = np.array([0,0])
    print("Calculating each day's spread in parallel...")
    pm_args = [(days[0],wind_data,*params.get_model_params(),
            params.r_start)]
    pm_args.extend([(day,wind_data,*params.get_model_params()) 
            for day in days[1:params.ndays]])
    pool = Pool()
    pmf_list = pool.starmap(PM.prob_mass,pm_args)
    pool.close()
    pool.join()
    for pmf in pmf_list:
        for dim in range(2):
            if pmf.shape[dim] > max_shape[dim]:
                max_shape[dim] = pmf.shape[dim]
                
    r_spread = [] # holds the one-day spread for each release day.
    
    
    # Reshape the prob. mass function of each release day into solution form
    for ii in range(params.r_dur):
        offset = params.domain_info[1] - pmf_list[ii].shape[0]//2
        dom_len = params.domain_info[1]*2 + 1
        r_spread.append(sparse.coo_matrix((pmf_list[ii].data, 
            (pmf_list[ii].row+offset,pmf_list[ii].col+offset)),
            shape=(dom_len,dom_len)).tocsr())
    
    ### PHASE TWO ###
    # Pass the probability list, pmf_list, and other info to convolution solver.
    #   This will return the finished population model.
    with Capturing() as output:
        modelsol = get_populations(r_spread,pmf_list,days,params.ndays,dom_len,
                    max_shape,params.r_dur,params.r_number,params.r_mthd())
    
    # modelsol now holds the model results for this run
    return modelsol
    


@pm.deterministic
def landscape_emergence(modelsol=run_model,beta=beta):
    '''Emergence observed from the model. This parses modelsol using the
    sampling model. It should return a 2D array where the rows are each day
    sampled and the columns are the number of E. Hayati observed.
    
    Args:
        modelsol: model solution
        beta: Bernoulli trial probability by density'''
    pass
    
    
    
class Capturing(list):
    '''This class creates a list object that can be used in 'with' environments
    to capture the stdout of the enclosing functions. If used multiple times,
    it can extend itself to make a longer list containing everything.
    
    Usage:
        with Capturing() as output:
            <code in which stdout is captured>
            
        # subsequent usage, to extend previous output list:
        with Capturing(output) as output:
            <more code with stdout captured>'''
            
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout


    
# if __name__ == "__main__":
    # main(sys.argv[1:])