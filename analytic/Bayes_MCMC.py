#! /usr/bin/env python3

'''This module uses PyMC to fit parameters to the model via Bayesian inference.

Author: Christopher Strickland  
Email: cstrickland@samsi.info 
'''

__author__ = "Christopher Strickland"
__email__ = "cstrickland@samsi.info"
__status__ = "Development"
__version__ = "0.0"
__copyright__ = "Copyright 2015, Christopher Strickland"

import sys, os, time, math
from io import StringIO
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy import sparse
from matplotlib.path import Path
import pymc as pm
import globalvars
from Run import Params
import ParasitoidModel as PM
from CalcSol import get_populations



###############################################################################
#                                                                             #
#                              Data Import                                    #
#                                                                             #
###############################################################################

class LocInfo():
    '''Class def to hold all info on experimental location and resulting data'''
    
    def __init__(self,location,release_latlong,domain_info):
        '''
        Args: 
            location: required string giving the location name.
                        All data files must be stored in ./data with the proper 
                        naming convention (see below)
            release_latlong: lat/long coord of the release point
            domain_info: Run.Params.domain_info
        '''
        
        ##### Import sentinal field locations from text file #####
        # Field labels should match those used in the emergence data 
        self.field_polys = get_fields('./data/'+location+'fields.txt',
            release_latlong) # this is a dict. keys are field labels
        # Convert to dict of cell indices
        self.field_cells = get_field_cells(self.field_polys,domain_info)
        # Get the number of cells in each sentinal field
        self.field_sizes = {}
        for key,val in self.field_cells.items():
            self.field_sizes[key] = max(val.shape)
                
        ##### Import and parse full release field grid information #####
        #   What this looks like depends on your data. See implementation of
        #       self.get_release_grid for details.
        #   grid_data contains the following columns:
        #       xcoord: distance east from release point in meters
        #       ycoord: distance north from release point in meters
        #       samples: sampling effort at each point via direct observation
        #       collection: collection effort at each point (for emergence)
        self.grid_data = self.get_release_grid('./data/'+location+'releasegrid.txt')
        # Get row/column indices from xcoord and ycoord in grid_data
        res = domain_info[0]/domain_info[1] # cell length in meters
        self.grid_cells = np.array([-self.grid_data['ycoord'].values,
                                    self.grid_data['xcoord'].values])
        self.grid_cells = (np.around(self.grid_cells/res) + 
                            domain_info[1]).astype(int)
        # self.grid_cells is now transposed: row 0 = row index, row 1 = col index
        
        ##### Import and parse sentinel field emergence data #####
        #   What this section looks like will be highly dependent on the 
        #       particulars of your dataset - every dataset is different.
        #       Fire up pandas in an ipython notebook and play with your data
        #       until you have what you need. Then put that procedure in the
        #       following method. What the method needs to accomplish is detailed
        #       in its docstring.
        #   Initializes:
        #       self.release_date
        #       self.collection_dates
        #       self.sent_DataFrames
        self.get_sentinel_emergence(location)
        # Sort and get ordered list(s) of sentinel field ids
        self.sent_ids = []
        for dframe in self.sent_DataFrames:
            dframe.sort_values(['datePR','id'],inplace=True)
            self.sent_ids.append(dframe['id'].unique())

        ##### Import and parse release field emergence data #####
        #   Again, highly dependent on what your dataset looks like. See method
        #       docstring for details.
        #   Initializes:
        #       self.releasefield_id
        #       self.release_DataFrames
        self.get_releasefield_emergence(location)
        # Further parsing
        self.emerg_grids = []
        for dframe in self.release_DataFrames:
            # Get row/column indices from xcoord and ycoord
            dframe['row'] = ((-dframe['ycoord']/res).round(0) +
                domain_info[1]).astype(int)
            dframe['column'] = ((dframe['xcoord']/res).round(0) +
                domain_info[1]).astype(int)
            # Sort the dataframes so that as one loops over the days, the row/col
            #   info will occur in the same order.
            dframe.sort_values(['datePR','row','column'],inplace=True)
            # Get the grid points that were collected
            oneday = dframe['datePR'] == dframe['datePR'].min()
            self.emerg_grids.append(zip(dframe['row'][oneday].values,
                                        dframe['column'][oneday].values))
                                        
        ##### Gather data in a form that can be quickly compared to the #####
        #####   output of popdensity_to_emergence                       #####
        # Want two lists: release_emerg and sentinel_emerg
        # Rows are locations, columns are days
        self.release_emerg = []
        self.release_collection = []
        #self.release_emerg_total = [] #all emergence observations summed
        self.sentinel_emerg = []
        #self.sentinel_emerg_total = []
        for dframe in self.release_DataFrames:
            obs_datesPR = dframe['datePR'].unique()
            datelen = len(dframe['row'][dframe['datePR'] == dframe['datePR'].min()].values)
            #Get collection effort at each unique grid point
            r_array = []
            for x,y in dframe.loc[dframe['datePR'] == dframe['datePR'].min(),\
                ['xcoord','ycoord']].values:
                r_array.append(self.grid_data[(self.grid_data['xcoord'] == x) & \
                    (self.grid_data['ycoord'] == y)]['collection'].values[0])
            r_array = np.array(r_array)
            r_array = r_array/r_array.max()
            self.release_collection.append(r_array)
            #Collect E. Hayati emergence into an ndarray
            E_array = np.zeros((datelen,len(obs_datesPR)))
            #All_array = np.zeros((datelen,len(obs_datesPR)))
            for ndate,date in enumerate(obs_datesPR):
                E_array[:,ndate] = dframe[dframe['datePR'] == date]['E_total'].values
                #All_array[:,ndate] = dframe[dframe['datePR'] == date]['All_total'].values
            self.release_emerg.append(E_array)
            #self.release_emerg_total.append(All_array)
        for ndframe,dframe in enumerate(self.sent_DataFrames):
            obs_datesPR = dframe['datePR'].unique()
            E_array = np.zeros((len(self.sent_ids[ndframe]),len(obs_datesPR)))
            #All_array = np.zeros((len(self.sent_ids[ndframe]),len(obs_datesPR)))
            for ndate,date in enumerate(obs_datesPR):
                E_array[:,ndate] = dframe[dframe['datePR'] == date]['E_total'].values
                #All_array[:,ndate] = dframe[dframe['datePR'] == date]['All_total'].values
            self.sentinel_emerg.append(E_array)
            #self.sentinel_emerg_total.append(All_array)
        
            
                
    def get_release_grid(self,filename):
        '''Read in data on the release field's grid and sampling effort.
        This will need to be edited depending on what your data looks like.
        Data is expected to contain info about the data collection points in the
            release field. Something in the other loaded columns needs to give
            an indication of the sampling (direct observation) effort, to be 
            parsed and stored in self.grid_samples, and the collection 
            (for later emergence) effort, to be parsed and stored in 
            self.grid_collection.
            
        Returns:
            DataFrame with the following columns:
                xcoord: distance east from release point in meters
                ycoord: distance north from release point in meters
                samples: sampling effort at each point via direct observation
                collection: collection effort at each point (for emergence)
        '''
        
        grid_data = []
        with open(filename,'r') as f:
            for line in f:
                #deal with possible comments
                c_ind = line.find('#')
                if c_ind >= 0:
                    line = line[:c_ind]
                if line.strip() != '':
                    #non-empty line. parse data
                    dat_list = line.split(',')
                    line_data = []
                    for dat in dat_list:
                        line_data.append(float(dat))
                    grid_data.append(line_data)
        # try to convert to numpy array
        grid_data = np.array(grid_data)
        # if no data is missing, this will have dim=2
        assert len(grid_data.shape) == 2, 'Could not convert data into 2D array.\n'+\
            'Likely, a line in {} is incomplete.'.format(filename)
            
        # column 2 and 3 are redundant for our purposes
        grid_data = np.delete(grid_data,2,axis=1)
        
        # Convert to pandas dataframe
        grid_info = pd.DataFrame(grid_data,
                            columns=['xcoord','ycoord','samples','collection'])
        return grid_info
        
    def get_sentinel_emergence(self,location):
        '''Get data relating to sentinel field emergence observations.
        This implementation will need to change completely according to the
            structure of your dataset. Parsing routines for multiple locations'
            data can be stored here - just extend the if-then clause based on
            the value of the location argument.
        WHAT IS REQURIED:
            self.release_date: pandas Timestamp of release date (no time of day)
            self.collection_dates: a list of Timestamp collection dates
            self.sent_DataFrames: a list of pandas DataFrames, one for each
                                    collection date.
        EACH DATAFRAME MUST INCLUDE THE FOLLOWING COLUMNS:
            id: Field indentifier that matches the keys in self.field_cells
            datePR: Num of days the emergence occured post-release (dtype=Timedelta)
            E_total: Total number of wasp emergences in that field on that date
            All_total: Total number of all emergences in that field on that date
                        (this will later be summed to obtain emergences per
                         field/collection)
        '''
        
        if location == 'kalbar':
            # location of data excel file
            data_loc = 'data/sampling_details.xlsx'
            # date of release (as a pandas TimeStamp, year-month-day)
            #   (leave off time of release)
            self.release_date = pd.Timestamp('2005-03-15')
            # dates of collection as a list of TimeStamps
            self.collection_dates = [pd.Timestamp('2005-03-31')]
            # list of sentinel field collection dates (as pandas TimeStamps)
            #self.sent_collection_dates = [pd.Timestamp('2005-05-31')]
            # initialize list of sentinel emergence DataFrames
            self.sent_DataFrames = []
            
            ### Pandas
            # load the sentinel fields sheet
            sentinel_fields_data = pd.read_excel(
                                    data_loc,sheetname='Kal-sentinels-raw')
            # rename the headings with spaces in them
            sentinel_fields_data.rename(
                    columns={"Field descrip":"descrip","date emerged":"date", 
                            "Field ID (jpgs)": "id",
                            "Field ID (paper)":"paperid"}, inplace=True)
            sentinel_fields_data.drop('descrip',1,inplace=True)
            sentinel_fields_data.drop('paperid',1,inplace=True)
            sentinel_fields_data.sort_values(['id','date'], inplace=True)
            # get sum of all the emergences
            col_list = list(sentinel_fields_data)
            for name in ['id','date']:
                col_list.remove(name)
            sentinel_fields_data['All_total'] = \
                        sentinel_fields_data[col_list].sum(axis=1)
            # get the number of E Hayati emergences per day
            sentinel_fields_data['E_total'] = \
                        sentinel_fields_data[['Efemales','Emales']].sum(axis=1)
            # get the dates post-release
            sentinel_fields_data['datePR'] = \
                        sentinel_fields_data['date'] - self.release_date
                        
            ### Store DataFrame in list
            self.sent_DataFrames.append(sentinel_fields_data)
            
        else:
            raise NotImplementedError
        
    def get_releasefield_emergence(self,location):
        '''Get data relating to release field emergence observations.
        This implementation will need to change completely according to the
            structure of your dataset. Parsing routines for multiple locations'
            data can be stored here - just extend the if-then clause based on
            the value of the location argument.
        WHAT IS REQURIED:
            self.releasefield_id: identifier matching the release field key
                                    in the dict self.field_cells
            self.release_DataFrames: a list of pandas DataFrames, one for each
                                    collection date.
            (Note: we assume the collection dates are the same as for the
                sentinel fields)
        EACH DATAFRAME MUST INCLUDE THE FOLLOWING COLUMNS:
            xcoord: distance east from release point in meters (grid collection point)
            ycoord: distance north from release point in meters (grid collection point)
            datePR: Num of days the emergence occured post-release (dtype=Timedelta)
            E_total: Total number of wasp emergences in that field on that date
            All_total: Total number of all emergences in that field on that date
                        (this will later be summed to obtain emergences per
                         field/collection)
        '''
        
        if location == 'kalbar':
            # location of data excel file
            data_loc = 'data/sampling_details.xlsx'
            # release field id
            self.releasefield_id = 'A'
            # initialize list of sentinel emergence DataFrames
            self.release_DataFrames = []
            
            ### Pandas
            # load the sentinel fields sheet
            release_field_data = pd.read_excel(
                                    data_loc,sheetname='Kal-releasefield-raw')
            # in our data, North was on the left of the grid. So switch coordinates
            release_field_data['temp'] = release_field_data['xcoord']
            release_field_data['xcoord'] = release_field_data['ycoord']
            # need to flip orientation
            release_field_data['ycoord'] = -release_field_data['temp']
            release_field_data.drop('temp',1,inplace=True)
            # put release point at the origin
            release_field_data['ycoord'] += 300
            release_field_data['xcoord'] -= 200
            col_list = list(release_field_data)
            for name in ['Field','xcoord','ycoord','date emerged']:
                col_list.remove(name)
            release_field_data['All_total'] = \
                release_field_data[col_list].sum(axis=1)
            release_field_data['E_total'] = \
                release_field_data[['Efemales','Emales']].sum(axis=1)
            release_field_data['datePR'] = \
                release_field_data['date emerged'] - self.release_date
                
            ### Store DataFrame in list
            self.release_DataFrames.append(release_field_data)
            
        else:
            raise NotImplementedError
    
    
    
def get_fields(filename,center):
    '''This function reads in polygon data from a file which describs boundaries
    of the fields that make up each collection site. The polygon data is in the
    form of lists of vertices, the locations of which are given in x,y
    coordinates away from the release point. This function then returns a dict
    of matplotlib Path objects which allow point testing for inclusion.
    
    Args:
        filename: file name to open
        center: lat/long coord of the release point
        
    Returns:
        polys: dict of Path objects'''
    
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
        
        # Equirectangular approximation
        x = R*(long-o_long)*math.cos((o_lat+lat)/2)
        y = R*(lat-o_lat)
        
        return (x,y)
    
    polys = {}
    with open(filename,'r') as f:
        verts = []
        codes = []
        id = None
        for line in f:
            # deal with possible comments
            c_ind = line.find('#')
            if c_ind >= 0:
                line = line[:c_ind]
            if line.strip() == '':
                if len(verts) > 0:
                    # end of current polygon. convert into Path object.
                    verts.append((0.,0.)) # ignored
                    codes.append(Path.CLOSEPOLY)
                    polys[id] = Path(verts,codes)
                    verts = []
                    codes = []
                    id = None
            else:
                # parse the data
                if id is None:
                    # expect a field identifier first
                    id = line
                else:
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
        polys[id] = Path(verts,codes)
        
    return polys
    
 
    
def get_field_cells(polys,domain_info):
    '''Get a list of lists of cell indices that represent each field.
    
    Args:
        polys: Dict of Path objects representing each field
        domain_info: (dist (m), cells) from release point to side of domain
                        (as in the Run.Params class)
                        
    Returns:
        fields: dict containing lists (fields) cells indices'''
    
    fields = {}
    res = domain_info[0]/domain_info[1] #cell resolution
    # construct a list of all x,y coords (in meters) for the center of each cell
    centers = [(col*res,row*res) for row in range(domain_info[1],-domain_info[1]-1,-1)
                            for col in range(-domain_info[1],domain_info[1]+1)]
    for id,poly in polys.items():
        fields[id] = np.argwhere(
            poly.contains_points(centers).reshape(
            domain_info[1]*2+1,domain_info[1]*2+1))
    
    # fields is row,col information assuming the complete domain.
    return fields
            
    
    
###############################################################################
#                                                                             #
#                         Supporting functions                                #
#                                                                             #
###############################################################################
    
def popdensity_to_emergence(modelsol,locinfo):
    '''Translate population model to corresponding expected number of wasps in
    a given location whose oviposition would result in a given emergence date. 
    Only use the locations in which data was actually collected.
    '''
    
    # Assume collections are done at the end of the day.
    
    ### Oviposition to emergence time ###
    # For now, assume this is a constant
    incubation_time = 15 # days (Average of 16 & 14)
    
    ### Project release field grid ###
    release_emerg = []
    for nframe,dframe in enumerate(locinfo.release_DataFrames):
        # Each dataframe should be sorted already, 'datePR','row','column'.
        # Also, the grid for each collection is stored in the list
        #   locinfo.emerg_grids.
        # Find the min and max emergence observation dates PR.
        dframe_min = dframe['datePR'].min().days
        dframe_max = dframe['datePR'].max().days
        
        ### Find the earliest and latest oviposition date PR that we need to ###
        ### consider for this collection. 0 = end of release day.            ###
        # This is dependent on how incubation time is defined
        start_day = max(dframe_min - incubation_time,0) # days post release!
        ########################################################################
        
        # The last day oviposition is possible is always on the day of collection
        last_day = (locinfo.collection_dates[nframe]-locinfo.release_date).days
        
        #
        # Go through each feasible oviposition day of the model, projecting emergence
        #
        
        # emerg_proj holds each grid point in its rows and a different emergence
        #   day in its columns.
        # Feasible emergence days start at collection and go until observation stopped
        emerg_proj = np.zeros((len(locinfo.emerg_grids[nframe]), 
            dframe_max + 1 - last_day))
        
        # go through feasible oviposition days
        for nday,day in enumerate(range(start_day,last_day+1)):
            n = 0 # row/col count
            # in each one, go through grid points projecting emergence date
            #   potentials per adult wasp per cell.
            for r,c in locinfo.emerg_grids[nframe]:
                ###                Project forward and store                 ###
                ### This function can be more complicated if we want to try  ###
                ###   and be more precise. It's a mapping from feasible      ###
                ###   oviposition dates to array of feasible emergence dates ###
                emerg_proj[n,nday] = modelsol[day][r,c]
                ################################################################
                n += 1
                
        # now consolidate these days into just the days data was collected.
        # first, get unique dates
        obs_datesPR = dframe['datePR'].map(lambda t: t.days).unique()
        modelsol_grid_emerg = np.zeros((len(locinfo.emerg_grids[nframe]),
                                        len(obs_datesPR)))
        col_indices = obs_datesPR - last_day
        modelsol_grid_emerg[:,0] = emerg_proj[:,0:col_indices[0]+1]
        for n,col in enumerate(col_indices[1:]):
            col_last = col_indices[n]
            modelsol_grid_emerg[n+1] = emerg_proj[col_last+1:col+1].sum(axis=1)
        release_emerg.append(modelsol_grid_emerg)
        
    ### Project sentinel field emergence ###
    sentinel_emerg = []
    for nframe,dframe in enumerate(locinfo.sent_DataFrames):
        # Each dataframe should be sorted already, 'datePR','id' 
        # Find the min and max emergence observation dates PR.
        dframe_min = dframe['datePR'].min().days
        dframe_max = dframe['datePR'].max().days
        
        ### Find the earliest and latest oviposition date PR that we need to ###
        ### consider for this collection. 0 = end of release day.            ###
        # This is dependent on how incubation time is defined
        start_day = max(dframe_min - incubation_time,0) # days post release!
        ########################################################################
        
        # The last day oviposition is possible is always on the day of collection
        last_day = (locinfo.collection_dates[nframe]-locinfo.release_date).days
        
        #
        # Go through each feasible oviposition day of the model, projecting emergence
        #
        
        # emerg_proj holds each sentinel field in its rows and a different 
        #   emergence day in its columns.
        # Feasible emergence days start at collection and go until observation stopped
        emerg_proj = np.zeros((len(locinfo.sent_ids[nframe]), 
            dframe_max + 1 - last_day))
            
        # go through feasible oviposition days
        for nday,day in enumerate(range(start_day,last_day+1)):
            # for each day, aggregate the population in each sentinel field
            for n,field_id in enumerate(locinfo.sent_ids[nframe]):
                ###     Sum the field cells, project forward and store       ###
                ### This function can be more complicated if we want to try  ###
                ###   and be more precise. It's a mapping from feasible      ###
                ###   oviposition dates to array of feasible emergence dates ###
                field_total = modelsol[day][locinfo.field_cells[field_id][:,0],
                                    locinfo.field_cells[field_id][:,1]].sum()
                emerg_proj[n,nday] = field_total
                ################################################################
        
        # now consolidate these days into just the days data was collected.
        # first, get unique dates
        obs_datesPR = dframe['datePR'].map(lambda t: t.days).unique()
        modelsol_field_emerg = np.zeros((len(locinfo.sent_ids[nframe]),
                                        len(obs_datesPR)))
        col_indices = obs_datesPR - last_day
        modelsol_field_emerg[:,0] = emerg_proj[:,0:col_indices[0]+1]
        for n,col in enumerate(col_indices[1:]):
            col_last = col_indices[n]
            modelsol_field_emerg[n+1] = emerg_proj[col_last+1:col+1].sum(axis=1)
        sentinel_emerg.append(modelsol_field_emerg)
        
    ### This process results in two lists, release_emerg and sentinel_emerg.
    ###     Each entry corresponds to a data collection day.
    ###     Each column corresponds to an emergence observation day (as in data)
    ###     Each row corresponds to a grid point or sentinel field, respectively
    ### This format will need to match a structured data arrays for comparison
    
    return (release_emerg,sentinel_emerg)
        
        
    
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
    
    

###############################################################################
#                                                                             #
#                             PYMC Setup & Run                                #
#                                                                             #
###############################################################################

def main():
    '''Need to catch PM.BndsError and return a zero likelihood'''

    params = Params()
    # Set up location here with command line arguments in a list.
    params.cmd_line_chg(['--kalbar'])
    assert params.site_name+'fields.txt' == 'data/kalbarfields.txt'
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

    locinfo = LocInfo(params.dataset,params.coord,params.domain_info)
    
    
    
    #### Model priors ####
    lam = pm.Beta("lambda",5,1,value=0.95)
    f_a1 = pm.TruncatedNormal("a_1",6,1,0,12,value=6)
    f_a2 = pm.TruncatedNormal("a_2",18,1,12,24,value=18)
    f_b1 = pm.Gamma("b_1",3,1,value=3) #alpha,beta parameterization
    f_b2 = pm.Gamma("b_2",3,1,value=3)
    g_aw = pm.Gamma("a_w",2.2,1,value=2.2)
    g_bw = pm.Gamma("b_w",5,1,value=5)
    sig_x = pm.Gamma("sig_x",42.2,2,value=21.1)
    sig_y = pm.Gamma("sig_y",10.6,1,value=10.6)
    corr = pm.Uniform("rho",-1,1,value=0)
    sig_x_l = pm.Gamma("sig_x",42.2,2,value=21.1)
    sig_y_l = pm.Gamma("sig_y",10.6,1,value=10.6)
    corr_l = pm.Uniform("rho",-1,1,value=0)
    mu_r = pm.Normal("mu_r",1.5,1/0.75**2,value=1.5)
    #n_periods = pm.Poisson("t_dur",10)
    #r = prev. time exponent
    xi = pm.Gamma("xi",1,1,value=1) # presence to oviposition/emergence factor
    em_obs_prob = pm.Beta("em_obs_prob",1,1,value=0.5) # obs prob of emergence in release
                                                       #  field given max leaf collection
    
    #### Data collection model background for sentinel fields ####
    # Need to fix linear units for area. Let's use cells.
    # Effective collection area (constant between fields) is very uncertain
    A_collected = pm.TruncatedNormal("A_collected",25,50,0,
                                    min(locinfo.field_sizes.values()),value=16)  
    # Each field has its own binomial probability.
    N = len(locinfo.sent_ids)
    field_obs_probs = np.empty(N, dtype=object)
    field_obs_vars = np.empty(N, dtype=object)
    field_obs_means = np.empty(N, dtype=object)
    for n,key in enumerate(locinfo.sent_ids):
        #auto-create deterministic variables
        field_obs_means[n] = A_collected/locinfo.field_sizes[key]
        #Lambda to get deterministic variables
        field_obs_vars[n] = pm.Lambda("var_{}".format(key),lambda m=mean: min(m,0.1))
        sent_obs_probs[n] = pm.Beta("sent_obs_prob_{}".format(key),
            field_obs_means[n]*(1-field_obs_means[n]-field_obs_vars[n])/field_obs_vars[n],
            (field_obs_means[n]-field_obs_vars[n])*(1-field_obs_means[n])/field_obs_vars[n])
    
    
    #### Collect variables ####
    @pm.deterministic
    def params_obj(params=params,g_aw=g_aw,g_bw=g_bw,f_a1=f_a1,f_b1=f_b1,
        f_a2=f_a2,f_b2=f_b2,sig_x=sig_x,sig_y=sig_y,corr=corr,
        sig_x_l=sig_x_l,sig_y_l=sig_y_l,corr_l=corr_l,lam=lam,mu_r=mu_r):
        '''Return altered parameter object to be passed in to simulation'''
        
        # g wind function parameters
        params.g_params = (g_aw,g_bw)
        # f time of day function parameters
        params.f_params = (f_a1,f_b1,f_a2,f_b2)
        # Diffusion coefficients
        params.Dparams = (sig_x,sig_y,corr)
        params.Dlparams = (sig_x_l,sig_y_l,corr_l)
        # Probability of any flight during the day under ideal circumstances
        params.lam = lam
        
        # TRY BOTH - VARYING mu_r OR n_periods
        # scaling flight advection to wind advection
        params.mu_r = mu_r
        # number of time periods (based on interp_num) in one flight
        #params.n_periods = n_periods # if interp_num = 30, this is # of minutes
        
        return params
        
        
    
    #### Run model ####
    @pm.deterministic
    def pop_model(params=params_obj,locinfo=locinfo,wind_data=wind_data,days=days):
        '''This function acts as an interface between PyMC and the model.
        Not only does it run the model, but it provides an emergence potential 
        based on the population model result projected forward from feasible
        oviposition dates. To modify how this projection happens, edit 
        popdensity_to_emergence. Returned values from this function should be
        nearly ready to compare to data.
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
        
        # modelsol now holds the model results for this run as CSR sparse arrays
        
        # get emergence potential (measured in expected number of wasps previously
        #   present whose oviposition would result in emergence on the given date)
        #   from the model result
        release_emerg,sentinel_emerg = popdensity_to_emergence(modelsol,locinfo)
        
        ## This process results in two lists, release_emerg and sentinel_emerg.
        ##     Each entry corresponds to a data collection day.
        ##     Each column corresponds to an emergence observation day (as in data)
        ##     Each row corresponds to a grid point or sentinel field, respectively
        return (release_emerg,sentinel_emerg)
        
        
        
    ### Parse the results of pop_model into separate deterministic variables ###
    @pm.deterministic
    def sent_poi_rates(locinfo=locinfo,xi=xi,emerg_model=pop_model[1],
            betas=sent_obs_probs):
        '''xi is constant, emerg is a list of ndarrays, betas is a 1D array of
        field probabilities'''
        Ncollections = len(locinfo.sent_DataFrames)
        poi_rates = []
        for ii in range(Ncollections):
            ndays = len(locinfo.sent_DataFrames[ii]['datePR'].unique())
            tile_betas = np.tile(betas,(ndays,1)).T
            poi_rates.append(xi*emerg_model[ii]*tile_betas)
        return poi_rates
        
    @pm.deterministic
    def rel_poi_rates(locinfo=locinfo,xi=xi,beta=em_obs_prob,emerg_model=pop_model[0]):
        '''xi is constant, emerg is a list of ndarrays. collection effort is
        specified in locinfo.'''
        Ncollections = len(locinfo.release_DataFrames)
        poi_rates = []
        for ii in range(Ncollections):
            r_effort = locinfo.release_collection[ii] #fraction of max collection
            ndays = len(locinfo.release_DataFrames[ii]['datePR'].unique())
            tile_betas = np.tile(r_effort*beta,(ndays,1)).T
            poi_rates.append(xi*emerg_model[ii]*tile_betas)
        return poi_rates
            
    

    
    # Given the expected wasp densities from pop_model, actual wasp densities
    #   are modeled as a thinned Poisson random variable about that mean.
    # Each wasp in the area then has a small probability of being seen.
    
    ### Connect sentinel emergence data to model ###
    N_sent_collections = len(locinfo.sent_DataFrames)
    # Create list of collection variables
    sent_collections = []
    for ii in range(N_sent_collections):
        sent_collections.append(pm.Poisson("sent_em_obs_{}".format(ii),
            sent_poi_rates[ii], value=locinfo.sentinel_emerg[ii], observed=True))
            
    ### Connect release-field emergence data to model ###
    N_release_collections = len(locinfo.release_DataFrames)
    # Create list of collection variables
    rel_collections = []
    for ii in range(N_release_collections):
        rel_collections.append(pm.Poisson("rel_em_obs_{}".format(ii),
            rel_poi_rates[ii], value=locinfo.release_emerg[ii], observed=True))
        

    
if __name__ == "__main__":
    main(sys.argv[1:])