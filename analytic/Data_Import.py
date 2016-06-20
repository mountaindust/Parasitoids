'''This module imports all data (except wind data) and stores it in LocInfo

Author: Christopher Strickland  
Email: cstrickland@samsi.info 
'''

import time, math
import numpy as np
import pandas as pd
from matplotlib.path import Path

class LocInfo(object):
    '''Class def to hold all info on experimental location and resulting data.
    Properties:
        ### Loaded from field boundary information ###
        field_polys: dict of Path objects
        field_cells: dict of cell lists
        field_sizes: dict of cell counts

        ### Loaded from release grid info ###
        grid_data: DataFrame (xcoord,ycoord,samples,collection)
        grid_cells: grid location array, col 0 = row index, col 1 = col index

        ### Sentinel field emergence data ###
        release_date: Timestamp
        collection_datesPR: list of TimeDeltas
        sent_DataFrames: list of sample days, (id,datePR,E_total,All_total)
        sent_ids: list of field ID strings. Assume list is const over collections

        ### Release field emergence data ###
        releasefield_id: string, field ID
        release_DataFrames: list of sample days, (row,column,xcoord,ycoord,
                                                  datePR,E_total,All_total)
        emerg_grids: list of (row,col) lists, grid pts used in emerg collection
        
        ### Release field grid observation data ###
        grid_obs_DataFrame: DataFrame (xcoord,ycoord,datePR,obs_count)
        grid_obs_datesPR: list of obs dates PR (Timedelta)
        grid_obs: ndarray to compare to popdensity_grid
        grid_samples: ndarray with relative sampling effort

        ### Cardinal direction observation data ###
        card_obs_DataFrames: list of sample days, (direction,distance,obs_count)
        card_obs_datesPR: list of obs dates PR (Timedelta)
        step_size: list of floats (meters)
        card_obs: list of arrays

        ### PyMC friendly data structures ###
        release_emerg: list of arrays
        release_collection: list of arrays, relative collection effort
        sentinel_emerg: list of arrays'''
    
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
        self.field_polys = LocInfo.get_fields('./data/'+location+'fields.txt',
            release_latlong) # this is a dict. keys are field labels
        ## ## Remove furthest field ## ##
        del self.field_polys['G']
        # Convert to dict of cell indices
        self.field_cells = LocInfo.get_field_cells(self.field_polys,domain_info)
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
        #####
        ### Get row/column indices from xcoord and ycoord in grid_data
        res = domain_info[0]/domain_info[1] # cell length in meters
        self.grid_cells = np.array([-self.grid_data['ycoord'].values,
                                    self.grid_data['xcoord'].values])
        self.grid_cells = (np.around(self.grid_cells/res) + 
                            domain_info[1]).T.astype(int)
        # self.grid_cells is now: col 0 = row index, col 1 = col index
        
        ##### Import and parse sentinel field emergence data #####
        #   What this section looks like will be highly dependent on the 
        #       particulars of your dataset - every dataset is different.
        #       Fire up pandas in an ipython notebook and play with your data
        #       until you have what you need. Then put that procedure in the
        #       following method. What the method needs to accomplish is detailed
        #       in its docstring.
        #   Initializes:
        #       self.release_date
        #       self.collection_datesPR
        #       self.sent_DataFrames
        self.get_sentinel_emergence(location)
        #####
        ### Get ordered list of sentinel field ids
        ### Assume all sentinel fields were used in each collection
        
        self.sent_ids = list(self.sent_DataFrames[0]['id'].unique())

        ##### Import and parse release field emergence data #####
        #   Again, highly dependent on what your dataset looks like. See method
        #       docstring for details.
        #   Initializes:
        #       self.releasefield_id
        #       self.release_DataFrames
        self.get_releasefield_emergence(location)
        #####
        ### Further parsing, including sorting the DataFrame
        self.emerg_grids = []
        for dframe in self.release_DataFrames:
            # Get row/column indices from xcoord and ycoord
            dframe['row'] = ((-dframe['ycoord']/res).round(0) +
                domain_info[1]).astype(int)
            dframe['column'] = ((dframe['xcoord']/res).round(0) +
                domain_info[1]).astype(int)
            # Re-sort the dataframes so that as one loops over the days, the row/col
            #   info will occur in the same order.
            dframe.sort_values(['datePR','row','column'],inplace=True)
            dframe.reset_index(inplace=True,drop=True)
            # Get the grid points that were collected
            oneday = dframe['datePR'] == dframe['datePR'].min()
            self.emerg_grids.append(list(zip(dframe['row'][oneday].values,
                                        dframe['column'][oneday].values)))

        ##### Import and parse grid adult observation data #####
        #   Dependent on what your dataset looks like; add pandas to method.
        #   Initializes:
        #       self.grid_obs_DataFrame
        #       self.grid_obs_datesPR
        self.get_grid_observations(location)
        #####
        ### Form a data structure that can be compared to popdensity_grid
        self.grid_obs = np.zeros((self.grid_cells.shape[0],
                                  len(self.grid_obs_datesPR)))
        self.grid_samples = np.zeros((self.grid_cells.shape[0],
                                  len(self.grid_obs_datesPR)))
        for nday,date in enumerate(self.grid_obs_datesPR):
            # for each date, get the number of samples taken and num observed
            #   at each grid point
            for n in range(self.grid_data.shape[0]):
                self.grid_samples[n,nday] = self.grid_data['samples'].iloc[n]
                xyrow = pd.merge(self.grid_data.iloc[n:n+1],
                                 self.grid_obs_DataFrame[
                                 self.grid_obs_DataFrame['datePR']==date],
                                 on=['xcoord','ycoord'], how='inner')
                if not xyrow.empty:
                    self.grid_obs[n,nday] = xyrow['obs_count'].values
        self.grid_samples = self.grid_samples/self.grid_samples.max()

        ##### Import and parse cardinal direction adult observation data #####
        #   Dependent on what your dataset looks like; add pandas to method.
        #   Initializes:
        #       self.card_obs_DataFrames
        #       self.card_obs_datesPR
        #       self.step_size
        self.get_card_observations(location)
        #####
        ### Sort each DataFrame and form a data structure that can be compared 
        ### to popdensity_grid
        self.card_obs = []
        for dframe in self.card_obs_DataFrames:
            dframe.sort_values(['direction','distance'],inplace=True)
            north = dframe[dframe['direction']=='north']['obs_count'].values
            south = dframe[dframe['direction']=='south']['obs_count'].values
            east = dframe[dframe['direction']=='east']['obs_count'].values
            west = dframe[dframe['direction']=='west']['obs_count'].values
            maxlen = max(north.size,south.size,east.size,west.size)
            card_obs = np.zeros((4,maxlen))
            card_obs[0,:north.size] = north
            card_obs[1,:south.size] = south
            card_obs[2,:east.size] = east
            card_obs[3,:west.size] = west
            self.card_obs.append(card_obs)
                                        
        ##### Gather data in a form that can be quickly compared to the #####
        #####   output of popdensity_to_emergence                       #####
        # Want three lists: release emerg, collection effort, sentinel emerg
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
                valary = self.grid_data[(self.grid_data['xcoord'] == x) & \
                    (self.grid_data['ycoord'] == y)]['collection'].values
                assert valary.shape == (1,) #each specified only once
                r_array.append(valary[0])
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
            E_array = np.zeros((len(self.sent_ids),len(obs_datesPR)))
            #All_array = np.zeros((len(self.sent_ids),len(obs_datesPR)))
            for ndate,date in enumerate(obs_datesPR):
                E_array[:,ndate] = dframe[dframe['datePR'] == date]['E_total'].values
                #All_array[:,ndate] = dframe[dframe['datePR'] == date]['All_total'].values
            self.sentinel_emerg.append(E_array)
            #self.sentinel_emerg_total.append(All_array)



    @staticmethod
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
                line = line.strip()
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
    
 
    
    @staticmethod
    def get_field_cells(polys,domain_info):
        '''Get a dict of lists of cell indices that represent each field.
    
        Args:
            polys: Dict of Path objects representing each field
            domain_info: (dist (m), cells) from release point to side of domain
                            (as in the Run.Params class)
                        
        Returns:
            fields: dict containing lists (fields) cells indices'''
    
        fields = {}
        res = domain_info[0]/domain_info[1] #cell resolution
        # construct a list of all x,y coords (in meters) for the center of each cell
        colmesh,rowmesh = np.meshgrid(
                        res*np.arange(-domain_info[1],domain_info[1]+1),
                        res*np.arange(domain_info[1],-domain_info[1]-1,-1))
        centers = np.array([colmesh.flatten(),rowmesh.flatten()]).T
        # For each field, get the cells that are located in the field.
        for id,poly in polys.items():
            fields[id] = np.argwhere(
                poly.contains_points(centers).reshape(
                domain_info[1]*2+1,domain_info[1]*2+1))
    
        # fields is row,col information assuming the complete domain.
        return fields
        
            
    
    @staticmethod            
    def get_release_grid(filename):
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
            self.collection_datesPR: a list of TimeDelta collection dates PR
            self.sent_DataFrames: a list of pandas DataFrames, one for each
                                    collection date.
        EACH DATAFRAME MUST INCLUDE THE FOLLOWING COLUMNS:
            id: Field indentifier that matches the keys in self.field_cells
            datePR: Num of days the emergence occured post-release (dtype=Timedelta)
            E_total: Total number of wasp emergences in that field on that date
            All_total: Total number of all emergences in that field on that date
                        (this will later be summed to obtain emergences per
                         field/collection)
        BE SURE TO SORT EACH DATAFRAME AND RESET THE INDICES BEFORE RETURNING!
        '''
        
        if location == 'kalbar':
            # location of data excel file
            data_loc = 'data/sampling_details.xlsx'
            # date of release (as a pandas TimeStamp, year-month-day)
            #   (leave off time of release)
            self.release_date = pd.Timestamp('2005-03-13')
            # dates of collection PR as a list of TimeDeltas
            self.collection_datesPR = [pd.Timestamp('2005-03-31')]
            for n,date in enumerate(self.collection_datesPR):
                self.collection_datesPR[n] = date - self.release_date
            # list of sentinel field collection dates (as pandas TimeStamps)
            #self.collection_dates = [pd.Timestamp('2005-05-31')]
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
            
            ### Sort DataFrame
            sentinel_fields_data.sort_values(['datePR','id'],inplace=True)
            sentinel_fields_data.reset_index(inplace=True,drop=True)
                                    
            ## ## Remove furthest field ## ##
            sentinel_fields_data = sentinel_fields_data[
                                    sentinel_fields_data['id'] != 'G']
            
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
            
            ### Remove origin collection data (not well defined)
            release_field_data = release_field_data[\
                (release_field_data['xcoord'] != 0) & \
                (release_field_data['ycoord'] != 0)]
                
            ### Store DataFrame in list
            self.release_DataFrames.append(release_field_data)
            
        else:
            raise NotImplementedError

    def get_grid_observations(self,location):
        '''Get data relating to release field grid observations
        This implementation will need to change completely according to the
            structure of your dataset. Parsing routines for multiple locations'
            data can be stored here - just extend the if-then clause based on
            the value of the location argument.
        WHAT IS REQURIED:
            self.grid_obs_DataFrame: a pandas DataFrame with all non-zero
                                     observations. It will be assumed that the
                                     entire grid was sampled, but that omissions
                                     are zeros.
            self.grid_obs_datesPR: list of Timedeltas of observation dates
        THE DATAFRAME MUST INCLUDE THE FOLLOWING COLUMNS:
            xcoord: distance east from release point in meters (grid collection point)
            ycoord: distance north from release point in meters (grid collection point)
            datePR: Num of days the observation occured post-release (dtype=Timedelta)
            obs_count: Total number of wasp observations in that field on that date
        BE SURE TO SORT EACH DATAFRAME AND RESET THE INDICES BEFORE RETURNING!
        '''

        if location == 'kalbar':
            # location of data excel file
            data_loc = 'data/adult_counts_kalbar.xlsx'

            ### Pandas
            # load the grid adult counts sheet
            grid_obs = pd.read_excel(data_loc,sheetname='adult counts field A')
            # rename the headings with spaces in them
            grid_obs.rename(columns={"x coor":"x","y coor":"y", 
                                     "num leaves viewed": "leaves",
                                     "num hayati":"obs_count"}, inplace=True)
            # we don't really care about the leaf num columns
            grid_obs = grid_obs[['date','collector','x','y','leaves','obs_count']]
            # in our data, North was on the left of the grid. So switch coordinates
            grid_obs['xcoord'] = grid_obs['y']
            grid_obs['ycoord'] = -grid_obs['x'] # need to flip orientation
            grid_obs.drop(['x','y'],1,inplace=True)
            # put release point at the origin
            grid_obs['ycoord'] += 300
            grid_obs['xcoord'] -= 200
            # convert date to datePR
            grid_obs['datePR'] = grid_obs['date'] - self.release_date
            ### Sort the DataFrame
            grid_obs.sort_values(['datePR','xcoord','ycoord'],inplace=True)
            grid_obs.reset_index(inplace=True,drop=True)
            self.grid_obs_datesPR = []
            # unique() returns ndarray of numpy.timedelta.
            # want: list of pd.Timedelta
            for npdate in grid_obs['datePR'].unique():
                self.grid_obs_datesPR.append(pd.Timedelta(npdate))
            self.grid_obs_DataFrame = grid_obs

    def get_card_observations(self,location):
        '''Get data relating to release field grid observations
        This implementation will need to change completely according to the
            structure of your dataset. Parsing routines for multiple locations'
            data can be stored here - just extend the if-then clause based on
            the value of the location argument.
        WHAT IS REQURIED:
            self.card_obs_DataFrames: list of pandas DataFrames with cardinal
                                        direction observations. Each DataFrame
                                        is a separate date.
            self.card_obs_datesPR: list of Timedeltas of observation dates
            self.step_size: list of step sizes (meters) for sampling
        THE DATAFRAME MUST INCLUDE THE FOLLOWING COLUMNS:
            direction: string, cardinal direction (north,south,east,west)
            distance: in meters
            obs_count: Total number of wasp observations in that field on that date
        '''

        if location == 'kalbar':
            # location of data excel file
            data_loc = 'data/adult_counts_kalbar.xlsx'

            # names of the data sheets
            sheets = ['cardinal 15 mar 05','cardinal 21 mar 05']
            self.step_size = [2,2]
            self.card_obs_DataFrames = []
            self.card_obs_datesPR = []
            for sheet in sheets:
                # load the cardinal directions sheet
                cardinal_obs = pd.read_excel(data_loc,sheetname=sheet)
                # rename the one heading with a space
                cardinal_obs.rename(columns={"num adults":"obs_count"},inplace=True)
                cardinal_obs.drop('num viewers',1,inplace=True)
                cardinal_obs['datePR'] = cardinal_obs['date'] - self.release_date
                self.card_obs_datesPR.append(cardinal_obs['datePR'].iloc[0])
                self.card_obs_DataFrames.append(cardinal_obs)