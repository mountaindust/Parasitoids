#! /usr/bin/env python3

'''Test module for Bayes_MCMC

Author: Christopher Strickland'''

import pytest
import numpy as np
import pandas as pd
from matplotlib.path import Path
from Data_Import import LocInfo
import Bayes_MCMC as Bayes

###############################################################################
#                                                                             #
#                              Test Fixtures                                  #
#                                                                             #
###############################################################################

@pytest.fixture()
def field_info():
   # kalbar info
   filename = 'data/kalbar'
   center = (-27.945752,152.58474)
   domain_info = (5000.0,1000)
   return (filename,center,domain_info)
   
   
   
###############################################################################
#                                                                             #
#                                   Tests                                     #
#                                                                             #
###############################################################################

def test_LocInfo(field_info):
    '''Test initialization of LocInfo object'''
    filename,center,domain_info = field_info
    
    locinfo = LocInfo('kalbar',center,domain_info)

    ### Field boundary information ###

    # field_polys should be a dict of Path objects
    assert type(locinfo.field_polys) is dict
    # Fields: A, B, C, D, E, F, G
    assert type(locinfo.field_polys['A']) is Path
    assert len(locinfo.field_polys) == 7

    # field_cells should be a dict of lists
    assert type(locinfo.field_cells) is dict
    assert isinstance(locinfo.field_cells['A'],np.ndarray)
    assert len(locinfo.field_cells) == 7

    # field_sizes should be a dict of cell counts
    assert type(locinfo.field_sizes) is dict
    assert type(locinfo.field_sizes['A']) is int
    assert len(locinfo.field_sizes) == 7

    ### Release field grid info ###

    assert isinstance(locinfo.grid_data,pd.DataFrame)
    for key in ['xcoord','ycoord','samples','collection']:
        assert key in locinfo.grid_data.keys()
    assert isinstance(locinfo.grid_cells,np.ndarray)
    assert locinfo.grid_cells.shape[0] == 2
    assert locinfo.grid_data['xcoord'].size == locinfo.grid_cells.shape[1]

    # Less trivial testing for the above is accomplished by running
    #     Plot_SampleLocations.py which plots the fields and polys together

    ### Sentinel field emergence data ###
    assert isinstance(locinfo.release_date,pd.Timestamp)
    assert isinstance(locinfo.collection_dates,list)
    assert isinstance(locinfo.collection_dates[0],pd.Timestamp)
    assert locinfo.collection_dates[0] > locinfo.release_date
    assert isinstance(locinfo.sent_DataFrames[0],pd.DataFrame)
    for key in ['id','datePR','E_total','All_total']:
        assert key in locinfo.sent_DataFrames[0].keys()
    assert np.all(locinfo.sent_DataFrames[0]['E_total'].values <=
                  locinfo.sent_DataFrames[0]['All_total'].values)
    # test that we have the field cell info for all the sentinel field data
    for key in locinfo.sent_ids[0]:
        assert key in locinfo.field_cells.keys()
    # Test emergence post release dates
    minTimedelta = locinfo.collection_dates[0] - locinfo.release_date
    for Td in locinfo.sent_DataFrames[0]['datePR']:
        assert Td >= minTimedelta

    ### Release field emergence data ###
    assert isinstance(locinfo.releasefield_id,str)
    for key in ['row','column','xcoord','ycoord','datePR','E_total','All_total']:
        assert key in locinfo.release_DataFrames[0].keys()
    for coord in locinfo.release_DataFrames[0][['xcoord','ycoord']].values:
        assert coord in locinfo.grid_data[['xcoord','ycoord']].values
    assert np.all(locinfo.release_DataFrames[0]['E_total'].values <=
                  locinfo.release_DataFrames[0]['All_total'].values)
    for Td in locinfo.release_DataFrames[0]['datePR']:
        assert Td >= minTimedelta
    grid_cells_list = locinfo.grid_cells.T.tolist()
    for cell in locinfo.release_DataFrames[0][['row','column']].values.tolist():
        assert cell in grid_cells_list
        assert tuple(cell) in locinfo.emerg_grids[0]

    ### PyMC friendly data structures ###