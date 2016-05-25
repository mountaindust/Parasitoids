#! /usr/bin/env python3

'''Test module for Data_Import and Bayes_funcs

Author: Christopher Strickland
Email: cstrickland@samsi.info
'''

import pytest
import numpy as np
import pandas as pd
from scipy import sparse
from matplotlib.path import Path
from Data_Import import LocInfo
from Run import Params
import Bayes_funcs as Bayes
from conftest import data_avail

###############################################################################
#                                                                             #
#                              Test Fixtures                                  #
#                                                                             #
###############################################################################

@pytest.fixture(scope="module")
def locinfo(domain_info):
   # kalbar info
   loc_name = 'kalbar'
   center = (-27.945752,152.58474)
   return LocInfo(loc_name,center,domain_info)

   
###############################################################################
#                                                                             #
#                                   Tests                                     #
#                                                                             #
###############################################################################

def test_LocInfo(locinfo):
    '''Test initialization of LocInfo object'''

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
    assert locinfo.grid_cells.shape[1] == 2
    assert locinfo.grid_data['xcoord'].size == locinfo.grid_cells.shape[0]

    # Less trivial testing for the above is accomplished by running
    #     Plot_SampleLocations.py which plots the fields and polys together

    ### Sentinel field emergence data ###
    assert isinstance(locinfo.release_date,pd.Timestamp)
    assert isinstance(locinfo.collection_datesPR,list)
    assert isinstance(locinfo.collection_datesPR[0],pd.Timedelta)
    assert locinfo.collection_datesPR[0] > pd.Timedelta('0 days')
    assert isinstance(locinfo.sent_DataFrames[0],pd.DataFrame)
    for key in ['id','datePR','E_total','All_total']:
        assert key in locinfo.sent_DataFrames[0].keys()
    assert np.all(locinfo.sent_DataFrames[0]['E_total'].values <=
                  locinfo.sent_DataFrames[0]['All_total'].values)
    # test that we have the field cell info for all the sentinel field data
    for key in locinfo.sent_ids:
        assert key in locinfo.field_cells.keys()
    # Test emergence post release dates
    minTimedelta = locinfo.collection_datesPR[0]
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
    grid_cells_list = locinfo.grid_cells.tolist()
    for cell in locinfo.release_DataFrames[0][['row','column']].values.tolist():
        assert cell in grid_cells_list
        assert tuple(cell) in locinfo.emerg_grids[0]

    ### Grid observation data ###
    assert isinstance(locinfo.grid_obs_DataFrame,pd.DataFrame)
    assert isinstance(locinfo.grid_obs_datesPR,list)
    assert isinstance(locinfo.grid_obs_datesPR[0],pd.Timedelta)
    assert isinstance(locinfo.grid_obs,np.ndarray)
    assert isinstance(locinfo.grid_samples,np.ndarray)
    assert np.all(locinfo.grid_obs.shape == locinfo.grid_samples.shape)
    assert locinfo.grid_samples.max() == 1
    # grid_obs should not be all zeros, asssuming something was seen
    assert locinfo.grid_obs.max() > 0

    ### Cardinal direction data ###
    assert isinstance(locinfo.card_obs_DataFrames,list)
    assert isinstance(locinfo.card_obs_DataFrames[0],pd.DataFrame)
    assert isinstance(locinfo.card_obs_datesPR,list)
    assert isinstance(locinfo.card_obs_datesPR[0],pd.Timedelta)
    assert isinstance(locinfo.step_size,list)
    assert isinstance(locinfo.card_obs,list)
    assert isinstance(locinfo.card_obs[0],np.ndarray)
    assert len(locinfo.card_obs_DataFrames) == len(locinfo.card_obs_datesPR)\
        == len(locinfo.step_size) == len(locinfo.card_obs)
    for c_obs in locinfo.card_obs:
        assert c_obs.shape[0] == 4

    ### PyMC friendly data structures ###
    # these primarily need to be verfied against model output, so we will test
    #   them there.



@data_avail
def test_model_emergence(locinfo,modelsol):
    '''Test the translation of population model results to emergence information,
    and how this compares with the PyMC friendly data structures in LocInfo'''

    release_emerg,sentinel_emerg = Bayes.popdensity_to_emergence(modelsol,locinfo)

    # This process results in two lists, release_emerg and sentinel_emerg.
    #     Each list entry corresponds to a data collection day (one array)
    #     In each array:
    #     Each column corresponds to an emergence observation day (as in data)
    #     Each row corresponds to a grid point or sentinel field, respectively

    # These lists are emergence potential as in wasp population numbers.
    #   To get observed emergence, collection and oviposition rate must be
    #   modeled. But this is done in Bayesian fashion and won't be reproduced here.

    # Regardless, these numbers are now considered independent variables and
    #   should match the general shape of the data stored in locinfo.

    assert isinstance(release_emerg,list)
    for ii in range(len(release_emerg)):
        n_grid_pts, n_obs = release_emerg[ii].shape
        # test shape against data info
        assert n_grid_pts == len(locinfo.emerg_grids[ii])
        assert n_obs == len(locinfo.release_DataFrames[ii]['datePR'].unique())
        # test shape against locinfo ndarrays
        assert n_grid_pts == locinfo.release_emerg[ii].shape[0]
        assert n_obs == locinfo.release_emerg[ii].shape[1]
        assert n_grid_pts == locinfo.release_collection[ii].size

        n_fields, n_obs = sentinel_emerg[ii].shape
        # test shape against data info
        assert n_fields == len(locinfo.sent_ids)
        assert n_obs == len(locinfo.sent_DataFrames[ii]['datePR'].unique())
        # test shape against locinfo ndarray
        assert n_fields == sentinel_emerg[ii].shape[0]
        assert n_obs == sentinel_emerg[ii].shape[1]

        # make sure that the grid points match from model to data
        for n,cell in enumerate(locinfo.emerg_grids[ii]):
            for day in locinfo.release_DataFrames[ii]['datePR'].unique():
                assert tuple(locinfo.release_DataFrames[ii]
                    [locinfo.release_DataFrames[ii]['datePR']==day]
                    [['row','column']].values[n,:]) == cell

        # same for sentinel fields
        for n,field in enumerate(locinfo.sent_ids):
            for day in locinfo.sent_DataFrames[ii]['datePR'].unique():
                assert locinfo.sent_DataFrames[ii]\
                    [locinfo.sent_DataFrames[ii]['datePR']==day]\
                    ['id'].values[n] == field

        # release_collection should be relative numbers
        assert locinfo.release_collection[ii].max() == 1
        assert locinfo.release_collection[ii].min() >= 0



@data_avail
def test_model_sampling(locinfo,modelsol,domain_info):
    '''Test the translation of population model results to the PyMC friendly
    data structures in LocInfo'''

    grid_counts = Bayes.popdensity_grid(modelsol,locinfo)
    card_counts = Bayes.popdensity_card(modelsol,locinfo,domain_info)

    # grid_counts should be comparable to locinfo.grid_obs and locinfo.grid_samples
    assert np.all(grid_counts.shape == locinfo.grid_obs.shape == 
                  locinfo.grid_samples.shape)
    # they should be non-negative, and something should be > 0
    assert grid_counts.max() > 0
    assert grid_counts.min() >= 0

    # each entry in card_counts should match each cooresponding entry in
    #   locinfo.card_obs
    for nobs,obs in enumerate(locinfo.card_obs):
        assert np.all(obs.shape == card_counts[nobs].shape)
        # Simulation should be >= 0, with a max > 0
        card_counts[nobs].max() > 0
        card_counts[nobs].min() >= 0