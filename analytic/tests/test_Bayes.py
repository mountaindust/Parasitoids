#! /usr/bin/env python3

'''Test module for Bayes_MCMC

Author: Christopher Strickland'''

import os.path
import pytest
import numpy as np
import pandas as pd
from scipy import sparse
from matplotlib.path import Path
from Data_Import import LocInfo
from Run import Params
import Bayes_MCMC as Bayes

###############################################################################
#                                                                             #
#                              Test Fixtures                                  #
#                                                                             #
###############################################################################

@pytest.fixture(scope="module")
def locinfo():
   # kalbar info
   loc_name = 'kalbar'
   center = (-27.945752,152.58474)
   domain_info = (5000.0,1000)
   return LocInfo(loc_name,center,domain_info)

# path to sample pop model output
sample_data = 'output/from_nemo/kalbar_pop0420-1939'

@pytest.fixture()
def modelsol():
    # return a sample model solution
    if not (os.path.isfile(sample_data+'.npz') and 
            os.path.isfile(sample_data+'.json')):
        return None
    else:
        # load parameters
        params = Params()
        params.file_read_chg(sample_data)

        dom_len = params.domain_info[1]*2 + 1

        # load data
        modelsol = []
        with np.load(sample_data+'.npz') as npz_obj:
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
        return modelsol
   


############                    Decorators                ############

data_avail = pytest.mark.skipif(not (os.path.isfile(sample_data+'.npz') and 
                                     os.path.isfile(sample_data+'.json')),
                                reason = 'Could not find file {}.'.format(
                                    sample_data))
   
   
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
        assert n_fields == len(locinfo.sent_ids[ii])
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
        for n,field in enumerate(locinfo.sent_ids[ii]):
            for day in locinfo.sent_DataFrames[ii]['datePR'].unique():
                assert locinfo.sent_DataFrames[ii]\
                    [locinfo.sent_DataFrames[ii]['datePR']==day]\
                    ['id'].values[n] == field

        # release_collection should be relative numbers
        assert locinfo.release_collection[ii].max() == 1
        assert locinfo.release_collection[ii].min() >= 0