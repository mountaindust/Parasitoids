#! /usr/bin/env python3

'''Test module for Bayes_MCMC

Author: Christopher Strickland'''

import pytest
import numpy as np
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

def test_get_fields(field_info):
    ''' Test the static functions of LocInfo which load data on 
    field locations and the release grid.
    '''
    filename,center,domain_info = field_info
    filename += 'fields.txt'
    polys = LocInfo.get_fields(filename,center)
    # polys should be a dict of Path objects
    assert type(polys) is dict
    # Fields: A, B, C, D, E, F, G
    assert type(polys['A']) is Path
    assert len(polys) == 7

    # Also test get_field_cells
    field_cells = LocInfo.get_field_cells(polys,domain_info)
    # This should be a list of lists
    assert type(field_cells) is dict
    assert isinstance(field_cells['A'],np.ndarray)

    ### Less trivial testing for these methods is accomplished by running
    ###     Plot_SampleLocations.py which plots the fields and polys together