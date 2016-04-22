#! /usr/bin/env python3

'''Test module for Bayes_MCMC

Author: Christopher Strickland'''

import pytest
import numpy as np
from matplotlib.path import Path
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
   return (filename,center)
   
   
   
###############################################################################
#                                                                             #
#                                   Tests                                     #
#                                                                             #
###############################################################################

def test_get_fields(field_info):
    filename,center = field_info
    filename += 'fields.txt'
    polys = Bayes.get_fields(filename,center)
    # polys should be a list of Path objects
    assert type(polys) is dict
    assert type(list(polys.values())[0]) is Path
    assert len(polys.keys()) == 7