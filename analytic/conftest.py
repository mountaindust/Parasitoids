'''File for configuring py.test tests'''

import os.path
import pytest
import numpy as np
from scipy import sparse
from Run import Params

def pytest_addoption(parser):
    parser.addoption('--runslow', action='store_true', help='run slow tests')
    
############                    Decorators                ############

# path to sample pop model output
sample_data = 'output/from_nemo/kalbar_pop0522-0047'

data_avail = pytest.mark.skipif(not (os.path.isfile(sample_data+'.npz') and 
                                     os.path.isfile(sample_data+'.json')),
                                reason = 'Could not find file {}.'.format(
                                    sample_data))
                                    
############                    Fixtures                  ############

@pytest.fixture(scope="session")   
def domain_info():
    # Return infomation about the domain size and refinement
    
    # distance from release point to a side of the domain in meters
    rad_dist = 5000.0
    # number of cells from the center to side of domain
    rad_res = 1000
    
    # (each cell is thus rad_dist/rad_res meters squared in size)
    return (rad_dist,rad_res)
    
@pytest.fixture(scope="session")
def modelsol(domain_info):
    
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
        # check that modelsol dimensions match expected dimensions
        assert modelsol[0].shape[0] == 2*domain_info[1]+1, 'Unexpected dimensions.'
        return modelsol