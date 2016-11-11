'''File for configuring py.test tests'''

import os.path
import pytest
import numpy as np
from scipy import sparse
from Run import Params

def pytest_addoption(parser):
    parser.addoption('--runslow', action='store_true', help='run slow tests')

############                    Decorators                ############

#### path to sample pop model output (do not include file extension) ####
sample_data = 'output/kalbar_pop0912-1132'

data_avail = pytest.mark.skipif(not (os.path.isfile(sample_data+'.npz') and
                                     os.path.isfile(sample_data+'.json')),
                                reason = 'Could not find file {}.'.format(
                                    sample_data))

############                    Fixtures                  ############

@pytest.fixture(scope="session")
def params():
    # load parameters if possible
    if os.path.isfile(sample_data+'.json'):
        params = Params()
        params.file_read_chg(sample_data)
        return params
    else:
        return None

@pytest.fixture(scope="session")
def domain_info(params):
    # Return infomation about the domain size and refinement
    if params is not None:
        return params.domain_info
    else:
        # default values: 0.5 domain size, # of cells from center to side
        return (8000.0,320)

@pytest.fixture(scope="session")
def modelsol(domain_info):

    # return a sample model solution
    if not (os.path.isfile(sample_data+'.npz') and
            os.path.isfile(sample_data+'.json')):
        return None
    else:
        dom_len = domain_info[1]*2 + 1

        # load data
        modelsol = []
        with np.load(sample_data+'.npz') as npz_obj:
            days = npz_obj['days']
            # some code here to make loading robust to both COO and CSR.
            CSR = False
            for day in days:
                V = npz_obj[str(day)+'_data']
                # Solution should be reconstructed as CSR sparse
                indices = npz_obj[str(day)+'_ind']
                indptr = npz_obj[str(day)+'_indptr']
                modelsol.append(sparse.csr_matrix((V,indices,indptr),
                                                    shape=(dom_len,dom_len)))

        # check that modelsol dimensions match expected dimensions
        assert modelsol[0].shape[0] == 2*domain_info[1]+1, 'Unexpected dimensions.'
        return modelsol