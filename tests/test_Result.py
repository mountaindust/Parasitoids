''' Test the output of the population model

Author: Christopher Strickland
Email: wcstrick@live.unc.edu'''

import pytest
import numpy as np
from scipy import sparse
from Run import Params
from conftest import data_avail

@data_avail
def test_result(params,modelsol):
    # params was loaded with modelsol, so they should match
    # modelsol should be a population model

    assert params.domain_info[1]*2+1 == modelsol[0].shape[0]

    # tests after release has finished
    for mday in modelsol[:5]:
        # check positivity
        assert mday.min() >= -1e-10
        # unless parasitoids have been exiting the domain, the total number
        #   should be close to the starting number
        assert params.r_number*0.99 < mday.sum() < params.r_number*1.01
