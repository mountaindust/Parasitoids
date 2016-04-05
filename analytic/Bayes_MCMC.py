#! /usr/bin/env python3

'''This module uses PyMC to fit parameters to the model via Bayesian inference.

Author: Christopher Strickland  
Email: cstrickland@samsi.info 
'''

__author__ = "Christopher Strickland"
__email__ = "cstrickland@samsi.info"
__status__ = "Development"
__version__ = "0.1"
__copyright__ = "Copyright 2015, Christopher Strickland"

import sys, os, time
from io import StringIO
from multiprocessing import Pool
import numpy as np
from scipy import sparse
import pymc as pm
from Run import Params
import ParasitoidModel as PM
from CalcSol import get_solutions


@pm.deterministic
def run_model(g_aw,g_bw,f_a1,f_b1,f_a2,f_b2,sig_x,sig_y,corr,lam,mu_r,ndays):
    '''This function acts as an interface between PyMC and the model.
    It takes in pymc variables, puts together a parameter object from them,
    runs the model, and then parses the result for comparison with emergence
    data. The idea is that we will compute the likelihood of the emergence data
    based on the assumption that this function's result is ground truth. The
    emergence data is then a realization of a stochastic variable based on
    the probability of observing x number of wasps given that the number of
    wasps present was as described in this function.
    '''
    
                    ########## SET PARAMETERS ##########
                    
    params = Params()
    # g wind function parameters
    params.g_params = (g_aw,g_bw)
    # f time of day function parameters
    params.f_params = (f_a1,f_b1,f_a2,f_b2)
    # Diffusion coefficients
    params.Dparams = (sig_x,sig_y,corr)
    # Probability of any flight during the day under ideal circumstances
    params.lam = lam
    
    # TRY BOTH - VARYING mu_r OR n_periods
    # scaling flight advection to wind advection
    params.mu_r = mu_r
    # number of time periods (based on interp_num) in one flight
    #params.n_periods = n_periods # if interp_num = 30, this is # of minutes
    
    # Set parameters specific to Bayesian runs
    params.PLOT = False
    params.OUTPUT = False
    params.ndays = ndays
    
                        ########## SETUP ##########
                        
    # This sends a message to CalcSol on whether or not to use CUDA
    if params.CUDA:
        globalvars.cuda = True
    else:
        globalvars.cuda = False
    # get wind data and day labels
    wind_data,days = PM.get_wind_data(*params.get_wind_params())
    
                        ########## RUN MODEL ##########
    
    ### PHASE ONE ###
    # First, get spread probability for each day as a coo sparse matrix
    pmf_list = []
    max_shape = np.array([0,0])
    print("Calculating each day's spread in parallel...")
    pm_args = [(days[0],wind_data,*params.get_model_params(),
            params.r_start)]
    pm_args.extend([(day,wind_data,*params.get_model_params()) 
            for day in days[1:ndays]])
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
    modelsol = get_populations(r_spread,pmf_list,days,ndays,dom_len,max_shape,
                                params.r_dur,params.r_number,params.r_mthd())
    
    # modelsol now holds the model results for this run
    
                    ########## PARSE SOLUTION ##########
                    
    pass
    
    
    
class Capturing(list):
    '''This class creates a list object that can be used in 'with' environments
    to capture the stdout of the enclosing functions. If used multiple times,
    it will extend itself to make a longer list containing everything.
    
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
    
    
    
def main(argv):
    '''Run MCMC on the model parameters. (Plot resulting distributions?)
    
    Priors and other stochastic model elements should be defined in here.'''
    pass
    
if __name__ == "__main__":
    main(sys.argv[1:])