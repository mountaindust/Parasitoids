#! /usr/bin/env python3

'''This module uses PyMC to fit parameters to the model via Bayesian inference.

Author: Christopher Strickland  
Email: cstrickland@samsi.info 
'''

__author__ = "Christopher Strickland"
__email__ = "cstrickland@samsi.info"
__status__ = "Development"
__version__ = "0.3"
__copyright__ = "Copyright 2015, Christopher Strickland"

import sys, time
from io import StringIO
import os.path
import numpy as np
from scipy import sparse
from multiprocessing import Pool
import pymc as pm
import globalvars
from Run import Params
from Data_Import import LocInfo
from CalcSol import get_populations
import ParasitoidModel as PM
from Bayes_funcs import *
import IPython



class Capturing(list):
    '''This class creates a list object that can be used in 'with' environments
    to capture the stdout of the enclosing functions. If used multiple times,
    it can extend itself to make a longer list containing everything.
    
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



###############################################################################
#                                                                             #
#                             PYMC Setup & Run                                #
#                                                                             #
###############################################################################

def main():

    params = Params()
    # Set up location here with command line arguments in a list.
    params.cmd_line_chg(['--kalbar'])
    assert params.site_name+'fields.txt' == 'data/kalbarfields.txt'
    # Set parameters specific to Bayesian runs
    params.PLOT = False
    params.OUTPUT = False
    params.ndays = -1 # locinfo.get_landscape_sample_datesPR()[-1]?

    # This sends a message to CalcSol on whether or not to use CUDA
    if params.CUDA:
        globalvars.cuda = True
    else:
        globalvars.cuda = False
    # get wind data and day labels
    wind_data,days = PM.get_wind_data(*params.get_wind_params())

    locinfo = LocInfo(params.dataset,params.coord,params.domain_info)
    
    
    
    #### Model priors ####
    lam = pm.Beta("lambda",5,1,value=0.95)
    f_a1 = pm.TruncatedNormal("a_1",6,1,0,12,value=6)
    f_a2 = pm.TruncatedNormal("a_2",18,1,12,24,value=18)
    f_b1 = pm.Gamma("b_1",3,1,value=3) #alpha,beta parameterization
    f_b2 = pm.Gamma("b_2",3,1,value=3)
    g_aw = pm.Gamma("a_w",2.2,1,value=2.2)
    g_bw = pm.Gamma("b_w",5,1,value=5)
    sig_x = pm.Gamma("sig_x",42.2,2,value=21.1)
    sig_y = pm.Gamma("sig_y",10.6,1,value=10.6)
    corr = pm.Uniform("rho",-1,1,value=0)
    sig_x_l = pm.Gamma("sig_x",42.2,2,value=21.1)
    sig_y_l = pm.Gamma("sig_y",10.6,1,value=10.6)
    corr_l = pm.Uniform("rho",-1,1,value=0)
    mu_r = pm.Normal("mu_r",1.5,1/0.75**2,value=1.5)
    #n_periods = pm.Poisson("t_dur",10)
    #r = prev. time exponent
    xi = pm.Gamma("xi",1,1,value=1) # presence to oviposition/emergence factor
    em_obs_prob = pm.Beta("em_obs_prob",1,1,value=0.5) # obs prob of emergence 
                                # in release field given max leaf collection
    grid_obs_prob = pm.Beta("grid_obs_prob",1,1,value=0.5) # probability of
            # observing a wasp present in the grid cell given max leaf sampling
    card_obs_prob = pm.Beta("card_obs_prob",1,1,value=0.5) # probability of
            # observing a wasp present in the grid cell given max leaf sampling
    
    #### Data collection model background for sentinel fields ####
    # Need to fix linear units for area. Let's use cells.
    # Effective collection area (constant between fields) is very uncertain
    A_collected = pm.TruncatedNormal("A_collected",25,1/50,0,
                                    min(locinfo.field_sizes.values()),value=16)  
    # Each field has its own binomial probability.
    ## Probabilities are likely to be small, and pm.Beta cannot handle small
    ##  parameter values. So we will use TruncatedNormal again.
    N = len(locinfo.sent_ids)
    sent_obs_probs = np.empty(N, dtype=object)
    for n,key in enumerate(locinfo.sent_ids):
        sent_obs_probs[n] = pm.TruncatedNormal("sent_obs_prob_{}".format(key),
            A_collected/locinfo.field_sizes[key],0.05,0,1)
    
    #### Collect variables ####
    params_ary = np.array([g_aw,g_bw,f_a1,f_b1,f_a2,f_b2,sig_x,sig_y,corr,
                            sig_x_l,sig_y_l,corr_l,lam,mu_r],dtype=object)

    #### Run model ####
    @pm.deterministic(plot=False,trace=False)
    def pop_model(params=params,params_ary=params_ary,locinfo=locinfo,
                  wind_data=wind_data,days=days):
        '''This function acts as an interface between PyMC and the model.
        Not only does it run the model, but it provides an emergence potential 
        based on the population model result projected forward from feasible
        oviposition dates. To modify how this projection happens, edit 
        popdensity_to_emergence. Returned values from this function should be
        nearly ready to compare to data.
        '''
        ### Alter params with stochastic variables ###

        # g wind function parameters
        params.g_params = tuple(params_ary[0:2])
        # f time of day function parameters
        params.f_params = tuple(params_ary[2:6])
        # Diffusion coefficients
        params.Dparams = tuple(params_ary[6:9])
        params.Dlparams = tuple(params_ary[9:12])
        # Probability of any flight during the day under ideal circumstances
        params.lam = params_ary[12]
        
        # TRY BOTH - VARYING mu_r OR n_periods
        # scaling flight advection to wind advection
        params.mu_r = params_ary[13]
        # number of time periods (based on interp_num) in one flight
        #params.n_periods = n_periods # if interp_num = 30, this is # of minutes

        
        ### PHASE ONE ###
        # First, get spread probability for each day as a coo sparse matrix
        pmf_list = []
        max_shape = np.array([0,0])
        pm_args = [(days[0],wind_data,*params.get_model_params(),
                params.r_start)]
        pm_args.extend([(day,wind_data,*params.get_model_params()) 
                for day in days[1:params.ndays]])
    
        ###################### Get pmf_list from multiprocessing
        pool = Pool()
        try:
            pmf_list = pool.starmap(PM.prob_mass,pm_args)
        except PM.BndsError as e:
            # return output full of zeros, but of correct type/size
            release_emerg = []
            for nframe,dframe in enumerate(locinfo.release_DataFrames):
                obs_datesPR = dframe['datePR'].map(lambda t: t.days).unique()
                release_emerg.append(np.zeros((len(locinfo.emerg_grids[nframe]),
                                        len(obs_datesPR))))
            sentinel_emerg = []
            for nframe,dframe in enumerate(locinfo.sent_DataFrames):
                obs_datesPR = dframe['datePR'].map(lambda t: t.days).unique()
                sentinel_emerg.append(np.zeros((len(locinfo.sent_ids),
                                        len(obs_datesPR))))
            grid_counts = np.zeros((locinfo.grid_cells.shape[0],
                                    len(locinfo.grid_obs_datesPR)))
            card_counts = []
            for ndate,date in enumerate(locinfo.card_obs_datesPR):
                card_counts.append(np.zeros((4,locinfo.card_obs[nday].shape[1])))
            return (release_emerg,sentinel_emerg,grid_counts,card_counts)
        finally:
            pool.close()
            pool.join()
        ######################
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
        with Capturing() as output:
            modelsol = get_populations(r_spread,pmf_list,days,params.ndays,dom_len,
                        max_shape,params.r_dur,params.r_number,params.r_mthd())
        
        # modelsol now holds the model results for this run as CSR sparse arrays
        
        # get emergence potential (measured in expected number of wasps previously
        #   present whose oviposition would result in emergence on the given date)
        #   from the model result
        release_emerg,sentinel_emerg = popdensity_to_emergence(modelsol,locinfo)

        # get the expected wasp populations at grid points on sample days
        grid_counts = popdensity_grid(modelsol,locinfo)

        # get the expected wasp populations in cardinal directions
        card_counts = popdensity_card(modelsol,locinfo,params.domain_info)
        
        ## For the lists release_emerg and sentinel_emerg:
        ##    Each list entry corresponds to a data collection day (one array)
        ##    In each array:
        ##    Each column corresponds to an emergence observation day (as in data)
        ##    Each row corresponds to a grid point or sentinel field, respectively
        ## For the array grid_counts:
        ##    Each column corresponds to an observation day
        ##    Each row corresponds to a grid point
        ## For the list card_counts:
        ##    Each list entry corresponds to a sampling day (one array)
        ##    Each column corresponds to a step in a cardinal direction
        ##    Each row corresponds to a cardinal direction 
        return (release_emerg,sentinel_emerg,grid_counts,card_counts)
        
        
        
    ### Parse the results of pop_model into separate deterministic variables ###
    @pm.deterministic(plot=False)
    def sent_poi_rates(locinfo=locinfo,xi=xi,betas=sent_obs_probs,
                        emerg_model=pop_model[1]):
        '''Return Poisson probabilities for sentinal field emergence
        xi is constant, emerg is a list of ndarrays, betas is a 1D array of
        field probabilities'''
        Ncollections = len(locinfo.sent_DataFrames)
        poi_rates = []
        for ii in range(Ncollections):
            ndays = len(locinfo.sent_DataFrames[ii]['datePR'].unique())
            tile_betas = np.tile(betas,(ndays,1)).T
            poi_rates.append(xi*emerg_model[ii]*tile_betas)
        return poi_rates
        
    @pm.deterministic(plot=False)
    def rel_poi_rates(locinfo=locinfo,xi=xi,beta=em_obs_prob,
                        emerg_model=pop_model[0]):
        '''Return Poisson probabilities for release field grid emergence
        xi is constant, emerg is a list of ndarrays. collection effort is
        specified in locinfo.'''
        Ncollections = len(locinfo.release_DataFrames)
        poi_rates = []
        for ii in range(Ncollections):
            r_effort = locinfo.release_collection[ii] #fraction of max collection
            ndays = len(locinfo.release_DataFrames[ii]['datePR'].unique())
            tile_betas = np.tile(r_effort*beta,(ndays,1)).T
            poi_rates.append(xi*emerg_model[ii]*tile_betas)
        return poi_rates
            
    @pm.deterministic(plot=False)
    def grid_poi_rates(locinfo=locinfo,beta=grid_obs_prob,
                        obs_model=pop_model[2]):
        '''Return Poisson probabilities for grid sampling
        obs_model is an ndarray, sampling effort is specified in locinfo.'''
        return beta*locinfo.grid_samples*obs_model

    @pm.deterministic(plot=False)
    def card_poi_rates(beta=card_obs_prob,obs_model=pop_model[3]):
        '''Return Poisson probabilities for cardinal direction sampling
        obs_model is a list of ndarrays, sampling effort is assumed constant'''
        poi_rates = []
        for obs in obs_model:
            poi_rates.append(beta*obs)
        return poi_rates

    
    # Given the expected wasp densities from pop_model, actual wasp densities
    #   are modeled as a thinned Poisson random variable about that mean.
    # Each wasp in the area then has a small probability of being seen.
    
    ### Connect sentinel emergence data to model ###
    N_sent_collections = len(locinfo.sent_DataFrames)
    # Create list of collection variables
    sent_collections = []
    for ii in range(N_sent_collections):
        sent_collections.append(pm.Poisson("sent_em_obs_{}".format(ii),
            sent_poi_rates[ii], value=locinfo.sentinel_emerg[ii], observed=True))
            
    ### Connect release-field emergence data to model ###
    N_release_collections = len(locinfo.release_DataFrames)
    # Create list of collection variables
    rel_collections = []
    for ii in range(N_release_collections):
        rel_collections.append(pm.Poisson("rel_em_obs_{}".format(ii),
            rel_poi_rates[ii], value=locinfo.release_emerg[ii], observed=True))

    ### Connect grid sampling data to model ###
    grid_obs = pm.Poisson("grid_obs",grid_poi_rates,value=locinfo.grid_obs,
                            observed=True)

    ### Connect cardinal direction data to model ###
    N_card_collections = len(locinfo.card_obs_DataFrames)
    # Create list of sampling variables
    card_collections = []
    for ii in range(N_card_collections):
        card_collections.append(pm.Poisson("card_obs_{}".format(ii),
            card_poi_rates[ii], value=locinfo.card_obs[ii], observed=True))

    ### Collect model ###
    Bayes_model = pm.Model([lam,f_a1,f_a2,f_b1,f_b2,g_aw,g_bw,
                            sig_x,sig_y,corr,sig_x_l,sig_y_l,corr_l,mu_r,
                            card_obs_prob,grid_obs_prob,xi,em_obs_prob,
                            A_collected,sent_obs_probs,params_ary,pop_model,
                            card_poi_rates,grid_poi_rates,rel_poi_rates,sent_poi_rates,
                            card_collections,grid_obs,rel_collections,sent_collections])


    ######################################################################
    #####                   Start Interactive Menu                   #####
    ######################################################################
    print('--------------- MCMC MAIN MENU ---------------')
    print(" 'new': Start a new MCMC chain from the beginning.")
    print("'cont': Continue a previous MCMC chain from an hdf5 file.")
    #print("'plot': Plot traces/distribution from an hdf5 file.")
    print("'quit': Quit.")
    cmd = input('Enter: ')
    cmd = cmd.strip().lower()
    if cmd == 'new':
        print('\n\n')
        print('--------------- New MCMC Chain ---------------')
        while True:
            val = input("Enter number of realizations or 'quit' to quit:")
            val = val.strip()
            if val == 'q' or val == 'quit':
                return
            else:
                try:
                    nsamples = int(val)
                    val2 = input("Enter number of realizations to discard:")
                    val2 = val2.strip()
                    if val2 == 'q' or val2 == 'quit':
                        return
                    else:
                        burn = int(val2)
                    fname = input("Enter filename to save or 'back' to cancel:")
                    fname = fname.strip()
                    if fname == 'q' or fname == 'quit':
                        return
                    elif fname == 'b' or fname == 'back':
                        continue
                    else:
                        fname = fname+'.h5'
                        break # BREAK LOOP AND RUN MCMC WITH GIVEN VALUES
                except ValueError:
                    print('Unrecognized input.')
                    continue
        ##### RUN FIRST MCMC HERE #####
        mcmc = pm.MCMC(Bayes_model,db='hdf5',dbname=fname,
                        dbmode='a',dbcomplevel=0)
        try:
            mcmc.sample(nsamples,burn)
            # sampling finished. commit to database and continue
            mcmc.save_state()
            mcmc.commit()
        except:
            print('Exception: database closing...')
            mcmc.db.close()
            raise

    elif cmd == 'cont':
        # Load db and continue
        print('\n')
        while True:
            fname = input("Enter path to database to load, or 'q' to quit:")
            fname = fname.strip()
            if fname.lower() == 'q' or fname.lower() == 'quit':
                return
            else:
                if fname[-3:] != '.h5':
                    fname += '.h5'
                if os.path.isfile(fname):
                    db = pm.database.hdf5.load(fname)
                    mcmc = pm.MCMC(Bayes_model,db=db)
                    break # database loaded
                else:
                    print('File not found.')
                    #continue

    elif cmd == 'plot':
        # Get filename and pass to plotting routine.
        pass
        # return
    elif cmd == 'quit' or cmd == 'q':
        return
    else:
        print('Command not recognized.')
        print('Quitting....')
        return
        
    ##### MCMC Loop #####
    # This should be reached only by cmd == 'new' or 'cont' with a database.
    # It resumes sampling of a previously sampled chain.
    print('\n')
    while True:
        print('--------------- MCMC ---------------')
        print("'inspect': launch IPython to inspect state")
        print("    'run': conduct further sampling")
        print("   'quit': Quit")
        cmd = input('Enter: ')
        cmd = cmd.strip().lower()
        if cmd == 'inspect':
            try:
                IPython.embed()
            except:
                print('Exception: database closing...')
                mcmc.db.close()
                raise
        elif cmd == 'run':
            val = input("Enter number of realizations or 'back':")
            val = val.strip()
            if val == 'back' or val == 'b':
                continue
            else:
                try:
                    nsamples = int(val)
                except ValueError:
                    print('Unrecognized input.')
                    continue
            # Run chain
            try:
                mcmc.sample(nsamples)
                # sampling finished. commit to database and continue
                mcmc.save_state()
                mcmc.commit()
            except:
                print('Exception: database closing...')
                mcmc.db.close()
                raise
        elif cmd == 'quit' or cmd == 'q':
            mcmc.db.close()
            print('Database closed.')
            break
        else:
            print('Command not recognized.')
    
if __name__ == "__main__":
    main()
    #main(sys.argv[1:])