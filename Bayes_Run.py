#! /usr/bin/env python3

'''This module uses PyMC to fit parameters to the model via Bayesian inference.

Author: Christopher Strickland
Email: wcstrick@live.unc.edu
'''

__author__ = "Christopher Strickland"
__email__ = "wcstrick@live.unc.edu"
__status__ = "Release"
__version__ = "1.0"
__copyright__ = "Copyright 2015, Christopher Strickland"

import warnings
import argparse
import sys, time, warnings
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

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--new", help='Start new MCMC run and exit on completion.',
    nargs=3,metavar=('iterations','burn-in','db_name'))
group.add_argument("--resume", help='Resume sampling.',nargs=2,
    metavar=('db_name','iterations'))

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

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

def main(mcmc_args=None):

    print('Setting up parameters and priors...')

    params = Params()
    # Set up location here with command line arguments in a list.
    params.cmd_line_chg(['--kalbar'])
    assert params.site_name+'fields.txt' == 'data/kalbarfields.txt'
    # Set parameters specific to Bayesian runs
    params.PLOT = False
    params.OUTPUT = False

    # This sends a message to CalcSol on whether or not to use CUDA
    if params.CUDA:
        globalvars.cuda = True
    else:
        globalvars.cuda = False
    # get wind data and day labels
    wind_data,days = PM.get_wind_data(*params.get_wind_params())
    params.ndays = len(days)

    # reduce domain
    params.domain_info = (10000.0,400) #25 m sided cells
    domain_res = params.domain_info[0]/params.domain_info[1]
    cell_area = domain_res**2

    locinfo = LocInfo(params.dataset,params.coord,params.domain_info)



    ######################################################################
    #####                        Model Priors                        #####
    ######################################################################
    lam = pm.Beta("lam",5,1,value=0.95)
    f_a1 = pm.TruncatedNormal("f_a1",6,0.3,0,9,value=6)
    f_a2 = pm.TruncatedNormal("f_a2",20,0.3,15,24,value=20)
    f_b1_p = pm.Gamma("fb1_p",2,1,value=1.5,trace=False,plot=False) #alpha,beta parameterization
    @pm.deterministic(trace=True,plot=True)
    def f_b1(f_b1_p=f_b1_p):
        return f_b1_p + 1
    f_b2_p = pm.Gamma("fb2_p",2,1,value=1.5,trace=False,plot=False)
    @pm.deterministic(trace=True,plot=True)
    def f_b2(f_b2_p=f_b2_p):
        return f_b2_p + 1
    g_aw = pm.Gamma("g_aw",2.2,1,value=1.0)
    g_bw = pm.Gamma("g_bw",5,1,value=3.8)
    # flight diffusion parameters. note: mean is average over flight advection
    sig_x = pm.Gamma("sig_x",26,0.15,value=180)
    sig_y = pm.Gamma("sig_y",15,0.15,value=150)
    corr_p = pm.Beta("corr_p",5,5,value=0.5,trace=False,plot=False)
    @pm.deterministic(trace=True,plot=True)
    def corr(corr_p=corr_p):
        return corr_p*2 - 1
    # local spread paramters
    sig_x_l = pm.Gamma("sig_xl",2,0.08,value=10)
    sig_y_l = pm.Gamma("sig_yl",2,0.14,value=10)
    corr_l_p = pm.Beta("corr_l_p",5,5,value=0.5,trace=False,plot=False)
    @pm.deterministic(trace=True,plot=True)
    def corr_l(corr_l_p=corr_l_p):
        return corr_l_p*2 - 1
    mu_r = pm.Normal("mu_r",1.,1,value=1)
    n_periods = pm.Poisson("n_periods",30,value=30)
    #alpha_pow = prev. time exponent in ParasitoidModel.h_flight_prob
    xi = pm.Gamma("xi",1,1,value=0.75) # presence to oviposition/emergence factor
    em_obs_prob = pm.Beta("em_obs_prob",1,1,value=0.05) # per-wasp prob of
            # observing emergence in release field grid given max leaf collection
            # this is dependent on the size of the cell surrounding the grid point
            # ...not much to be done about this.
    grid_obs_prob = pm.Beta("grid_obs_prob",1,1,value=0.005) # probability of
            # observing a wasp present in the grid cell given max leaf sampling

    #card_obs_prob = pm.Beta("card_obs_prob",1,1,value=0.5) # probability of
            # observing a wasp present in the grid cell given max leaf sampling

    #### Data collection model background for sentinel fields ####
    # Need to fix linear units for area. Meters would be best.
    # Effective collection area (constant between fields) is very uncertain
    with warnings.catch_warnings():
        # squelsh a warning based on pymc coding we don't need to worry about
        warnings.simplefilter("ignore",RuntimeWarning)
        A_collected = pm.TruncatedNormal("A_collected",2500,1/2500,0,
                                         min(locinfo.field_sizes.values())*
                                         cell_area,value=2500)  # in m**2
    # Each field has its own binomial probability.
    # Probabilities are likely to be small, and pm.Beta cannot handle small
    #   parameter values. So we will use TruncatedNormal again.
    N = len(locinfo.sent_ids)
    sent_obs_probs = np.empty(N, dtype=object)
    # fix beta for the Beta distribution
    sent_beta = 40
    # mean of Beta distribution will be A_collected/field size
    for n,key in enumerate(locinfo.sent_ids):
        sent_obs_probs[n] = pm.Beta("sent_obs_probs_{}".format(key),
            A_collected/(locinfo.field_sizes[key]*cell_area)*sent_beta/(
            1 - A_collected/(locinfo.field_sizes[key]*cell_area)),
            sent_beta, value=0.1*3600/(locinfo.field_sizes[key]*cell_area))

    sent_obs_probs = pm.Container(sent_obs_probs)

    # Max a Posterirori estimates have consistantly returned a value near zero
    #   for sprd_factor. So we will comment these sections.
    # if params.dataset == 'kalbar':
    #     # factor for kalbar initial spread
    #     sprd_factor = pm.Uniform("sprd_factor",0,1,value=0.3)
    # else:
    #     sprd_factor = None
    sprd_factor = None

    #### Collect variables and setup block update ####
    params_ary = pm.Container(np.array([g_aw,g_bw,f_a1,f_b1,f_a2,f_b2,
                                        sig_x,sig_y,corr,sig_x_l,sig_y_l,corr_l,
                                        lam,n_periods,mu_r],dtype=object))
    # The stochastic variables in this list (and the stochastics behind the
    #   deterministic ones) should be block updated in order to avoid the large
    #   computational expense of evaluating the model multiple times for each
    #   MCMC iteration. To do this, starting step variances must be definied
    #   for each variable. This is done via a scaling dict.
    stoc_vars = [g_aw,g_bw,f_a1,f_b1_p,f_a2,f_b2_p,sig_x,sig_y,corr_p,
        sig_x_l,sig_y_l,corr_l_p,lam,n_periods,mu_r]
    step_scales = {
        g_aw:0.04, g_bw:0.08,
        f_a1:0.25, f_b1_p:0.05, f_a2:0.25, f_b2_p:0.05,
        sig_x:2, sig_y:2, corr_p:0.0005,
        sig_x_l:2, sig_y_l:2, corr_l_p:0.0005,
        lam:0.0005,
        n_periods:1,
        mu_r:0.005
    }


    print('Getting initial model values...')

    ######################################################################
    #####                          Run Model                         #####
    ######################################################################
    @pm.deterministic(plot=False,trace=False)
    def pop_model(params=params,params_ary=params_ary,locinfo=locinfo,
                  wind_data=wind_data,days=days,sprd_factor=sprd_factor):
        '''This function acts as an interface between PyMC and the model.
        Not only does it run the model, but it provides an emergence potential
        based on the population model result projected forward from feasible
        oviposition dates. To modify how this projection happens, edit
        popdensity_to_emergence. Returned values from this function should be
        nearly ready to compare to data.
        '''
        modeltic = time.time()
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

        # TRY BOTH SCALINGS - VARYING mu_r and n_periods
        # scaling flight advection to wind advection
        # number of time periods (based on interp_num) in one flight
        params.n_periods = params_ary[13] # if interp_num = 30, this is # of minutes
        params.mu_r = params_ary[14]

        ### PHASE ONE ###
        # First, get spread probability for each day as a coo sparse matrix
        max_shape = np.array([0,0])
        pm_args = [(days[0],wind_data,*params.get_model_params(),
                params.r_start)]
        pm_args.extend([(day,wind_data,*params.get_model_params())
                for day in days[1:params.ndays]])

        ##### Kalbar wind started recording a day late. Spread the population
        #####   locally before running full model.
        if sprd_factor is not None:
            res = params.domain_info[0]/params.domain_info[1]
            mean_drift = np.array([-25.,15.])
            xdrift_int = int(mean_drift[0]//res)
            xdrift_r = mean_drift[0] % res
            ydrift_int = int(mean_drift[1]//res)
            ydrift_r = mean_drift[1] % res
            longsprd = PM.get_mvn_cdf_values(res,np.array([xdrift_r,ydrift_r]),
                        PM.Dmat(params_ary[6],params_ary[7],params_ary[8]))
            shrtsprd = PM.get_mvn_cdf_values(res,np.array([0.,0.]),
                        PM.Dmat(params_ary[9],params_ary[10],params_ary[11]))

            mlen = int(max(longsprd.shape[0],shrtsprd.shape[0]) +
                       max(abs(xdrift_int),abs(ydrift_int))*2)
            sprd = np.zeros((mlen,mlen))
            lbds = [int(mlen//2-longsprd.shape[0]//2),
                    int(mlen//2+longsprd.shape[0]//2+1)]
            sprd[lbds[0]-ydrift_int:lbds[1]-ydrift_int,
                 lbds[0]+xdrift_int:lbds[1]+xdrift_int] = longsprd*sprd_factor
            sbds = [int(mlen//2-shrtsprd.shape[0]//2),
                    int(mlen//2+shrtsprd.shape[0]//2+1)]
            sprd[sbds[0]:sbds[1],sbds[0]:sbds[1]] += shrtsprd*(1-sprd_factor)

            sprd[int(sprd.shape[0]//2),int(sprd.shape[0]//2)] += max(0,1-sprd.sum())
            pmf_list = [sparse.coo_matrix(sprd)]
        else:
            pmf_list = []

        ###################### Get pmf_list from multiprocessing
        pmf_list.extend(pool.starmap(PM.prob_mass,pm_args))

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
            if sprd_factor is not None:
                # extend day count by one
                days_ext = [days[0]-1]
                days_ext.extend(days)
                modelsol = get_populations(r_spread,pmf_list,days_ext,params.ndays+1,
                                           dom_len,max_shape,params.r_dur,
                                           params.r_number,params.r_mthd())
                # remove the first one and start where wind started.
                modelsol = modelsol[1:]
            else:
                modelsol = get_populations(r_spread,pmf_list,days,params.ndays,
                                           dom_len,max_shape,params.r_dur,
                                           params.r_number,params.r_mthd())

        # modelsol now holds the model results for this run as CSR sparse arrays

        # get emergence potential (measured in expected number of wasps previously
        #   present whose oviposition would result in emergence on the given date)
        #   from the model result
        release_emerg,sentinel_emerg = popdensity_to_emergence(modelsol,locinfo)

        # get the expected wasp populations at grid points on sample days
        grid_counts = popdensity_grid(modelsol,locinfo)

        # get the expected wasp populations in cardinal directions
        '''card_counts = popdensity_card(modelsol,locinfo,params.domain_info)'''

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
        # print('{:03.1f} sec./model at {}'.format(time.time() - modeltic,
        #     time.strftime("%H:%M:%S %d/%m/%Y")),end='\r')
        # sys.stdout.flush()
        return (release_emerg,sentinel_emerg,grid_counts) #,card_counts)

    print('Parsing model output and connecting to Bayesian model...')

    ######################################################################
    #####                   Connect Model to Data                    #####
    ######################################################################

    ### Parse the results of pop_model into separate deterministic variables ###
    '''Get Poisson probabilities for sentinal field emergence. Parameters:
        xi is constant, emerg is a list of ndarrays, betas is a 1D array of
        field probabilities'''
    Ncollections = len(locinfo.sent_DataFrames)
    sent_poi_rates = []
    for ii in range(Ncollections):
        s_ndays = len(locinfo.sent_DataFrames[ii]['datePR'].unique())
        sent_poi_rates.append(pm.Lambda('sent_poi_rate_{}'.format(ii),
            lambda xi=xi, ndays=s_ndays, betas=sent_obs_probs,
            emerg_model=pop_model[1][ii]:
            xi*emerg_model*np.tile(betas,(ndays,1)).T,trace=False))
    sent_poi_rates = pm.Container(sent_poi_rates)

    '''Return Poisson probabilities for release field grid emergence. Parameters:
        xi is constant, emerg is a list of ndarrays. collection effort is
        specified in locinfo.'''
    Ncollections = len(locinfo.release_DataFrames)
    rel_poi_rates = []
    for ii in range(Ncollections):
        r_effort = locinfo.release_collection[ii] #fraction of max collection
        r_ndays = len(locinfo.release_DataFrames[ii]['datePR'].unique())
        rel_poi_rates.append(pm.Lambda('rel_poi_rate_{}'.format(ii),
            lambda xi=xi, ndays=r_ndays, r_effort=r_effort, beta=em_obs_prob,
            emerg_model=pop_model[0][ii]:
            xi*emerg_model*np.tile(r_effort*beta,(ndays,1)).T,trace=False))
    rel_poi_rates = pm.Container(rel_poi_rates)

    @pm.deterministic(plot=False,trace=False)
    def grid_poi_rates(locinfo=locinfo,beta=grid_obs_prob,
                        obs_model=pop_model[2]):
        '''Return Poisson probabilities for grid sampling
        obs_model is an ndarray, sampling effort is specified in locinfo.'''
        return beta*locinfo.grid_samples*obs_model

    '''Return Poisson probabilities for cardinal direction sampling
        obs_model is a list of ndarrays, sampling effort is assumed constant'''
    '''
    card_poi_rates = []
    for ii,obs in enumerate(pop_model[3]):
        card_poi_rates.append(pm.Lambda('card_poi_rate_{}'.format(ii),
            lambda beta=card_obs_prob, obs=obs: beta*obs))
    card_poi_rates = pm.Container(card_poi_rates)
    '''

    # Given the expected wasp densities from pop_model, actual wasp densities
    #   are modeled as a thinned Poisson random variable about that mean.
    # Each wasp in the area then has a small probability of being seen.

    ### Connect sentinel emergence data to model ###
    N_sent_collections = len(locinfo.sent_DataFrames)
    # Create list of collection variables
    sent_collections = []
    for ii in range(N_sent_collections):
        # Apparently, pymc does not play well with 2D array parameters
        sent_collections.append(np.empty(sent_poi_rates[ii].value.shape,
                                         dtype=object))
        for n in range(sent_collections[ii].shape[0]):
            for m in range(sent_collections[ii].shape[1]):
                sent_collections[ii][n,m] = pm.Poisson(
                    "sent_em_obs_{}_{}_{}".format(ii,n,m),
                    sent_poi_rates[ii][n,m],
                    value=float(locinfo.sentinel_emerg[ii][n,m]),
                    observed=True)
    sent_collections = pm.Container(sent_collections)

    ### Connect release-field emergence data to model ###
    N_release_collections = len(locinfo.release_DataFrames)
    # Create list of collection variables
    rel_collections = []
    for ii in range(N_release_collections):
        rel_collections.append(np.empty(rel_poi_rates[ii].value.shape,
                                        dtype=object))
        for n in range(rel_collections[ii].shape[0]):
            for m in range(rel_collections[ii].shape[1]):
                rel_collections[ii][n,m] = pm.Poisson(
                    "rel_em_obs_{}_{}_{}".format(ii,n,m),
                    rel_poi_rates[ii][n,m],
                    value=float(locinfo.release_emerg[ii][n,m]),
                    observed=True)
    rel_collections = pm.Container(rel_collections)

    ### Connect grid sampling data to model ###
    grid_obs = np.empty(grid_poi_rates.value.shape,dtype=object)
    for n in range(grid_obs.shape[0]):
        for m in range(grid_obs.shape[1]):
            grid_obs[n,m] = pm.Poisson("grid_obs_{}_{}".format(n,m),
                grid_poi_rates[n,m], value=float(locinfo.grid_obs[n,m]),
                observed=True)
    grid_obs = pm.Container(grid_obs)

    ### Connect cardinal direction data to model ###
    '''
    N_card_collections = len(locinfo.card_obs_DataFrames)
    # Create list of sampling variables
    card_collections = []
    for ii in range(N_card_collections):
        card_collections.append(np.empty(card_poi_rates[ii].value.shape,
                                         dtype=object))
        for n in range(card_collections[ii].shape[0]):
            for m in range(card_collections[ii].shape[1]):
                card_collections[ii][n,m] = pm.Poisson(
                    "card_obs_{}_{}_{}".format(ii,n,m),
                    card_poi_rates[ii][n,m],
                    value=locinfo.card_obs[ii][n,m],
                    observed=True, plot=False)
    card_collections = pm.Container(card_collections)
    '''

    ######################################################################
    #####                   Collect Model and Run                    #####
    ######################################################################

    ### Collect model ###
    if sprd_factor is not None:
        Bayes_model = pm.Model([lam,f_a1,f_a2,f_b1_p,f_b2_p,f_b1,f_b2,g_aw,g_bw,
                                sig_x,sig_y,corr_p,corr,sig_x_l,sig_y_l,
                                corr_l_p,corr_l,n_periods,mu_r,sprd_factor,
                                grid_obs_prob,xi,em_obs_prob,A_collected,
                                sent_obs_probs,params_ary,pop_model,
                                grid_poi_rates,rel_poi_rates,sent_poi_rates,
                                grid_obs,rel_collections,sent_collections])
    else:
        Bayes_model = pm.Model([lam,f_a1,f_a2,f_b1_p,f_b2_p,f_b1,f_b2,g_aw,g_bw,
                                sig_x,sig_y,corr_p,corr,sig_x_l,sig_y_l,
                                corr_l_p,corr_l,n_periods,mu_r,
                                grid_obs_prob,xi,em_obs_prob,A_collected,
                                sent_obs_probs,params_ary,pop_model,
                                grid_poi_rates,rel_poi_rates,sent_poi_rates,
                                grid_obs,rel_collections,sent_collections])

    ### Run if parameters were passed in ###
    if mcmc_args is not None:
        if len(mcmc_args) == 3:
            # New run
            nsamples = int(mcmc_args[0])
            burn = int(mcmc_args[1])
            fname = mcmc_args[2]
            if fname[-3:] != '.h5':
                fname += '.h5'
            mcmc = pm.MCMC(Bayes_model,db='hdf5',dbname=fname,
                        dbmode='a',dbcomplevel=0)
            mcmc.use_step_method(pm.AdaptiveMetropolis,stoc_vars,
                scales=step_scales,interval=500,shrink_if_necessary=True)
            try:
                tic = time.time()
                print('Sampling...')
                mcmc.sample(nsamples,burn)
                # sampling finished. commit to database and continue
                print('Sampling finished.')
                print('Time elapsed: {}'.format(time.time()-tic))
                print('Saving...')
                #mcmc.save_state()
                mcmc.commit()
                print('Closing...')
                mcmc.db.close()
            except:
                print('Exception: database closing...')
                mcmc.db.close()
                raise
            return
        elif len(mcmc_args) == 2:
            # Resume run
            fname = mcmc_args[0]
            nsamples = int(mcmc_args[1])
            fname = fname.strip()
            if fname[-3:] != '.h5':
                fname += '.h5'
            if os.path.isfile(fname):
                db = pm.database.hdf5.load(fname)
                mcmc = pm.MCMC(Bayes_model,db=db)
                mcmc.use_step_method(pm.AdaptiveMetropolis,stoc_vars,
                    scales=step_scales,interval=500,shrink_if_necessary=True)
                # database loaded.
            else:
                print('File not found: {}'.format(fname))
                return
            try:
                tic = time.time()
                print('Sampling...')
                mcmc.sample(nsamples)
                # sampling finished. commit to database and continue
                print('Sampling finished.')
                print('Time elapsed: {}'.format(time.time()-tic))
                print('Saving...')
                #mcmc.save_state()
                mcmc.commit()
                print('Closing...')
                mcmc.db.close()
            except:
                print('Exception: database closing...')
                mcmc.db.close()
                raise
            return


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
        mcmc.use_step_method(pm.AdaptiveMetropolis,stoc_vars,scales=step_scales,
            interval=500,shrink_if_necessary=True)
        try:
            tic = time.time()
            print('Sampling...')
            mcmc.sample(nsamples,burn)
            # sampling finished. commit to database and continue
            print('Sampling finished.')
            print('Time elapsed: {}'.format(time.time()-tic))
            print('Saving...')
            #mcmc.save_state()
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
                    mcmc.use_step_method(pm.AdaptiveMetropolis,stoc_vars,
                        scales=step_scales,interval=500,
                        shrink_if_necessary=True)
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
        print(" 'report': generate report on traces")
        print("'inspect': launch IPython to inspect state")
        print("    'run': conduct further sampling")
        print("   'quit': Quit")
        cmd = input('Enter: ')
        cmd = cmd.strip()
        cmd = cmd.lower()
        if cmd == 'inspect':
            try:
                import IPython
                IPython.embed()
            except ImportError:
                print('IPython not found.')
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
                tic = time.time()
                print('Sampling...')
                mcmc.sample(nsamples)
                # sampling finished. commit to database and continue
                print('Sampling finished.')
                print('Time elapsed: {}'.format(time.time()-tic))
                print('Saving...')
                #mcmc.save_state()
                mcmc.commit()
            except:
                print('Exception: database closing...')
                mcmc.db.close()
                raise
        elif cmd == 'report':
            try:
                import Bayes_Plot
                Bayes_Plot.plot_traces(db=db)
                print('Gelman-Rubin statistics')
                gr = pm.gelman_rubin(mcmc)
                print(gr)
                with open('./diagnostics/gelman-rubin.txt','w') as f:
                    f.write('Variable        R_hat\n')
                    f.write('---------------------\n')
                    for key,val in gr.items():
                        f.write(key+': {}\n'.format(val))
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
    args = parser.parse_args()
    with Pool() as pool:
        if args.new is not None:
            main(args.new)
        elif args.resume is not None:
            main(args.resume)
        else:
            main()
