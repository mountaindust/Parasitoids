#! /usr/bin/env python3

'''This module uses PyMC to fit parameters to the model via Bayesian inference.

Author: Christopher Strickland  
Email: cstrickland@samsi.info 
'''

__author__ = "Christopher Strickland"
__email__ = "cstrickland@samsi.info"
__status__ = "Development"
__version__ = "0.5"
__copyright__ = "Copyright 2015, Christopher Strickland"

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
group.add_argument("--MAP", help="Find max a posteriori estimate"+
                   " and exit on completion.", action="store_true")
group.add_argument("--norm", help="Find normal approximation",
                   metavar="database_name")
# Normal approximation - we would like to explore covarience?
# We need to add something here...

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

def main(RUNFLAG):

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
    params.domain_info = (5000.0,200) #25 m sided cells
    domain_res = params.domain_info[0]/params.domain_info[1]
    cell_area = domain_res**2

    locinfo = LocInfo(params.dataset,params.coord,params.domain_info)
    
    prior_eps = {}
    
    #### Model priors ####
    lam = pm.Beta("lam",5,1,value=0.95)
    prior_eps[lam] = 0.01
    f_a1 = pm.TruncatedNormal("a_1",6,1,0,9,value=6)
    prior_eps[f_a1] = 0.05
    f_a2 = pm.TruncatedNormal("a_2",18,1,15,24,value=18)
    prior_eps[f_a2] = 0.05
    f_b1_p = pm.Gamma("fb1_p",2,1,value=2,trace=False,plot=False) #alpha,beta parameterization
    prior_eps[f_b1_p] = 0.05
    @pm.deterministic(trace=True,plot=True)
    def f_b1(f_b1_p=f_b1_p): 
        return f_b1_p + 1
    f_b2_p = pm.Gamma("fb2_p",2,1,value=2,trace=False,plot=False)
    prior_eps[f_b2_p] = 0.05
    @pm.deterministic(trace=True,plot=True)
    def f_b2(f_b2_p=f_b2_p):
        return f_b2_p + 1
    g_aw = pm.Gamma("a_w",2.2,1,value=2.2)
    prior_eps[g_aw] = 0.05
    g_bw = pm.Gamma("b_w",5,1,value=5)
    prior_eps[g_bw] = 0.1
    # flight diffusion parameters. note: mean is average over flight advection
    sig_x = pm.Gamma("sig_x",26,.15,value=211.) #32.4
    prior_eps[sig_x] = 1 #0.1
    sig_y = pm.Gamma("sig_y",15,.15,value=106.) #16.2
    prior_eps[sig_y] = 1 #0.1
    corr_p = pm.Beta("rho_p",5,5,value=0.5,trace=False,plot=False)
    prior_eps[corr_p] = 0.01
    @pm.deterministic(trace=True,plot=True)
    def corr(corr_p=corr_p):
        return corr_p*2 - 1
    # local spread paramters
    sig_x_l = pm.Gamma("sig_xl",3,0.04,value=21.) #32.4
    prior_eps[sig_x_l] = 1 #0.1
    sig_y_l = pm.Gamma("sig_yl",5,0.10,value=16.) #16.2
    prior_eps[sig_y_l] = 1 #0.1
    corr_l_p = pm.Beta("rho_l_p",5,5,value=0.5,trace=False,plot=False)
    prior_eps[corr_l_p] = 0.01
    @pm.deterministic(trace=True,plot=True)
    def corr_l(corr_l_p=corr_l_p):
        return corr_l_p*2 - 1
    #pymc.MAP can only take float values, so we vary mu_r and set n_periods.
    mu_r = pm.Normal("mu_r",1.,1,value=1.)
    prior_eps[mu_r] = 0.05
    params.n_periods = 30
    #alpha_pow = prev. time exponent in ParasitoidModel.h_flight_prob
    xi = pm.Gamma("xi",1,1,value=1) # presence to oviposition/emergence factor
    prior_eps[xi] = 0.05
    
    #### Observation probabilities ####
    # Cut-off at 0.1 to reduce computation time
    em_obs_prob_p = pm.Beta("em_obs_prob_p",1,1,value=0.25,
        trace=False,plot=False) # per-wasp prob of observing emergence in
            # release field grid given max leaf collection.
            # This is dependent on the size of the cell surrounding the grid point,
            # but there's not much to be done about this. Just remember to
            # interpret this number based on grid coarseness.
    prior_eps[em_obs_prob_p] = 0.005
    @pm.deterministic(trace=True,plot=True)
    def em_obs_prob(em_obs_prob_p=em_obs_prob_p):
        return em_obs_prob_p*0.1
    grid_obs_prob_p = pm.Beta("grid_obs_prob_p",1,1,value=0.25,
        trace=False,plot=False) # probability of observing a wasp present in
        # the grid cell given max leaf sampling
    prior_eps[grid_obs_prob_p] = 0.005
    @pm.deterministic(trace=True,plot=True)
    def grid_obs_prob(grid_obs_prob_p=grid_obs_prob_p):
        return grid_obs_prob_p*0.1

    #card_obs_prob = pm.Beta("card_obs_prob",1,1,value=0.5) # probability of
            # observing a wasp present in the grid cell given max leaf sampling
    
    #### Data collection model background for sentinel fields ####
    # Need to fix linear units for area. Meters would be best.
    # Effective collection area (constant between fields) is very uncertain
    A_collected = pm.TruncatedNormal("A_collected",2500,1/2500,0,
                                    min(locinfo.field_sizes.values())*cell_area,
                                    value=3600)  # in m**2
    prior_eps[A_collected] = 10
    # Each field has its own binomial probability.
    # Probabilities are likely to be small, and pm.Beta cannot handle small
    #   parameter values. So we will use TruncatedNormal again.
    N = len(locinfo.sent_ids)
    sent_obs_probs = np.empty(N, dtype=object)
    sent_obs_probs_p = np.empty(N, dtype=object)
    # fix beta for the Beta distribution
    sent_beta = 10
    # mean of Beta distribution will be A_collected/field size
    ## Create function factory ##
    def make_f(input_prior):
        def f(prior=input_prior):
            return prior*0.1
        return f
    ## Loop over fields ##
    for n,key in enumerate(locinfo.sent_ids):
        sent_obs_probs_p[n] = pm.Beta("sent_obs_probs_p_{}".format(key),
            A_collected/(locinfo.field_sizes[key]*cell_area)*sent_beta/(
            1 - A_collected/(locinfo.field_sizes[key]*cell_area)),
            sent_beta, value=3600/(locinfo.field_sizes[key]*cell_area),
            trace=False,plot=False)
        prior_eps[sent_obs_probs_p[n]] = 0.005
        # get function from function factory
        sent_obs_probs[n] = pm.Deterministic(eval = make_f(sent_obs_probs_p[n]),
            name = "sent_obs_probs_{}".format(key),
            parents = {'prior':sent_obs_probs_p[n]},
            doc = "Probability of observing a given wasp emergence",
            trace=True,plot=False)
    
    sent_obs_probs_p = pm.Container(sent_obs_probs_p)        
    sent_obs_probs = pm.Container(sent_obs_probs)
    
    #### Collect variables ####
    params_ary = pm.Container(np.array([g_aw,g_bw,f_a1,f_b1,f_a2,f_b2,
                                        sig_x,sig_y,corr,sig_x_l,sig_y_l,corr_l,
                                        lam,mu_r],dtype=object))

    if params.dataset == 'kalbar':
        # factor for kalbar initial spread
        sprd_factor = pm.Uniform("sprd_factor",0,1,value=0.3)
        prior_eps[sprd_factor] = 0.01
    else:
        sprd_factor = None

    print('Getting initial model values...')

    #### Run model ####
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
        
        # scaling flight advection to wind advection
        params.mu_r = params_ary[13]

        
        ### PHASE ONE ###
        # First, get spread probability for each day as a coo sparse matrix
        max_shape = np.array([0,0])
        pm_args = [(days[0],wind_data,*params.get_model_params(),
                params.r_start)]
        pm_args.extend([(day,wind_data,*params.get_model_params()) 
                for day in days[1:params.ndays]])
        
        ##### Kalbar wind started recording a day late. Spread the population
        #####   locally before running full model.
        if params.dataset == 'kalbar':
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
            '''
            pmf_list = [sparse.coo_matrix(PM.get_mvn_cdf_values(
                        params.domain_info[0]/params.domain_info[1],
                        np.array([0.,0.]),
                        PM.Dmat(sprd_factor*params_ary[9],
                                sprd_factor*params_ary[10],params_ary[11])))]
            '''
            sprd[int(sprd.shape[0]//2),int(sprd.shape[0]//2)] += max(0,1-sprd.sum())
            pmf_list = [sparse.coo_matrix(sprd)]
        else:
            pmf_list = []

        ###################### Get pmf_list from multiprocessing
        try:
            pmf_list.extend(pool.starmap(PM.prob_mass,pm_args))
        except PM.BndsError as e:
            print('BndsErr caught: at {}'.format(
                time.strftime("%H:%M:%S %d/%m/%Y")),end='\r')
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
            '''
            card_counts = []
            for nday,date in enumerate(locinfo.card_obs_datesPR):
                card_counts.append(np.zeros((4,locinfo.card_obs[nday].shape[1])))
            '''
            return (release_emerg,sentinel_emerg,grid_counts) #,card_counts)
        except Exception as e:
            print('Unrecognized exception in pool with PM.prob_mass!!')
            print(e)
            raise
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
            if params.dataset == 'kalbar':
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
        print('{:03.1f} sec./model at {}'.format(time.time() - modeltic,
            time.strftime("%H:%M:%S %d/%m/%Y")),end='\r')
        sys.stdout.flush()
        return (release_emerg,sentinel_emerg,grid_counts) #,card_counts)
        
    print('Parsing model output and connecting to Bayesian model...')    
    
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

    ### Collect model ###
    if params.dataset == 'kalbar':
        Bayes_model = pm.Model([lam,f_a1,f_a2,f_b1_p,f_b2_p,f_b1,f_b2,g_aw,g_bw,
                                sig_x,sig_y,corr_p,corr,sig_x_l,sig_y_l,
                                corr_l_p,corr_l,mu_r,
                                sprd_factor,grid_obs_prob_p,grid_obs_prob,
                                xi,em_obs_prob_p,em_obs_prob,A_collected,
                                sent_obs_probs_p,sent_obs_probs,params_ary,pop_model,
                                grid_poi_rates,rel_poi_rates,sent_poi_rates,
                                grid_obs,rel_collections,sent_collections])
    else:
        Bayes_model = pm.Model([lam,f_a1,f_a2,f_b1_p,f_b2_p,f_b1,f_b2,g_aw,g_bw,
                                sig_x,sig_y,corr_p,corr,sig_x_l,sig_y_l,
                                corr_l_p,corr_l,mu_r,grid_obs_prob_p,grid_obs_prob,
                                xi,em_obs_prob_p,em_obs_prob,A_collected,
                                sent_obs_prob_p,sent_obs_probs,params_ary,pop_model,
                                grid_poi_rates,rel_poi_rates,sent_poi_rates,
                                grid_obs,rel_collections,sent_collections])


    ######################################################################
    #####              Run Methods and Interactive Menu              #####
    ######################################################################
    
    def MAP_run():
        '''Find Maximum a posteriori distribution'''
        tic = time.time()
        M = pm.MAP(Bayes_model,prior_eps)
        print('Fitting....')
        M.fit()
        # Return statistics
        print('Estimate complete. Time elapsed: {}'.format(
              time.time() - tic))
        print('Free stochastic variables: {}'.format(M.len))
        print('Joint log-probability of model: {}'.format(M.logp))
        print('Max joint log-probability of model: {}'.format(
              M.logp_at_max))
        print('Maximum log-likelihood: {}'.format(M.lnL))
        print("Akaike's Information Criterion {}".format(M.AIC),
            flush=True)
        print('---------------Variable estimates---------------')
        for var in Bayes_model.stochastics:
            print('{} = {}'.format(var,var.value))
        # Save result to file
        with open('Max_aPosteriori_Estimate.txt','w') as fobj:
            fobj.write('Time elapsed: {}\n'.format(time.time() - tic))
            fobj.write('Free stochastic variables: {}\n'.format(M.len))
            fobj.write('Joint log-probability of model: {}\n'.format(M.logp))
            fobj.write('Max joint log-probability of model: {}\n'.format(
                  M.logp_at_max))
            fobj.write('Maximum log-likelihood: {}\n'.format(M.lnL))
            fobj.write("Akaike's Information Criterion {}\n".format(M.AIC))
            fobj.write('---------------Variable estimates---------------\n')
            for var in Bayes_model.stochastics:
                fobj.write('{} = {}\n'.format(var,var.value))
        print('Result saved to Max_aPosteriori_Estimate.txt.')
        
        
        
    def norm_run(fname):
        '''Find normal approximation'''
        try:
            tic = time.time()
            M = pm.NormApprox(Bayes_model,eps=prior_eps,db='hdf5',dbname=fname,
                              dbmode='a',dbcomplevel=0)
            print('Fitting....')
            M.fit()
            # Return statistics
            print('Estimate complete. Time elapsed: {}'.format(
                  time.time() - tic))
            print('Free stochastic variables: {}'.format(M.len))
            print('Joint log-probability of model: {}'.format(M.logp))
            print('Max joint log-probability of model: {}'.format(
                  M.logp_at_max))
            print("Akaike's Information Criterion {}".format(M.AIC),
                flush=True)
            print('---------------Variable estimates---------------')
            print('Estimated means: ')
            for var in bio_model.stochastics:
                print('{} = {}'.format(var,M.mu[var]))
            print('Estimated variances: ')
            for var in bio_model.stochastics:
                print('{} = {}'.format(var,M.C[var]))
            # Save result to file
            with open('Normal_approx.txt','w') as fobj:
                fobj.write('Time elapsed: {}\n'.format(time.time() - tic))
                fobj.write('Free stochastic variables: {}\n'.format(M.len))
                fobj.write('Joint log-probability of model: {}\n'.format(M.logp))
                fobj.write('Max joint log-probability of model: {}\n'.format(
                      M.logp_at_max))
                fobj.write("Akaike's Information Criterion {}\n".format(M.AIC))
                fobj.write('---------------Variable estimates---------------\n')
                fobj.write('Estimated means: \n')
                for var in bio_model.stochastics:
                    fobj.write('{} = {}\n'.format(var,M.mu[var]))
                fobj.write('Estimated variances: \n')
                for var in bio_model.stochastics:
                    fobj.write('{} = {}\n'.format(var,M.C[var]))
            print('These results have been saved to Normal_approx.txt.')
        except Exception as e:
            print(e)
            print('Exception: database closing...')
            mcmc.db.close()
            print('Database closed.')
            raise
        
    
    
    # Parse run type
    if RUNFLAG == 'MAP_RUN':
        MAP_run()
    elif RUNFLAG is not None:
        norm_run(RUNFLAG)
    else:
        print('----- Maximum a posteriori estimates & Normal approximations -----')
        while True:
            print(" 'map': Calculate maximum a posteriori estimate")
            print("'norm': Calculate normal approximation")
            print("'quit': Quit.")
            cmd = input('Enter: ')
            cmd = cmd.strip()
            cmd = cmd.lower()
            if cmd == 'map':
                MAP_run()
                # Option to enter IPython
                cmd_py = input('Enter IPython y/[n]:')
                cmd_py = cmd_py.strip()
                cmd_py = cmd_py.lower()
                if cmd_py == 'y' or cmd_py == 'yes':
                    import IPython
                    IPython.embed()
            elif cmd == 'norm':
                fname = input("Enter database name or 'back' to cancel:")
                fname = fname.strip()
                if fname == 'q' or fname == 'quit':
                    return
                elif fname == 'b' or fname == 'back':
                    continue
                else:
                    fname = fname+'.h5'
                norm_run(fname)
                try:
                    print('For covariances, enter IPython and request a covariance'+
                          ' matrix by passing variables in the following syntax:\n'+
                          'M.C[var1,var2,...,varn]\n'+
                          'Example: M.C[f_a1,f_a2] gives the covariance matrix of\n'+
                          ' f_a1 and f_a2.')
                    # Option to enter IPython
                    cmd_py = input('Enter IPython y/[n]:')
                    cmd_py = cmd_py.strip()
                    cmd_py = cmd_py.lower()
                    if cmd_py == 'y' or cmd_py == 'yes':
                        import IPython
                        IPython.embed()
                    M.db.close()
                    print('Database closed.')
                except Exception as e:
                    print(e)
                    print('Exception: database closing...')
                    mcmc.db.close()
                    print('Database closed.')
                    raise
            elif cmd == 'quit' or cmd == 'q':
                    return
            else:
                print('Command not recognized.')

    
if __name__ == "__main__":
    args = parser.parse_args()
    if args.MAP:
        RUNFLAG = 'MAP_RUN'
    elif args.norm is not None:
        RUNFLAG = args.norm
    else:
        RUNFLAG = None
    with Pool() as pool:
        main(RUNFLAG)
        #main(sys.argv[1:])