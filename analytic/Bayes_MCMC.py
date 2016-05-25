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
import os.path
from io import StringIO
from multiprocessing import Pool
import numpy as np
#import pandas as pd
from scipy import sparse
import pymc as pm
import globalvars
from Run import Params
from Data_Import import LocInfo
import ParasitoidModel as PM
from CalcSol import get_populations

    
    
###############################################################################
#                                                                             #
#                         Supporting functions                                #
#                                                                             #
###############################################################################
    
def popdensity_to_emergence(modelsol,locinfo):
    '''Translate population model to corresponding expected number of wasps in
    a given location whose oviposition would result in a given emergence date. 
    Only use the locations in which data was actually collected.
    '''
    
    # Assume collections are done at the beginning of the day, observations
    #   of collection data at the end of the day. So, oviposition is not possible
    #   on the day of collection, but emergence is.
    
    ### Oviposition to emergence time ###
    # For now, assume this is a constant
    incubation_time = 15 # days (Average of 16 & 14)
    max_incubation_time = 15
    
    ### Project release field grid ###
    release_emerg = []
    for nframe,dframe in enumerate(locinfo.release_DataFrames):
        # Each dataframe should be sorted already, 'datePR','row','column'.
        # Also, the grid for each collection is stored in the list
        #   locinfo.emerg_grids.
        
        collection_day = (locinfo.collection_datesPR[nframe]).days

        ### Find the earliest and latest oviposition date PR that we need to ###
        ### simulate for this collection. 0 = release day.                   ###
        # The last day oviposition is possible is the day before collection
        # The earliest day oviposition is possible is the max incubation time
        #   before the first possible emergence
        start_day = max(collection_day - max_incubation_time,0) # days post release!
        ########################################################################
        
        #
        # Go through each feasible oviposition day of the model, projecting emergence
        #
        
        # emerg_proj holds each grid point in its rows and a different emergence
        #   day in its columns.
        # Feasible emergence days span the maximum incubation time.
        emerg_proj = np.zeros((len(locinfo.emerg_grids[nframe]), 
            max_incubation_time))
        
        # go through feasible oviposition days
        for nday,day in enumerate(range(start_day,collection_day)):
            n = 0 # row/col count
            # in each one, go through grid points projecting emergence date
            #   potentials per adult wasp per cell.
            for r,c in locinfo.emerg_grids[nframe]:
                ###                Project forward and store                 ###
                ### This function can be more complicated if we want to try  ###
                ###   and be more precise. It's a mapping from feasible      ###
                ###   oviposition dates to array of feasible emergence dates ###
                # nday represents index of feasible emergence days, [collect,+max)
                # day represents feasible oviposition days, [start,collect)
                emerg_proj[n,nday] = modelsol[day][r,c]
                ################################################################
                n += 1
                
        # now consolidate these days into just the days data was collected.
        # first, get unique dates
        obs_datesPR = dframe['datePR'].map(lambda t: t.days).unique()
        modelsol_grid_emerg = np.zeros((len(locinfo.emerg_grids[nframe]),
                                        len(obs_datesPR)))
        col_indices = obs_datesPR - collection_day
        modelsol_grid_emerg[:,0] = emerg_proj[:,0:col_indices[0]+1].sum(axis=1)
        for n,col in enumerate(col_indices[1:]):
            col_last = col_indices[n]
            modelsol_grid_emerg[:,n+1] = emerg_proj[:,col_last+1:col+1].sum(axis=1)
        release_emerg.append(modelsol_grid_emerg)
        
    ### Project sentinel field emergence ###
    sentinel_emerg = []
    for nframe,dframe in enumerate(locinfo.sent_DataFrames):
        # Each dataframe should be sorted already, 'datePR','id' 
        
        collection_day = (locinfo.collection_datesPR[nframe]).days

        ### Find the earliest and latest oviposition date PR that we need to ###
        ### simulate for this collection. 0 = release day.                   ###
        # The last day oviposition is possible is the day before collection
        # The earliest day oviposition is possible is the max incubation time
        #   before the first possible emergence
        start_day = max(collection_day - max_incubation_time,0) # days post release!
        ########################################################################
        
        #
        # Go through each feasible oviposition day of the model, projecting emergence
        #
        
        # emerg_proj holds each sentinel field in its rows and a different 
        #   emergence day in its columns.
        # Feasible emergence days start at collection and go until observation stopped
        emerg_proj = np.zeros((len(locinfo.sent_ids[nframe]), 
            max_incubation_time))
            
        # go through feasible oviposition days
        for nday,day in enumerate(range(start_day,collection_day)):
            # for each day, aggregate the population in each sentinel field
            for n,field_id in enumerate(locinfo.sent_ids[nframe]):
                ###     Sum the field cells, project forward and store       ###
                ### This function can be more complicated if we want to try  ###
                ###   and be more precise. It's a mapping from feasible      ###
                ###   oviposition dates to array of feasible emergence dates ###
                field_total = modelsol[day][locinfo.field_cells[field_id][:,0],
                                    locinfo.field_cells[field_id][:,1]].sum()
                emerg_proj[n,nday] = field_total
                ################################################################
        
        # now consolidate these days into just the days data was collected.
        # first, get unique dates
        obs_datesPR = dframe['datePR'].map(lambda t: t.days).unique()
        modelsol_field_emerg = np.zeros((len(locinfo.sent_ids[nframe]),
                                        len(obs_datesPR)))
        col_indices = obs_datesPR - collection_day
        modelsol_field_emerg[:,0] = emerg_proj[:,0:col_indices[0]+1].sum(axis=1)
        for n,col in enumerate(col_indices[1:]):
            col_last = col_indices[n]
            modelsol_field_emerg[:,n+1] = emerg_proj[:,col_last+1:col+1].sum(axis=1)
        sentinel_emerg.append(modelsol_field_emerg)
        
    ### This process results in two lists, release_emerg and sentinel_emerg.
    ###     Each list entry corresponds to a data collection day (one array)
    ##      In each array:
    ###     Each column corresponds to an emergence observation day (as in data)
    ###     Each row corresponds to a grid point or sentinel field, respectively
    ### This format will need to match a structured data arrays for comparison
    
    return (release_emerg,sentinel_emerg)
        
        
    
def popdensity_grid(modelsol,locinfo):
    '''Translate population model to corresponding expected number of wasps in
    each grid point
    '''

    # Assume observations are done at the beginning of the day.
    grid_counts = np.zeros((locinfo.grid_cells.shape[0],
                            len(locinfo.grid_obs_datesPR)))

    for nday,date in enumerate(locinfo.grid_obs_datesPR):
        n = 0 # row/col count
        # for each day, get expected population at each grid point
        for r,c in locinfo.grid_cells:
            # model holds end-of-day PR results
            grid_counts[n,nday] = modelsol[date.days-1][r,c]
            n += 1

    ### Return ndarray where:
    ###     Each column corresponds to an observation day
    ###     Each row corresponds to a grid point

    return grid_counts



def popdensity_card(modelsol,locinfo,domain_info):
    '''Translate population model to expected number of wasps in cardinal
    directions.
    '''

    # Assume observations are done at the beginning of the day.
    # Form a list, one entry per sample day
    card_counts = []
    res = domain_info[0]/domain_info[1] # cell length in meters

    for nday,date in enumerate(locinfo.card_obs_datesPR):
        # get array shape
        obslen = locinfo.card_obs[nday].shape[1]
        day_count = np.zeros((4,obslen))
        # the release point is an undisturbed 5x5m area from which all
        #   sampling distances are calculated.
        dist = 5
        for step in range(obslen):
            dist += locinfo.step_size[nday]
            cell_delta = dist//res
            # north
            day_count[0,step] = modelsol[date.days-1][domain_info[1]-cell_delta,
                                                      domain_info[1]]
            # south
            day_count[1,step] = modelsol[date.days-1][domain_info[1]+cell_delta,
                                                      domain_info[1]]
            # east
            day_count[2,step] = modelsol[date.days-1][domain_info[1],
                                                      domain_info[1]+cell_delta]
            # west
            day_count[3,step] = modelsol[date.days-1][domain_info[1],
                                                      domain_info[1]-cell_delta]
        card_counts.append(day_count)

    ### Return list of ndarrays where:
    ###     Each column corresponds to step in the cardinal direction
    ###     Each row corresponds to a cardinal direction: north,south,east,west

    return card_counts



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
    '''Need to catch any non-PM.BndsError and dump state'''

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
    A_collected = pm.TruncatedNormal("A_collected",25,50,0,
                                    min(locinfo.field_sizes.values()),value=16)  
    # Each field has its own binomial probability.
    N = len(locinfo.sent_ids)
    sent_obs_probs = np.empty(N, dtype=object)
    field_obs_vars = np.empty(N, dtype=object)
    field_obs_means = np.empty(N, dtype=object)
    for n,key in enumerate(locinfo.sent_ids):
        #auto-create deterministic variables
        field_obs_means[n] = A_collected/locinfo.field_sizes[key]
        #Lambda to get deterministic variables
        field_obs_vars[n] = pm.Lambda("var_{}".format(key),lambda m=mean: min(m,0.1))
        sent_obs_probs[n] = pm.Beta("sent_obs_prob_{}".format(key),
            field_obs_means[n]*(1-field_obs_means[n]-field_obs_vars[n])/field_obs_vars[n],
            (field_obs_means[n]-field_obs_vars[n])*(1-field_obs_means[n])/field_obs_vars[n])
    
    
    #### Collect variables ####
    params_ary = np.array([g_aw,g_bw,f_a1,f_b1,f_a2,f_b2,sig_x,sig_y,corr,
                           sig_x_l,sig_y_l,corr_l,lam,mu_r],dtype=object)
         
    
    #### Run model ####
    @pm.deterministic
    def pop_model(params=params,params_ary=params_ary,locinfo=locinfo,wind_data=wind_data,days=days):
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
        print("Calculating each day's spread in parallel...")
        pm_args = [(days[0],wind_data,*params.get_model_params(),
                params.r_start)]
        pm_args.extend([(day,wind_data,*params.get_model_params()) 
                for day in days[1:params.ndays]])
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
                sentinel_emerg.append(np.zeros((len(locinfo.sent_ids[nframe]),
                                      len(obs_datesPR))))
            return (release_emerg,sentinel_emerg)
        finally:
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
    @pm.deterministic
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
        
    @pm.deterministic
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
            
    @pm.deterministic
    def grid_poi_rates(locinfo=locinfo,beta=grid_obs_prob,
                       obs_model=pop_model[2]):
        '''Return Poisson probabilities for grid sampling
        obs_model is an ndarray, sampling effort is specified in locinfo.'''
        return beta*locinfo.grid_samples*obs_model

    @pm.deterministic
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

    #####       Setup model object      #####

    modelobj = pm.Model([lam,f_a1,f_a2,f_b1,f_b2,g_aw,g_bw,sig_x,sig_y,corr,
                         sig_x_l,sig_y_l,corr_l,mu_r,
        card_obs_prob,grid_obs_prob,xi,em_obs_prob,
        A_collected,field_obs_means,field_obs_vars,sent_obs_probs,
        params_ary,pop_model,
        card_poi_rates,grid_poi_rates,rel_poi_rates,sent_poi_rates,
        card_collections,grid_obs,rel_collections,sent_collections])

    #####       Start Interactive MCMC      #####
    print('--------------- MCMC MAIN MENU ---------------')
    print(" 'new': Start a new MCMC chain from the beginning.")
    print("'cont': Continue a previous MCMC chain from an hdf5 file.")
    print("'plot': Plot traces from an hdf5 file.")
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
                    iter = int(val)
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
        mcmc = pm.MCMC(modelobj,db='hdf5',dbname=fname,
                        dbmode='a',dbcomplevel=0)
        try:
            mcmc.sample(iter,burn)
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
                    mcmc = pm.MCMC(modelobj,db=db)
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
    # This should be reached only by cmd == 'new' or 'cont' with a database


    
if __name__ == "__main__":
    main(sys.argv[1:])