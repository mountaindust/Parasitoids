#! /usr/bin/env python3

'''Main file for running parasitoid model simulations.

Parameters are stored in an instance of the Params class, which also parses
command line options and settings in config.txt. Params also has methods for
returning wind data and a parameter list in the form expected by prob_mass in
ParasitoidModel.py. Calls to prob_mass can be done in parallel via the
multiprocessing python library. This module calls into CalcSol for the
convolution phase of the model simulation, and later possibly Plot_Result for
plotting the result. Saving simulations to file is handled internally.

Author: Christopher Strickland  
Email: cstrickland@samsi.info'''

__author__ = "Christopher Strickland"
__email__ = "cstrickland@samsi.info"
__status__ = "Development"
__version__ = "1.1"
__copyright__ = "Copyright 2015, Christopher Strickland"

import sys, os, time
import json
from multiprocessing import Pool
import numpy as np
from scipy import sparse
import globalvars
import ParasitoidModel as PM
from CalcSol import get_solutions, get_populations
import Plot_Result

### Parameters ###

class Params():
    '''Class definition to keep track with parameters for a model run'''
    ### Simulation flags ### (shared among all Params instances)
    OUTPUT = True
    PLOT = True
    CUDA = True
    
    def __init__(self):
        ### DEFAULT PARAMETERS ###
        # can be changed via command line
        # edit at will
        
        ### MODEL TYPE
        self.PROB_MODEL = True
        
        ### I/O
        # a couple of presets based on our data.
        # Current options: 'carnarvon' or 'kalbar' or None
        self.dataset = 'kalbar' 
        # get parameters based on this dataset
        self.my_datasets()
        
        # domain info, (dist (m), cells) from release point to side of domain 
        self.domain_info = (5000.0,500) # (float, int!) (this is 10 m res)
        # number of interpolation points per wind data point
        #   since wind is given every 30 min, 30 will give 1 min per point
        self.interp_num = 30
        # set this to a number >= 0 to only run the first n days
        self.ndays = 6

        ### function parameters
        # take-off scaling based on wind
        # aw,bw: first scalar centers the logistic, second one stretches it.
        self.g_params = (2.2, 5)
        # take-off probability mass function based on time of day
        # a1,b1,a2,b2: a# scalar centers logistic, b# stretches it.
        self.f_params = (6, 3, 18, 3)
        # In-flow diffusion coefficients, sig_x, sig_y, rho (units are meters)
        self.Dparams = (211,106,0)
        # Out-of-flow diffusion coefficients
        self.Dlparams = (21.1,10.6,0)

        ### general flight parameters
        # Probability of wind-based flight during the day under ideal conditions
        self.lam = 1.
        # scaling parameter for local drift
        self.mu_l_r = 0.2
        # scaling flight advection to wind advection
        self.mu_r = 1.0
        # number of time periods (based on interp_num) in one flight
        self.n_periods = 30 # if interp_num = 30, this is # of minutes per flight
        
        ### satellite imagry
        # Bing/Google maps key for satellite imagery
        self.maps_key = None
        self.maps_service = 'Google' #'Bing' or 'Google'
        
        # Parallel processing parameters
        self.min_ndays = 6 # min # of days necessary to use parallel processing
        
        ### check for config.txt and update these defaults accordingly
        self.default_chg()

    def my_datasets(self):
        if self.dataset is None:
            # defaults?
            self.site_name = 'data/carnarvonearl'
            self.start_time = '00:30'
            self.coord = None
            ### release information
            self.r_dur = None
            self.r_dist = None
            self.r_start = None
            self.r_number = None
            
        elif self.dataset == 'carnarvon':
            # site name and path
            self.site_name = 'data/carnarvonearl'
            # time of day at which the data starts ('00:00' or '00:30')
            self.start_time = '00:30'
            # coordinates (lat/long) of the release point. This is necessary for
            #   satellite imagery.
            self.coord = (-24.851614,113.731267)
            ### release information
            # release duration (days)
            self.r_dur = 5
            # release emergence distribution
            self.r_dist = 'uniform'
            # start time on first day (as a fraction of the day)
            self.r_start = 0.354 #8:30am (assumption. not specified in paper)
            # total number of wasps 
            self.r_number = 40000
            
        elif self.dataset == 'kalbar':
            self.site_name = 'data/kalbar'
            self.start_time = '00:00'
            self.coord = (-27.945752,152.58474)
            ### release information
            # release duration (days)
            self.r_dur = 1
            # release emergence distribution
            self.r_dist = 'uniform'
            # start time on first day (as a fraction of the day)
            self.r_start = None # wind didn't record until midnight post release
            # total number of wasps 
            self.r_number = 130000
            
        else:
            print('Unknown dataset in Params.dataset.')
        # name and path for output files
        if self.dataset is not None:
            if self.PROB_MODEL:
                self.outfile = 'output/'+self.dataset+time.strftime('%m%d-%H%M')
            else:
                self.outfile = 'output/'+self.dataset+'_pop'+time.strftime(
                            '%m%d-%H%M')
        else:
            if self.PROB_MODEL:
                self.outfile = 'output/'+time.strftime('%m%d-%H%M')
            else:
                self.outfile = 'output/poprun'+time.strftime('%m%d-%H%M')
  

  
    ########    Methods for multiple-day emergence    ########
        
    def uniform(self,day):
        '''Uniform distribution over emergence days. 1 <= day <= self.r_dur.'''
        
        return 1./self.r_dur
        
    def custom(self,day):
        '''Normal distribution over emergence days. 1 <= day <= self.r_dur.'''
        
        pass
        
    ####
    
    def r_mthd(self):
        '''Return function handle for the method to be used.
        We do this via this method (instead of directly) so that the parameter 
        r_dist can be saved in the json file with the other parameters.'''
        
        if self.r_dist == 'uniform':
            return self.uniform
        elif self.r_dist == 'custom':
            return self.custom

            
            
    ########    Methods for changing parameters    ########
    
    def default_chg(self):
        '''Look for a file called config.txt. If present, read it in and change
        the default parameters accordingly. If the file is not there, create it
        with some user friendly comments and examples.'''
        
        try:
            with open('config.txt', 'r') as f:
                for line in f:
                    c_ind = line.find('#')
                    if c_ind >= 0:
                        line = line[:c_ind] # chop off comments
                    words = line.split('=')
                    if len(words) > 1:
                        arg = words[0].strip()
                        val = words[1].strip()
                        self.chg_param(arg,val)
            self.my_datasets()
        except FileNotFoundError:
            with open('config.txt', 'w') as f:
                f.write('# local configuration file\n')
                f.write('\n')
                f.write('# Accepts keyword parameter assignments of the form '+
                    '<parameter> = <value>\n')
                f.write('# Any line starting with # will be ignored.\n')
                f.write('\n')
                f.write('# To include satellite imagery, please obtain a free '+
                    "Bing maps key at\n# https://www.bingmapsportal.com/ and "+
                    "assign it to the parameter 'maps_key'\n# in this file.\n")
        except ValueError:
            print(' in config.txt.')
            raise
                               
        
    def cmd_line_chg(self,args):
        '''Change parameters away from default based on command line args'''
        
        # Expect args to be a list of command line arguments in the form
        #   <param name>=<new value>
        
        for argstr in args:
            if argstr[0:2] == '--':
                # Flag set by option
                if argstr[2:].lower() == 'no_output':
                    self.OUTPUT = False
                elif argstr[2:].lower() == 'output':
                    self.OUTPUT = True
                elif argstr[2:].lower() == 'no_plot':
                    self.PLOT = False
                elif argstr[2:].lower() == 'plot':
                    self.PLOT = True
                elif argstr[2:].lower() == 'no_cuda':
                    self.CUDA = False
                elif argstr[2:].lower() == 'cuda':
                    self.CUDA = True
                elif argstr[2:].lower() == 'pop' or\
                    argstr[2:].lower() == 'popmodel' or\
                    argstr[2:].lower() == 'pop_model':
                    self.PROB_MODEL = False
                    self.my_datasets()
                elif argstr[2:].lower() == 'prob' or\
                    argstr[2:].lower() == 'probmodel' or\
                    argstr[2:].lower() == 'prob_model':
                    self.PROB_MODEL = True
                    self.my_datasets()
                ### known dataset locations ###
                elif argstr[2:].lower() == 'carnarvon':
                    self.dataset = 'carnarvon'
                    self.my_datasets()
                elif argstr[2:].lower() == 'kalbar':
                    self.dataset = 'kalbar'
                    self.my_datasets()
                else:
                    raise ValueError('Unrecognized option {0}.'.format(argstr))
            else:
                arg,eq,val = argstr.partition('=')
                self.chg_param(arg,val)               
                
                    
    def chg_param(self,arg,val):
        '''Change the parameter arg to val, where both are given as strings'''
        
        try:
            if arg.lower() == 'prob_model':
                self.prob_model = bool(val)
                self.my_datasets()
            elif arg == 'outfile':
                self.outfile = val
            elif arg == 'dataset':
                self.dataset = val
                self.my_datasets()
            elif arg == 'site_name':
                self.site_name = val
            elif arg == 'start_time':
                self.start_time = val
            elif arg == 'coord':
                val = val.strip(' ()')
                val = val.split(',')
                self.coord = (float(val[0]),float(val[1]))
            elif arg == 'domain_info':
                strinfo = val.strip('()').split(',')
                self.domain_info = (float(strinfo[0]),int(strinfo[1]))
            elif arg == 'interp_num':
                self.interp_num = int(val)
            elif arg == 'ndays':
                self.ndays = int(val)
            elif arg == 'r_dur':
                self.r_dur = int(val)
            elif arg == 'r_start':
                self.r_start == float(val)
            elif arg == 'r_number':
                self.r_number == int(val)
            elif arg == 'g_params':
                strinfo = val.strip('()').split(',')
                self.g_params = (float(strinfo[0]),float(strinfo[1]))
            elif arg == 'f_params':
                strinfo = val.strip('()').split(',')
                self.f_params = (float(strinfo[0]),float(strinfo[1]),
                    float(strinfo[2]),float(strinfo[3]))
            elif arg == 'Dparams':
                strinfo = val.strip('()').split(',')
                self.Dparams = (float(strinfo[0]),float(strinfo[1]),
                    float(strinfo[2]))
            elif arg == 'Dlparams':
                strinfo = val.strip('()').split(',')
                self.Dlparams = (float(strinfo[0]),float(strinfo[1]),
                    float(strinfo[2]))
            elif arg == 'lam':
                self.lam = float(val)
            elif arg == 'mu_l_r':
                self.mu_l_r = float(val)
            elif arg == 'mu_r':
                self.mu_r = float(val)
            elif arg == 'n_periods':
                self.n_periods = int(val)
            elif arg == 'min_ndays':
                self.min_ndays = int(val)
            elif arg == 'maps_key':
                self.maps_key = val
            elif arg == 'maps_service':
                self.maps_service = val
            elif arg == 'output':
                if val == 'True':
                    self.OUTPUT = True
                elif val == 'False':
                    self.OUTPUT = False
                else:
                    self.OUTPUT = bool(val)
            elif arg == 'plot':
                if val == 'True':
                    self.PLOT = True
                elif val == 'False':
                    self.PLOT = False
                else:
                    self.PLOT = bool(val)
            elif arg == 'cuda':
                if val == 'True':
                    self.CUDA = True
                elif val == 'False':
                    self.CUDA = False
                else:
                    self.CUDA = bool(val)
            else:
                raise LookupError('Unrecognized parameter {0}.'.format(arg))
        except LookupError:
            print('Could not parse {0}.'.format(arg)+'\n ')
            raise
        except ValueError:
            print('Could not parse {0}.'.format(arg)+
                ' Try enclosing this argument in quotations.\n ')
            raise
        
                
    def file_read_chg(self,filename):
        '''Read in parameters from a file'''
        if filename.rstrip()[-5:] != '.json':
            filename = filename.rstrip()+'.json'

        try:
            with open(filename) as fobj:
                param_dict = json.load(fobj)
        except FileNotFoundError as e:
            print('Could not open file {0}.'.format(filename))
            raise
        
        for key in param_dict:
            setattr(self,key,param_dict[key])

    

    ########    Methods for getting function parameters    ########
        
    def get_model_params(self):
        '''Return params in order of ParasitoidModel.prob_mass signature, 
        minus day & wind_data'''
        hparams = (self.lam,*self.g_params,*self.f_params)
        return (hparams,self.Dparams,self.Dlparams,self.mu_r,self.mu_l_r,
            self.n_periods,*self.domain_info)  
        
        
    def get_wind_params(self):
        '''Return wind params to pass to PM.get_wind_data'''
        return (self.site_name,self.interp_num,self.start_time)
        
        
        
def main(params):
    ''' This is the main routine for running model simulations.
    A Params object is required, which sets up all parameters for the simulation
    '''
        
    # This sends a message to CalcSol on whether or not to use CUDA
    if params.CUDA:
        globalvars.cuda = True
    else:
        globalvars.cuda = False
    
    wind_data,days = PM.get_wind_data(*params.get_wind_params())
    
    if params.ndays >= 0:
        ndays = min(params.ndays,len(days))
    else:
        ndays = len(days)
    
    # First, get spread probability for each day as a coo sparse matrix
    tic_total = time.time()
    tic = time.time()
    pmf_list = []
    max_shape = np.array([0,0])
    
    if ndays >= params.min_ndays:
        print("Calculating each day's spread in parallel...")
        if params.PROB_MODEL:
            pm_args = [(day,wind_data,*params.get_model_params()) 
                    for day in days[:ndays]]
        else:
            pm_args = [(days[0],wind_data,*params.get_model_params(),
                    params.r_start)]
            pm_args.extend([(day,wind_data,*params.get_model_params()) 
                    for day in days[1:ndays]])
        pool = Pool()
        try:
            pmf_list = pool.starmap(PM.prob_mass,pm_args)
        except PM.BndsError as e:
            print('BndsError caught in ParasitoidModel.py.')
            print(e)
            sys.exit()
        finally:
            pool.close()
            pool.join()
        for pmf in pmf_list:
            for dim in range(2):
                if pmf.shape[dim] > max_shape[dim]:
                    max_shape[dim] = pmf.shape[dim]
    else:
        for n,day in enumerate(days[:ndays]):
            print('Calculating spread for day {0} PR'.format(n+1))
            if params.PROB_MODEL:
                pmf_list.append(PM.prob_mass(
                               day,wind_data,*params.get_model_params()))
            else:
                if n == 0:
                    pmf_list.append(PM.prob_mass(
                                day,wind_data,*params.get_model_params(),
                                start_time=params.r_start))
                else:
                    pmf_list.append(PM.prob_mass(
                                day,wind_data,*params.get_model_params()))
            # record the largest shape of these
            for dim in range(2):
                if pmf_list[-1].shape[dim] > max_shape[dim]:
                    max_shape[dim] = pmf_list[-1].shape[dim]
                
    print('Time elapsed: {0}'.format(time.time()-tic))
    if params.PROB_MODEL:
        modelsol = [] # holds actual model solutions
        
        # Reshape the first probability mass function into a solution
        offset = params.domain_info[1] - pmf_list[0].shape[0]//2
        dom_len = params.domain_info[1]*2 + 1
        modelsol.append(sparse.coo_matrix((pmf_list[0].data, 
            (pmf_list[0].row+offset,pmf_list[0].col+offset)),
            shape=(dom_len,dom_len)))


        # Pass the first solution, pmf_list, and other info to convolution solver
        #   This updates modelsol with the rest of the solutions.
        tic = time.time()
        get_solutions(modelsol,pmf_list,days,ndays,dom_len,max_shape)
    else:
        r_spread = [] # holds the one-day spread for each release day.
    
        # Reshape the prob. mass function of each release day into solution form
        for ii in range(params.r_dur):
            offset = params.domain_info[1] - pmf_list[ii].shape[0]//2
            dom_len = params.domain_info[1]*2 + 1
            r_spread.append(sparse.coo_matrix((pmf_list[ii].data, 
                (pmf_list[ii].row+offset,pmf_list[ii].col+offset)),
                shape=(dom_len,dom_len)).tocsr())

        
        # Pass the probability list, pmf_list, and other info to convolution solver.
        #   This will return the finished population model.
        tic = time.time()
        modelsol = get_populations(r_spread,pmf_list,days,ndays,dom_len,max_shape,
                                   params.r_dur,params.r_number,params.r_mthd())
    
    # done.
    print('Done.')

    print('Time elapsed: {0}'.format(time.time()-tic))
    print('Total time elapsed: {0}'.format(time.time()-tic_total))

    ### save result ###
    if params.OUTPUT:
        print('Saving...')
        # for consistency, let's save both types of solutions as CSR sparse
        if params.PROB_MODEL:
            for n,sol in enumerate(modelsol):
                modelsol[n] = sol.tocsr()
        def outputGenerator():
            # Creates generator for output formatting
            for n,day in enumerate(days[:ndays]):
                yield (str(day)+'_data', modelsol[n].data)
                yield (str(day)+'_ind', modelsol[n].indices)
                yield (str(day)+'_indptr', modelsol[n].indptr)
            yield ('days',days[:ndays])
            
        outgen = outputGenerator()
        # check for directory path
        dir_file = params.outfile.rsplit('/',1)
        if len(dir_file) > 1:
            if not os.path.exists(dir_file[0]):
                os.makedirs(dir_file[0])
        np.savez(params.outfile,**{x: y for (x,y) in outgen})
        
        ### save parameters ###
        with open(params.outfile+'.json','w') as fobj:
            param_dict = dict(params.__dict__)
            param_dict.pop('maps_key') # don't save key
            json.dump(param_dict,fobj)
    
    ### plot result ###
    if params.PLOT:
        Plot_Result.plot_all(modelsol,params)
    

if __name__ == "__main__":
    ### Get and set parameters ###
    params = Params()
    
    if len(sys.argv[1:]) > 0:
        params.cmd_line_chg(sys.argv[1:])
    
    ### run model ###
    main(params)