#! /usr/bin/env python3

'''Main file for running parasitoid population model simulations.
Unlike in the probability model, release information must be specified.

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
__version__ = "0.2"
__copyright__ = "Copyright 2015, Christopher Strickland"

import sys, os, time
import json
from multiprocessing import Pool
import numpy as np
from scipy import sparse
import globalvars
import ParasitoidModel as PM
from CalcSol import get_populations
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

        ### I/O
        # name and path for output files
        self.outfile = 'output/kalbar_pop'
        # site name and path
        self.site_name = 'data/kalbar'
        # time of day at which the wind data starts ('00:00' or '00:30')
        # kalbar: '00:00', carvarvonearl: '00:30'
        self.start_time = '00:00'
        # coordinates (lat/long) of the release point. This is necessary for
        #     satellite imagery.
        # self.coord = (-24.851614,113.731267) #carnarvon
        self.coord = (-27.945752,152.85474) #kalbar
        # domain info, (dist (m), cells) from release point to side of domain 
        self.domain_info = (8000.0,1600) # (float, int!) (this is 5 m**2)
        # number of interpolation points per wind data point
        #   since wind is given every 30 min, 30 will give 1 min per point
        self.interp_num = 30
        # set this to a number >= 0 to only run the first n days
        self.ndays = 5

        ### release information
        # release duration (days)
        self.r_dur = 3
        # release emergence distribution
        self.r_dist = self.uniform
        # start time on first day (as a fraction of the day)
        self.r_start = 0.354 #8:30am
        # total number of wasps 
        # (for now, assume they are divided equally between release days)
        self.r_number = 130000
        
        ### function parameters
        # take-off scaling based on wind
        # aw,bw: first scalar centers the logistic, second one stretches it.
        self.g_params = (2.2, 5)
        # take-off probability mass function based on time of day
        # a1,b1,a2,b2: a# scalar centers logistic, b# stretches it.
        self.f_params = (6, 3, 18, 3)
        # Diffusion coefficients, sig_x, sig_y, rho (units are meters)
        self.Dparams = (21.1,10.6,0)

        ### general flight parameters
        # Probability of any flight during the day under ideal circumstances
        self.lam = 1.
        # scaling flight advection to wind advection
        self.mu_r = 1.
        # number of time periods (based on interp_num) in one flight
        self.n_periods = 10 # if interp_num = 30, this is # of minutes
        
        # Bing maps key for satellite imagery
        self.maps_key = None
        
        # Parallel processing parameters
        self.min_ndays = 6 # min # of days necessary to use parallel processing
        
        ### check for config.txt and update these defaults accordingly
        self.default_chg()

        
        
    ########    Methods for multiple-day emergence    ########
        
    def uniform(self,day):
        '''Uniform distribution over emergence days. 1 <= day <= self.r_dur.'''
        
        return 1./self.r_dur
        
    def trunc_norm(self,day):
        '''Normal distribution over emergence days. 1 <= day <= self.r_dur.'''
        
        pass
        

    ########    Methods for changing parameters    ########
    
    def default_chg(self):
        '''Look for a file called config.txt. If present, read it in and change
        the default parameters accordingly. If the file is not there, create it
        with some user friendly comments and examples.'''
        
        try:
            with open('config.txt', 'r') as f:
                for line in f:
                    words = line.split()
                    for n,word in enumerate(words):
                        if word == '#': #comment
                            break
                        elif word == '=':
                            arg = words[n-1]
                            val = words[n+1]
                            self.chg_param(arg,val)
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
                else:
                    raise ValueError('Unrecognized option {0}.'.format(argstr))
            else:
                arg,eq,val = argstr.partition('=')
                self.chg_param(arg,val)
                
                
                    
    def chg_param(self,arg,val):
        '''Change the parameter arg to val, where both are given as strings'''
        
        try:
            if arg == 'outfile':
                self.outfile = val
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
            elif arg == 'lam':
                self.lam = float(val)
            elif arg == 'mu_r':
                self.mu_r = float(val)
            elif arg == 'n_periods':
                self.n_periods = int(val)
            elif arg == 'min_ndays':
                self.min_ndays = int(val)
            elif arg == 'maps_key':
                self.maps_key = val
                
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
                raise ValueError('Unrecognized parameter {0}.'.format(arg))
        except:
            print('Could not parse {0}.'.format(arg),end='')
            raise
        
        
        
    def file_read_chg(self,filename):
        '''Read in parameters from a file'''
        if filename.rstrip()[-5:] != '.json':
            filename = filename.rstrip()+'.json'

        def get_param(pdict,pname,param):
            '''Modifies param with pname, or leaves it alone if pname not found'''
            try:
                param = pdict[pname]
            except KeyError as e:
                print('Could not load parameter value {0}'.format(e.args[0]))
                print('Using default value...')

        try:
            with open(filename) as fobj:
                param_dict = json.load(fobj)
        except FileNotFoundError as e:
            print('Could not open file {0}.'.format(filename))
            raise
            
        get_param(param_dict,'outfile',self.outfile)
        get_param(param_dict,'site_name',self.site_name)
        get_param(param_dict,'start_time',self.start_time)
        get_param(param_dict,'coord',self.coord)
        get_param(param_dict,'domain_info',self.domain_info)
        get_param(param_dict,'interp_num',self.interp_num)
        get_param(param_dict,'ndays',self.ndays)
        get_param(param_dict,'r_dur',self.r_dur)
        get_param(param_dict,'r_start',self.r_start)
        get_param(param_dict,'r_number',self.r_number)
        get_param(param_dict,'g_params',self.g_params)
        get_param(param_dict,'f_params',self.f_params)
        get_param(param_dict,'Dparams',self.Dparams)
        get_param(param_dict,'lam',self.lam)
        get_param(param_dict,'mu_r',self.mu_r)
        get_param(param_dict,'n_periods',self.n_periods)
        get_param(param_dict,'min_ndays',self.min_ndays)

    

    ########    Methods for getting function parameters    ########
        
    def get_model_params(self):
        '''Return params in order necessary to run model, 
        minus day & wind_data'''
        hparams = (self.lam,*self.g_params,*self.f_params)
        return (hparams,self.Dparams,self.mu_r,self.n_periods,*self.domain_info)
        
        
        
    def get_wind_params(self):
        '''Return wind params to pass to PM.get_wind_data'''
        return (self.site_name,self.interp_num,self.start_time)
        
        
        
def main(argv):
    ### Get and set parameters ###
    params = Params()
    
    if len(argv) > 0:
        params.cmd_line_chg(argv)
        
    # This sends a message to CalcSol to not use CUDA
    if params.CUDA:
        globalvars.cuda = True
    else:
        globalvars.cuda = False
        
    wind_data,days = PM.get_wind_data(*params.get_wind_params())
    
    ### run model ###
    if params.ndays >= 0:
        ndays = params.ndays
    else:
        ndays = len(days)
    
    # First, get spread probability for each day as a coo sparse matrix
    tic = time.time()
    pmf_list = []
    max_shape = np.array([0,0])
    
    if ndays >= params.min_ndays:
        print("Calculating each day's spread in parallel...")
        pm_args = [(days[0],wind_data,*params.get_model_params(),params.r_start)]
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
    else:
        for n,day in enumerate(days[:ndays]):
            print('Calculating spread for day {0} PR'.format(n+1))
            if n == 0:
                pmf_list.append(PM.prob_mass(
                                day,wind_data,*params.get_model_params(),
                                params.r_start))
            else:
                pmf_list.append(PM.prob_mass(
                                day,wind_data,*params.get_model_params()))
            # record the largest shape of these
            for dim in range(2):
                if pmf_list[-1].shape[dim] > max_shape[dim]:
                    max_shape[dim] = pmf_list[-1].shape[dim]
                
    print('Time elapsed: {0}'.format(time.time()-tic))
    r_spread = [] # holds the one-day spread for each release day.
    
    # Reshape the prob. mass function of each release day into solution form
    for ii in range(params.r_dur):
        offset = params.domain_info[1] - pmf_list[ii].shape[0]//2
        dom_len = params.domain_info[1]*2 + 1
        r_spread.append(sparse.coo_matrix((pmf_list[ii].data, 
            (pmf_list[ii].row+offset,pmf_list[ii].col+offset)),
            shape=(dom_len,dom_len)))


    # Pass the probability list, pmf_list, and other info to convolution solver.
    #   This will return the finished population model.
    tic = time.time()
    popmodel = get_populations(r_spread,pmf_list,days,ndays,dom_len,max_shape,
                               params.r_dur,params.r_number,params.r_dist)
    
    # done.
    print('Done.')

    print('Time elapsed: {0}'.format(time.time()-tic))

    ### save result ###
    if params.OUTPUT:
        print('Saving...')
        def outputGenerator():
            # Creates generator for output formatting
            for n,day in enumerate(days[:ndays]):
                yield (str(day)+'_data', popmodel[n].data)
                yield (str(day)+'_ind', popmodel[n].indices)
                yield (str(day)+'_indptr', popmodel[n].indptr)
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
        Plot_Result.plot_all(popmodel,params)
    

if __name__ == "__main__":
    main(sys.argv[1:])