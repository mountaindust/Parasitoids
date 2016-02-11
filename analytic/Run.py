#! /usr/bin/env python3

import sys
import os
import json
import numpy as np
from scipy import sparse
import globalvars
import ParasitoidModel as PM
from CalcSol import get_solutions
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
        self.outfile = 'output/carnarvonearl'
        # site name and path
        self.site_name = 'data/carnarvonearl'
        # time of day at which the data starts ('00:00' or '00:30')
        self.start_time = '00:30'
        # domain info, (dist (m), cells) from release point to side of domain 
        self.domain_info = (8000.0,1600) # (float, int!) (this is 5 m**2)
        # number of interpolation points per wind data point
        #   since wind is given every 30 min, 30 will give 1 min per point
        self.interp_num = 30
        # set this to a number >= 0 to only run the first n days
        self.ndays = 2

        ### function parameters
        # take-off scaling based on wind
        # aw,bw: first scalar centers the logistic, second one stretches it.
        self.g_params = (2.2, 5)
        # take-off probability mass function based on time of day
        # a1,b1,a2,b2: a# scalar centers logistic, b# stretches it.
        self.f_params = (6, 3, 18, 3)
        # Diffusion coefficients, sig_x, sig_y, rho (units are meters)
        self.Dparams = (4,4,0)

        ### general flight parameters
        # Probability of any flight during the day under ideal circumstances
        self.lam = 1.
        # scaling flight advection to wind advection
        self.mu_r = 1.
        # number of time periods (based on interp_num) in one flight
        self.n_periods = 10 # if interp_num = 30, this is # of minutes
        
        ### check for config.txt and update these defaults accordingly
        self.default_chg()
        
    def default_chg(self):
        '''Look for a file called config.txt. If present, read it in and change
        the default parameters accordingly.'''
        
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
            pass # do nothing - the file is not necessary to run
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
            elif arg == 'domain_info':
                strinfo = val.strip('()').split(',')
                self.domain_info = (float(strinfo[0]),int(strinfo[1]))
            elif arg == 'interp_num':
                self.interp_num = int(val)
            elif arg == 'ndays':
                self.ndays = int(val)
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
                raise ValueError('Unrecognized parameter.')
        except:
            print('Could not parse {0}.'.format(arg),end='')
            raise
        
                
    def file_read_chg(self,filename):
        '''Read in parameters from a file'''
        if filename.rstrip()[-5:] != '.json':
            filename = filename.rstrip()+'.json'

        def get_param(pdict,pname):
            try:
                return pdict[pname]
            except KeyError as e:
                print('Could not load parameter value {0}'.format(e.args[0]))
                print('Using default value...')

        try:
            with open(filename) as fobj:
                param_dict = json.load(fobj)
        except FileNotFoundError as e:
            print('Could not open file {0}.'.format(filename))
            raise
        self.outfile = get_param(param_dict,'outfile')
        self.site_name = get_param(param_dict,'site_name')
        self.start_time = get_param(param_dict,'start_time')
        self.domain_info = get_param(param_dict,'domain_info')
        self.interp_num = get_param(param_dict,'interp_num')
        self.ndays = get_param(param_dict,'ndays')
        self.g_params = get_param(param_dict,'g_params')
        self.f_params = get_param(param_dict,'f_params')
        self.Dparams = get_param(param_dict,'Dparams')
        self.lam = get_param(param_dict,'lam')
        self.mu_r = get_param(param_dict,'mu_r')
        self.n_periods = get_param(param_dict,'n_periods')

        
        
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
    pmf_list = []
    max_shape = np.array([0,0])
    for day in days[:ndays]:
        print('Calculating spread for day {0}'.format(day))
        pmf_list.append(PM.prob_mass(day,wind_data,*params.get_model_params()))
        # record the largest shape of these
        for dim in range(2):
            if pmf_list[-1].shape[dim] > max_shape[dim]:
                max_shape[dim] = pmf_list[-1].shape[dim]
                
    modelsol = [] # holds actual model solutions
    
    # Reshape the first probability mass function into a solution
    print('Reshaping day 1 solution')
    offset = params.domain_info[1] - pmf_list[0].shape[0]//2
    dom_len = params.domain_info[1]*2 + 1
    modelsol.append(sparse.coo_matrix((pmf_list[0].data, 
        (pmf_list[0].row+offset,pmf_list[0].col+offset)),
        shape=(dom_len,dom_len)))

    # Pass the first solution, pmf_list, and other info to convolution solver
    #   This updates modelsol with the rest of the solutions.
    get_solutions(modelsol,pmf_list,days,ndays,dom_len,max_shape)
    
    # done.
    print('Done.')
    
    ### save result ###
    if params.OUTPUT:
        # print('Removing small values from solutions...')
        # for n,sol in enumerate(modelsol):
            # modelsol[n] = PM.r_small_vals(sol)
        print('Saving...')
        def outputGenerator():
            # Creates generator for output formatting
            for n,day in enumerate(days[:ndays]):
                yield (str(day)+'_data', modelsol[n].data)
                yield (str(day)+'_row', modelsol[n].row)
                yield (str(day)+'_col', modelsol[n].col)
            yield ('days',days[:ndays])
            
        outgen = outputGenerator()
        # check for directory path
        dir_file = params.outfile.rsplit('/',1)
        if len(dir_file) > 1:
            if not os.path.exists(dir_file[0])
                os.makedirs(dir_file[0])
        np.savez(params.outfile,**{x: y for (x,y) in outgen})
        
        ### save parameters ###
        with open(params.outfile+'.json','w') as fobj:
            json.dump(params.__dict__,fobj)
    
    ### plot result ###
    if params.PLOT:
        Plot_Result.plot_all(modelsol,days,params.domain_info)
    

if __name__ == "__main__":
    main(sys.argv[1:])
