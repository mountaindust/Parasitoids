#! /usr/bin/env python3

import sys
import json
import numpy as np
from scipy import sparse
import ParasitoidModel as PM

### Parameters ###

class Params():
    '''Class definition to keep track with parameters for a model run'''
    
    def __init__(self):
        ### DEFAULT PARAMETERS ###
        # can be changed via command line
        # edit at will

        ### I/O
        # name and path for output files
        self.outfile = 'output\carnarvonearl'
        # site name and path
        self.site_name = 'data\carnarvonearl'
        # time of day at which the data starts ('00:00' or '00:30')
        self.start_time = '00:30'
        # domain info, (dist (m), cells) from release point to side of domain 
        self.domain_info = (8000.0,2000) # (float, int!)
        # number of interpolation points per wind data point
        #   since wind is given every 30 min, 30 will give 1 min per point
        self.interp_num = 30
        # set this to a number >= 0 to only run the first n days
        self.ndays = 2

        ### function parameters
        # take-off scaling based on wind
        # aw,bw: first scalar centers the logistic, second one stretches it.
        self.g_params = (1.8, 6)
        # take-off probability mass function based on time of day
        # a1,b1,a2,b2: a# scalar centers logistic, b# stretches it.
        self.f_params = (7, 1.5, 17, 1.5)
        # Diffusion coefficients, sig_x, sig_y, rho
        self.Dparams = (4, 4, 0)

        ### general flight parameters
        # Probability of any flight during the day under ideal circumstances
        self.lam = 1.
        # scaling flight advection to wind advection
        self.mu_r = 1.
        # number of time periods (based on interp_num) in one flight
        self.n_periods = 6 # if interp_num = 30, this is # of minutes
        
    def cmd_line_chg(self,args):
        '''Change parameters away from default based on command line args'''
        
        # Expect args to be a list of command line arguments in the form
        #   <param name>=<new value>
        
        for argstr in args:
            arg,eq,val = argstr.partition('=')
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
                else:
                    raise ValueError('Unrecognized parameter.')
            except:
                print('Could not parse {0}.'.format(arg))
                raise
                
    def file_read_chg(self,filename):
        '''Read in parameters from a file'''
        pass
        
    def get_model_params(self):
        '''Return params in order necessary to run model, 
        minus day & wind_data'''
        hparams = (self.lam,*self.g_params,*self.f_params)
        return (hparams,self.Dparams,self.mu_r,self.n_periods,*self.domain_info)
        
    def get_wind_params(self):
        '''Return wind params to pass to PM.get_wind_data'''
        return (self.site_name,self.interp_num,self.start_time)
        
        

def main(argv):
    params = Params()
    # pull together some of the parameters
    # hparams = (lam,*g_params,*f_params)
    # get wind data and ordered list of days
    # wind_data,days = PM.get_wind_data(site_name,interp_num,start_time)
    wind_data,days = PM.get_wind_data(*params.get_wind_params())
    
    ### run model ###
    if params.ndays >= 0:
        ndays = params.ndays
    else:
        ndays = len(days)
        
    for day in days[:ndays]:
        print('Calculating spread for day {0}'.format(day))
        # get prob mass function for the current day and make it sparse.
        # pmf = PM.prob_mass(
            # day,wind_data,hparams,Dparams,mu_r,n_periods,*domain_info)
        pmf = PM.prob_mass(day,wind_data,*params.get_model_params())
        
        if day == days[0]:
            offset = params.domain_info[1] - pmf.shape[0]//2
            dom_len = params.domain_info[1]*2 + 1
            firstsol = sparse.coo_matrix((pmf.data, 
                (pmf.row+offset,pmf.col+offset)),shape=(dom_len,dom_len))
            modelsol = [firstsol]
        else:
            # convolute with previous day and then add solution
            print('Finding convolution with previous solution...')
            modelsol.append(PM.fftconv2(modelsol[-1],pmf))
    # done.
    print('Done.')
    
    ### save result ###
    print('Saving...')
    def outputGenerator():
        # Creates generator for output formatting
        for n,day in enumerate(days[:ndays]):
            yield (str(day)+'_data', modelsol[n].data)
            yield (str(day)+'_row', modelsol[n].row)
            yield (str(day)+'_col', modelsol[n].col)
        yield ('days',days[:ndays])
            
    outgen = outputGenerator()
    np.savez(params.outfile,**{x: y for (x,y) in outgen})
    
    ### save parameters ###
    with open(params.outfile+'.json','w') as fobj:
        json.dump(params.__dict__,fobj)
    
    ### plot result ###
    pass
    

if __name__ == "__main__":
    main(sys.argv[1:])