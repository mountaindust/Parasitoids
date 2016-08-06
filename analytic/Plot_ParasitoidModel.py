#! /usr/bin/env python
"""
Plotting suite for ParasitoidModel

Created on Fri May 08 12:12:19 2015

Author: Christopher Strickland
"""

import numpy as np
import ParasitoidModel as PM
import matplotlib.pyplot as plt

#Here, we really want to solve two problems at the same time
# 1. test the functions in ParasitoidModel
# 2. play with them to get some reasonable dummy parameters

#SPATIAL DOMAIN:
# Release point will be placed in the center
# Domain will be defined in terms of distance from release and resolution

class Params():
    '''Class definition for parameter object.
    
    Objects from this class will hold all necessary parameters for the plotting
    routines in this file. A user needs only to update a parameter in the object
    and the results will be apparent when calling the functions.'''
    
    # dict of known site names and their start times
    known_sites = {'carnarvonearl': '00:30', 'kalbar': '00:00'}
    
    def __init__(self):
        ### DEFAULT PARAMETERS ###
        
        ### I/O
        self.rad_dist = 8000.0 #dist from release pt to side of the domain (meters)
        self.rad_res = 1600 #number of cells from center to side of domain
        # these default number correspond to each cell being 5 m**2
        
        
        # these should only be changed through set_site_name
        self._site_name = 'data/carnarvonearl'
        self._start_time = '00:30'
        self._emergence_data = PM.emergence_data(self._site_name)
        self.interp_num = 30 #number of interpolation points per wind data point
                             #30 will give 1 point = 1 min
        self.wind_data, self.days = PM.get_wind_data(
                            self._site_name,self.interp_num,self._start_time)
        
        
        ### Function parameters
        # take-off scaling based on wind
        # aw,bw: first scalar centers the logistic, second one stretches it.
        self.g_params = (2.2, 5)
        # take-off probability mass function based on time of day
        # a1,b1,a2,b2: a# scalar centers logistic, b# stretches it.
        self.f_params = (6, 3, 18, 3)
        # Diffusion coefficients, sig_x, sig_y, rho (units are meters)
        self.Dparams = (21,16,0)
        
        
        ### general flight parameters
        # Probability of any flight during the day under ideal circumstances
        self.lam = 1.
        # scaling flight advection to wind advection
        self.mu_r = 1.
        # number of time periods (based on interp_num) in one flight
        self.n_periods = 10 # if interp_num = 30, this is # of minutes
        
    def set_site_name(self,site_name):
        '''Change sites to a site name listed in self.known_sites'''
        self._start_time = self.known_sites[site_name]
        self._site_name = 'data/'+site_name
        self._emergence_data = PM.emergence_data(self._site_name)
        self.wind_data, self.days = PM.get_wind_data(
                            self._site_name,self.interp_num,self._start_time)
        
    def get_site_name(self):
        return self._site_name[5:]
        
    def get_start_time(self):
        return self._start_time
        

        
# This allows running without any setup.
params = Params()

#### Test g function for prob. during different wind speeds ####
def plot_g_wind_prob(params=params):
    aw,bw = params.g_params
    windr_range = np.arange(0,8,0.2) #a range of wind speeds
    g = PM.g_wind_prob(windr_range,aw,bw)
    plt.ion()
    plt.figure()
    #first scalar centers the logistic. Second one stretches it.
    plt.plot(windr_range,g)
    plt.xlabel('wind speed')
    plt.ylabel('probability of flight')
    plt.title('g func for prob of flight during given wind speed')
    # return g

#### Test f function for prob. during different times of the day    
def plot_f_time_prob(params=params):
    a1,b1,a2,b2 = params.f_params
    n = 24*60 #throw in a lot of increments to see a smooth 24hr plot
    day_time = np.linspace(0,24-24./n,n)
    #first scalar centers the logistic. Second one stretches it.
    #first set of two scalars is the first logistic
    f = PM.f_time_prob(n,a1,b1,a2,b2)
    plt.ion()
    plt.figure()
    plt.plot(day_time,f)
    plt.xlabel('time of day (hrs)')
    plt.ylabel('probability mass of flight')
    plt.title('f func for prob of flight during time of day')
    # return f
    
#### Test h function (and therefore g and f) with data ####
def plot_h_flight_prob(params=params,day=1):
    day_wind = params.wind_data[day]
    lam = params.lam
    day_time = np.linspace(0,24,day_wind.shape[0]+1)[:-1]
    hparams = (params.lam,*params.g_params,*params.f_params)
    h = PM.h_flight_prob(day_wind,*hparams)
    plt.ion()
    plt.figure()
    plt.plot(day_time,h)
    plt.xlabel('time of day (hrs)')
    plt.ylabel('probability density of flight')
    plt.title('h func for prob of flight given wind')
    # return h
    
#### Test p function, which gives the 2-D probability density####
def plot_prob_mass(params=params,day=1):
    wind_data = params.wind_data
    hparams = (params.lam,*params.g_params,*params.f_params)
    Dparams = params.Dparams
    mu_r = params.mu_r
    n_periods = params.n_periods
    rad_dist = params.rad_dist
    rad_res = params.rad_res
    pmf = PM.prob_mass(day,wind_data,hparams,Dparams,mu_r,n_periods,rad_dist,rad_res)
    #plt.pcolormesh is not practical on the full output. consumes 3.5GB of RAM
    #will need to implement resolution sensitive plotting
    
    # res is how far (# of cells) to plot away from the center
    res = (pmf.shape[0]-1)//2 # pmf is always square
    
    cell_dist = rad_dist/rad_res #dist from one cell to neighbor cell (meters).
    xmesh = np.arange(-res*cell_dist-cell_dist/2,res*cell_dist+cell_dist/2 + 
        cell_dist/3,cell_dist)
    # mask the view at negligible probabilities
    pmf_masked = np.ma.masked_less(pmf.toarray(),0.00001)
    # flip result for proper plotting orientation
    pmf_masked = np.flipud(pmf_masked)
    plt.ion()
    plt.figure()
    plt.pcolormesh(xmesh,xmesh,pmf_masked,cmap='viridis')
    plt.axis([xmesh[0],xmesh[-1],xmesh[0],xmesh[-1]])
    plt.xlabel('East-West (meters)')
    plt.ylabel('North-South (meters)')
    plt.title('Parasitoid prob. after one day')
    plt.colorbar()