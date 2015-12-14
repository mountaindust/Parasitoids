#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for ParasitoidModel

Created on Fri May 08 12:12:19 2015

:author: Christopher Strickland
"""

import numpy as np
import ParasitoidModel as PM
import matplotlib.pyplot as plt

# Implementation detail:
# Formation of each p(x,t_i), fft of each one, and ifft will need to be done in
#   parallel when doing Bayesian inference. fft of 8000^2 is 5.6 sec.
#   Want: number of processors = number of days to simulate

#load some emergence data
c_em = PM.emergence_data('carnarvonearl')
#load some wind data
wind_data = PM.read_wind_file('carnarvonearl')

#SPATIAL DOMAIN:
# Release point will be placed in the center
# Domain will be defined in terms of distance from release and resolution
# FUTURE: assign cells UTM values when plotting
rad_dist = 8000.0 #distance from release point to a side of the domain (meters)
rad_res = 4000.0 #number of cells from center to side of domain


dom_len = rad_res*2+1 #number of cells along one dimension of domain
dom_ticks = np.linspace(-rad_dist,rad_dist,dom_len) #label the center of each cell
                                                    #center cell is 0
cell_dist = rad_dist/rad_res #dist from one cell to neighbor cell.

#Here, we really want to solve two problems at the same time
# 1. test the functions in ParasitoidModel
# 2. play with them to get some reasonable dummy parameters

#### Test g function for prob. during different wind speeds ####
def test_g(aw=1.8,bw=6):
    windr_range = np.arange(0,3.1,0.1) #a range of wind speeds
    plt.figure()
    #first scalar centers the logistic. Second one stretches it.
    plt.plot(windr_range,PM.g_wind_prob(windr_range,aw,bw))
    plt.xlabel('wind speed')
    plt.ylabel('probability of flight')
    plt.title('g func for prob of flight during given wind speed')
    plt.show()

#### Test f function for prob. during different times of the day    
def test_f(a1=7,b1=1.5,a2=19,b2=1.5):
    n = 240 #throw in a lot of increments to see a smooth 24hr plot
    day_time = np.linspace(0,24,n)
    #first scalar centers the logistic. Second one stretches it.
    #first set of two scalars is the first logistic
    plt.plot(day_time,PM.f_time_prob(n,a1,b1,a2,b2))
    plt.xlabel('time of day (hrs)')
    plt.ylabel('probability density of flight')
    plt.title('f func for prob of flight during time of day')
    plt.show()
    
#### Test h function (and therefore g and f) with data ####
def test_h(day_wind=wind_data[1],lam=1.1):
    day_time = np.linspace(0,24,wind_data[1].shape[0])
    plt.figure()
    plt.plot(day_time,PM.h_flight_prob(day_wind,lam,1.8,6,7,1.5,19,1.5))
    plt.xlabel('time of day (hrs)')
    plt.ylabel('probability density of flight')
    plt.title('h func for prob of flight given wind')
    plt.show()
    
#### Test p function, which gives the 2-D probability density####
hparams = (1.1, 1.8, 6, 7, 1.5, 19, 1.5)
Dparams = (1, 1, 0)
# This seems to be returning an array that sums to a value less than one.
#   Should sum to one as a probability density?
def test_p(day=1,wind_data=wind_data,hparams=hparams,Dparams=Dparams,mu_r=1,\
rad_dist=rad_dist,rad_res=rad_res):
    ppdf = PM.prob_density(day,wind_data,hparams,Dparams,mu_r,rad_dist,rad_res)
    #plt.pcolormesh is not practical on the full output. consumes 3.5GB of RAM
    #will need to implement resolution sensitive plotting
    return ppdf