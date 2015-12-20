# -*- coding: utf-8 -*-
"""
Test suite for ParasitoidModel, for use with py.test

Created on Fri May 08 12:12:19 2015

:author: Christopher Strickland
"""

import pytest
import numpy as np
import math
import ParasitoidModel as PM

###############################################################################
#                                                                             #
#                              Test Fixtures                                  #  
#                                                                             #
###############################################################################

@pytest.fixture
def g_wind_prob_params():
    # Return logistic shape parameters for the g_wind_prob function
    aw = 1.8
    bw = 6
    return (aw,bw)
    
@pytest.fixture
def f_time_prob_params():
    # Return logistic parameters for the f_time_prob function, shape and bias
    a1 =7.
    b1 = 2.
    a2 = 19.
    b2 = 2.
    return (a1,b1,a2,b2)

@pytest.fixture    
def domain_info():
    # Return infomation about the domain size and refinement
    
    # distance from release point to a side of the domain in meters
    rad_dist = 8000.0
    # number of cells from the center to side of domain
    rad_res = 4000
    
    # (each cell is thus rad_dist/rad_res meters squared in size)
    return (rad_dist,rad_res)

@pytest.fixture(scope="module") #run only once
def emerg_data():
    emerg_data = PM.emergence_data('carnarvonearl')
    return emerg_data

@pytest.fixture(scope="module") 
def wind_data():
    wind_data = PM.read_wind_file('carnarvonearl')
    return wind_data


###############################################################################
#                                                                             #
#                                   Tests                                     #
#                                                                             #
###############################################################################
    
    
def test_emerg_data(emerg_data):
    # Basic tests for expected structure in emerg_data dict
    assert isinstance(emerg_data,dict)
    for field in emerg_data: #these should be fields.
        #emerg_data[field] is a dict
        for date in emerg_data[field]: #these should be dates since release
            assert isinstance(date,int)
    
def test_wind_data(wind_data):
    # Basic tests for expected structure in wind_data dict
    assert isinstance(wind_data,dict)
    for key in wind_data:
        assert isinstance(key,int) #date should be int since release
        assert wind_data[key].shape[1] == 4 #windx,windy,windr,theta

def test_g_prob_of_flying_by_wind_speed(g_wind_prob_params):
    ds = 0.1 #interval at which to test function
    
    # pass in a bunch of values to the function being tested,
    #   get out probability scaling
    flight_prob = PM.g_wind_prob(np.arange(0,3.1,ds),*g_wind_prob_params)
    
    # test for desirable properties #
    
    # check that all scaling values are between 0 and 1
    assert (np.all(0 <= flight_prob) and np.all(flight_prob <= 1))
    # should be a strictly decreasing function of wind speed
    for ii in range(flight_prob.size-1):
        assert flight_prob[ii] > flight_prob[ii+1]
    # low wind speeds should have no effect
    low = 0.5 #upper bound on a "low" wind speed
    #check that these low wind speeds return a value close to 1
    assert np.all(flight_prob[0:int(low/ds)+1] > 0.99)
    
def test_f_prob_of_flying_by_time_of_day(f_time_prob_params):
    # number of discrete times per day to look at
    n = 48 # data is every 30 min
    time_of_day = np.linspace(0,24,n)
    
    # get probability scaling
    flight_prob = PM.f_time_prob(n,*f_time_prob_params)
    
    # test for desirable properties #
    
    # check that all scaling values are between 0 and 1
    assert (np.all(0 <= flight_prob) and np.all(flight_prob <= 1))
    
    # no flights between 10 pm and midnight
    ii = 1
    while time_of_day[-ii] >= 22:
        try:
            assert flight_prob[-ii] < 0.01
        except:
            # Give some feedback on time of failure
            print('Time of failure: {0}\n'.format(time_of_day[-ii]))
            raise
        ii += 1
        
    # no flights between midnight and 3 am
    ii = 0
    while time_of_day[ii] <= 3:
        try:
            assert flight_prob[ii] < 0.01
        except:
            # Give some feedback on time of failure
            print('Time of failure: {0}\n'.format(time_of_day[ii]))
            raise
        ii += 1
        
    # no penalty between 11 am and 3 pm
    ii = 0
    while time_of_day[ii] < 11:
        # find 11:00
        ii += 1
    while time_of_day[ii] <= 15:
        try:
            assert flight_prob[ii] > 0.99
        except:
            print('Time of failure: {0}\n'.format(time_of_day[ii]))
            raise
        ii += 1
    
def test_h_flight_prob(wind_data,g_wind_prob_params,f_time_prob_params):
    # lambda constant controlling probability of flying in a given day
    #   under ideal conditions.
    lam = 1.0
    
    # try a few days of wind data
    for ii in range(1,4):
        day_wind = wind_data[ii]
        # get the probability function for the day
        flight_prob = PM.h_flight_prob(day_wind,lam,
            *g_wind_prob_params,*f_time_prob_params)
        # test that it has proper probability properties
        assert np.all(flight_prob >= 0)
        assert flight_prob.sum() <= 1
    
def test_prob_density_after_one_day(wind_data,g_wind_prob_params,
    f_time_prob_params,domain_info):
    
    # day to test
    day = 1
    # lambda constant in h_flight_prob
    lam = 1.0
    # parameters for diffusion covariance matrix, (sig_x,sig_y,rho)
    Dparams = (1., 1., 0.0)
    # meters to travel in advection per m/hr wind speed
    # TODO: EACH TIME PERIOD IS TYPICALLY 30 MIN. ARE WE DIVIDING BY 2 ?????
    mu_r = 1.
    
    # parameters for h_flight_prob
    hparams = (lam,*g_wind_prob_params,*f_time_prob_params)
    
    # get the day's probability density for location of a parasitoid
    ppdf = PM.prob_density(day,wind_data,hparams,Dparams,mu_r,*domain_info)
    
    # should be a probability density
    assert math.isclose(ppdf.sum(),1)
    # need more tests here...