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
    lam = 1.
    
    # try a few days of wind data
    for ii in range(1,4):
        day_wind = wind_data[ii]
        # get the probability function for the day
        flight_prob = PM.h_flight_prob(day_wind,lam,
            *g_wind_prob_params,*f_time_prob_params)
        # test that it has proper probability density properties
        assert np.all(flight_prob >= 0)
        #integral divided by lambda should be equal to 1
        assert math.isclose(flight_prob.sum()*24/day_wind.shape[0]/lam,1)
        
def test_get_mvn_cdf_values():
    # This test should make sure (x,y) coordinate pairs are correctly
    #   translated to row/columns, among other things.
    
    # Use a covarience matrix with some correlation. Test one with high varience
    #   vs. one with small varience to make sure that the adaptive integration
    #   is working properly.
    
    cell_length = 2
    mu = np.zeros(2)
    
    sig_x1 = 4; sig_y1 = 4 # (in meters)
    corr1 = 0.5
    S1 = np.array([[sig_x1**2, corr1*sig_x1*sig_y1],
                   [corr1*sig_x1*sig_y1, sig_y1**2]])
    # bigger
    sig_x2 = 10; sig_y2 = 10
    corr2 = -corr1
    S2 = np.array([[sig_x2**2, corr2*sig_x2*sig_y2],
                   [corr2*sig_x2*sig_y2, sig_y2**2]])
                   
    # Get cdf values
    cdf_mat1 = PM.get_mvn_cdf_values(cell_length,mu,S1)
    cdf_mat2 = PM.get_mvn_cdf_values(cell_length,mu,S2)
    
    # should behave like an approximation to a probability mass function
    assert 0.99 < cdf_mat1.sum() < 1
    assert 0.99 < cdf_mat2.sum() < 1
    
    # 2 should be bigger than 1
    assert cdf_mat2.size > cdf_mat1.size
    
    cdf_len = cdf_mat1.shape[0] #odd number
    cdf_cent = int(cdf_len/2) #center of cdf_mat
    # With positive correlation, we expect more probability in the first and
    #   third quadrants.
    # compare 2nd quadrant and 1st quadrant
    assert cdf_mat1[0:cdf_cent,0:cdf_cent].sum() < \
           cdf_mat1[0:cdf_cent,cdf_cent+1:].sum()
    # With negative correlation, we expect more probability in the second and
    #   fourth quadrants.
    # compare 3rd quadrant and 4th quadrant
    assert cdf_mat2[cdf_cent+1:,0:cdf_cent].sum() < \
           cdf_mat1[cdf_cent+1:,cdf_cent+1:].sum()
    
    # The mean is within the origin cell, so this should be the location with 
    #   the most probability
    mdpt = int(cdf_mat1.shape[0]/2) #shape is an odd number. flooring should
                                    #   get us where we want in a 0-based index
    assert cdf_mat1.max() == cdf_mat1[mdpt,mdpt]
    
    
    
def test_prob_mass_func_generation(wind_data,g_wind_prob_params,
    f_time_prob_params,domain_info):
    
    # day to test
    day = 1
    # lambda constant in h_flight_prob
    lam = 1.0
    # parameters for diffusion covariance matrix, (sig_x,sig_y,rho)
    Dparams = (1., 1., 0.0)
    # meters to travel in advection per km/hr wind speed
    mu_r = 0.2 #maybe 6 min total of flight time per day?
    
    midpt = domain_info[1] #this is rad_res, the center
    
    #### Run over a single 30 min period to see what happens in detail ####
    
    #   to do this, we pass in wind_data with only a single time period in the
    #   first day.
    
    # Data has only day one, with one time period (chosen from middle of day)
    sing_wind_data = {1:wind_data[1][24,:]}
    # Need to alter parameters to f function a bit to get probability of flying
    #   around midnight, when the time period will start...
    hparams1 = (lam,*g_wind_prob_params,-4.,2.,19.,2.)
    # This will give us one 24hr time period. mu_r has to scale accoringly
    mu_r1 = mu_r/48
    
    #pytest.set_trace()
    pmf = PM.prob_mass(1,sing_wind_data,hparams1,Dparams,mu_r1,*domain_info)
    
    # Check that the shifted normal distribution is in the correct quadrant
    #   given the wind vector's direction
    wind_sign = np.sign(sing_wind_data[1][0:2]) #signum, (x,y)
    if wind_sign[0] < 0: # x < 0, column < midpt
        if wind_sign[1] < 0: # y < 0, row > midpt
            assert pmf[midpt+5:,0:midpt-5].sum() > 0
        else: # y > 0, row < midpt
            assert pmf[0:midpt-5,0:midpt-5].sum() > 0
    else: # x > 0, column > midpt
        if wind_sign[1] < 0: # y < 0, row > midpt
            assert pmf[midpt+5:,midpt+5:].sum() > 0
        else: # y > 0, row < midpt
            assert pmf[0:midpt-5,midpt+5:].sum() > 0
    
    # DO THIS BLOCK LAST! ALTERS pmf
    # Midday on the first day had wind. Most of the probability will be at the
    #   origin because wind decreases the likelihood of flight, but other than
    #   this point, most of the probabiilty should be away from the origin.
    # assert np.unravel_index(pmf.argmax(), pmf.shape) != (midpt,midpt)
    pmf[midpt,midpt] = 0
    assert pmf.sum() > 0
    assert pmf[midpt-5:midpt+6,midpt-5:midpt+6].sum() == 0
    
    
    #### Run for the entire day, using full wind_data dictionary ####
    
    # parameters for h_flight_prob
    hparams = (lam,*g_wind_prob_params,*f_time_prob_params)
    
    # get the day's probability density for location of a parasitoid
    pmf = PM.prob_mass(day,wind_data,hparams,Dparams,mu_r,*domain_info)
    
    # should be a probability mass function
    assert math.isclose(pmf.sum(),1)
    # need more tests here...