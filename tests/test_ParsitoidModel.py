"""
Test suite for ParasitoidModel, for use with py.test

Created on Fri May 08 12:12:19 2015

Author: Christopher Strickland
Email: wcstrick@live.unc.edu
"""

import pytest
import numpy as np
import math
from scipy import sparse, signal
import ParasitoidModel as PM
import globalvars

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

@pytest.fixture(scope="module")
def D_params():
    sig_x = 4.0 # std deviation in meters
    sig_y = 4.0
    corr = 0.
    return (sig_x,sig_y,corr)

@pytest.fixture(scope="module")
def flight_consts():
    # lambda constant in h_flight_prob
    lam = 1.0
    # meters to travel in advection per km/hr wind speed
    mu_r = 1 # scaling flight advection to wind advection
    # number of time periods (minutes) in one flight
    n_periods = 6
    return (lam,mu_r,n_periods)

@pytest.fixture(scope="module")
def site_name():
    return 'data/carnarvonearl'

@pytest.fixture(scope="module")
def start_time(site_name):
    if site_name == 'data/carnarvonearl':
        return '00:30'
    elif site_name == 'data/kalbar':
        return '00:00'

@pytest.fixture(scope="module")
def emerg_data(site_name):
    emerg_data = PM.emergence_data(site_name)
    return emerg_data

@pytest.fixture(scope="module")
def wind_data(site_name,start_time):
    wind_data,days = PM.get_wind_data(site_name,30,start_time)
    return wind_data

@pytest.fixture
def wind_data_days(site_name,start_time):
    return PM.get_wind_data(site_name,30,start_time)

############                    Decorators                ############

slow = pytest.mark.skipif(not pytest.config.getoption('--runslow'),
    reason = 'need --runslow option to run')

cuda_run = pytest.mark.skipif(not globalvars.cuda,
    reason = 'need globalvars.cuda == True')

###############################################################################
#                                                                             #
#                                   Tests                                     #
#                                                                             #
###############################################################################


def test_emerg_data(site_name):
    # Basic tests for expected structure in emerg_data dict
    emerg_data = PM.emergence_data(site_name)
    assert isinstance(emerg_data,dict)
    for field in emerg_data: #these should be fields.
        #emerg_data[field] is a dict
        for date in emerg_data[field]: #these should be dates since release
            assert isinstance(date,int)

def test_wind_data(site_name):
    # Basic tests for expected structure in wind_data dict and days list
    wind_data,days = PM.read_wind_file(site_name)
    assert isinstance(wind_data,dict)
    assert isinstance(days,list)
    for day in days:
        assert days.count(day) == 1
        assert day in wind_data
        assert all(days[ii] < days[ii+1] for ii in range(len(days)-1))
    for key in wind_data:
        assert isinstance(key,int) #date should be int since release
        assert wind_data[key].shape[1] == 3 #windx,windy,windr

def test_get_wind_data(site_name,start_time):
    interp_num = 30

    wind_data_raw,days_raw = PM.read_wind_file(site_name)
    wind_data,days = PM.get_wind_data(site_name,interp_num,start_time)

    assert wind_data[days[0]].shape[0] == interp_num*wind_data_raw[days[0]].shape[0]
    assert days_raw == days

    if start_time == '00:30':
        assert all(wind_data[days[0]][0,:] == wind_data_raw[days[0]][0,:])
        assert all(wind_data[days[0]][interp_num-1,:] ==
            wind_data_raw[days[0]][0,:])
        for ii in range(wind_data_raw[days[0]].shape[0]-1):
            assert all(wind_data[days[0]][interp_num*(1+ii),:] ==
                wind_data_raw[days[0]][ii,:])
        assert all(wind_data[days[1]][0,:] == wind_data_raw[days[0]][-1,:])
    elif start_time == '00:00':
        assert all(wind_data[days[-1]][-1,:] == wind_data_raw[days[-1]][-1,:])
        assert all(wind_data[days[-1]][-interp_num+1,:] ==
            wind_data_raw[days[-1]][-1,:])
        for ii in range(wind_data_raw[days[0]].shape[0]-1):
            assert all(wind_data[days[0]][interp_num*(ii),:] ==
                wind_data_raw[days[0]][ii,:])
    for key in wind_data_raw:
        assert key in wind_data
        assert all(np.sqrt(wind_data[key][:,0]**2+wind_data[key][:,1]**2) ==
            wind_data[key][:,2])

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

    # check that f is a probability mass function
    assert np.all(flight_prob >= 0)
    assert math.isclose(flight_prob.sum(),1)

    # no flights between 10 pm and midnight
    ii = 1
    while time_of_day[-ii] >= 22:
        try:
            assert flight_prob[-ii] < 0.01/n
        except:
            # Give some feedback on time of failure
            print('Time of failure: {0}\n'.format(time_of_day[-ii]))
            raise
        ii += 1

    # no flights between midnight and 3 am
    ii = 0
    while time_of_day[ii] <= 3:
        try:
            assert flight_prob[ii] < 0.01/n
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
            assert flight_prob[ii] > 0.99/n
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
        # get f and g to test for certain properties in comparison with h
        n = day_wind.shape[0] #number of wind data entries in the day
        #get just the windr values
        try:
            windr = day_wind[:,2]
        except IndexError:
            windr = day_wind[2] # for testing prob_mass
            n = 1
        f_func = PM.f_time_prob(n,*f_time_prob_params)
        g_func = PM.g_wind_prob(windr,*g_wind_prob_params)
        assert np.all(f_func*g_func <= f_func)
        assert (f_func-f_func*g_func).sum() <=1
        # get the probability function for the day
        flight_params = (*g_wind_prob_params,*f_time_prob_params)
        flight_prob = PM.h_flight_prob(day_wind,lam,
            *flight_params)
        # test that it has proper probability properties
        assert np.all(flight_prob >= 0)
        # should be add up to less than or equal to 1,
        #   1-sum is prob. of not flying
        assert flight_prob.sum() <= 1
        # we should be strictly adding probability to f_func*g_func
        assert np.all(flight_prob >= f_func*g_func)

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
    f_time_prob_params,D_params,flight_consts,domain_info):

    # day to test
    day = 1

    # lambda constant in h_flight_prob
    lam = flight_consts[0]

    #### Run over a single 30 min period to see what happens in detail ####

    #   to do this, we pass in wind_data with only a single time period in the
    #   first day.

    # Data has only day one, with one time period (chosen from middle of day)
    sing_wind_data = {1:wind_data[1][24*30,:]}
    sing_wind_data_cpy = dict(sing_wind_data)
    # Need to alter parameters to f function a bit to get probability of flying
    #   around midnight, when the time period will start...
    hparams1 = (lam,*g_wind_prob_params,-4.,2.,19.,2.)
    # This will give us one 24hr time period. mu_r has to scale accoringly
    mu_r1 = 0.1/24 # 6 min flight at full wind advection

    #pytest.set_trace()
    pmf = PM.prob_mass(1,sing_wind_data,hparams1,D_params,D_params,mu_r1,1,
                    *domain_info)
    pmf = pmf.tocsr()

    # Find the center. pmf is always square
    midpt = pmf.shape[0]//2

    # sing_wind_data is mutable. Verify that it is unchanged.
    assert sing_wind_data == sing_wind_data_cpy

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

    # DO THIS BLOCK LAST FOR SINGLE RUN! ALTERS pmf
    # Midday on the first day had wind. Most of the probability is around the
    #   origin because wind decreases the likelihood of flight, but other than
    #   this, much of the probabiilty should be away from the origin.
    # assert np.unravel_index(pmf.argmax(), pmf.shape) != (midpt,midpt)
    pmf[midpt,midpt] = 0
    assert 1 > pmf.sum() > 0


    #### Run for the entire day, using full wind_data dictionary ####

    # parameters for h_flight_prob
    hparams = (lam,*g_wind_prob_params,*f_time_prob_params)

    # wind_data is mutable. have a copy on hand to check against
    wind_data_cpy = dict(wind_data)

    # get the day's probability density for location of a parasitoid
    params = (*flight_consts[1:],*domain_info)
    pmf = PM.prob_mass(day,wind_data,hparams,D_params,D_params,*params)

    # wind_data should be unchanged
    assert wind_data == wind_data_cpy

    #check offseting algorithm
    offset = domain_info[1] - pmf.shape[0]//2
    dom_len = domain_info[1]*2 + 1
    firstsol = sparse.coo_matrix((pmf.data,
        (pmf.row+offset,pmf.col+offset)),shape=(dom_len,dom_len))

    # should be a probability mass function, before/after conversion
    firstsol = firstsol.tocsr()
    assert math.isclose(pmf.sum(),1)
    assert math.isclose(firstsol.sum(),1)

    # most of the probability should still be near the origin
    midpt = firstsol.shape[0]//2
    assert firstsol[midpt-4:midpt+5,midpt-4:midpt+5].sum() > (firstsol.sum()
        - firstsol[midpt-4:midpt+5,midpt-4:midpt+5].sum())
    # but not all
    assert not math.isclose(firstsol[midpt-4:midpt+5,midpt-4:midpt+5].sum(),1)



    #### Run again, but this time with noon release ####

    pmf2 = PM.prob_mass(day,wind_data,hparams,D_params,D_params,*params,0.5)

    # should be a probability mass function
    assert math.isclose(pmf2.sum(),1)

    # most of the probability should still be at the origin
    midpt2 = pmf2.shape[0]//2
    pmf2 = pmf2.tocsr()
    assert pmf2[midpt2-4:midpt2+5,midpt2-4:midpt2+5].sum() > (pmf2.sum()
        - pmf2[midpt2-4:midpt2+5,midpt2-4:midpt2+5].sum())
    # but not all
    assert not math.isclose(pmf2[midpt2-4:midpt2+5,midpt2-4:midpt2+5].sum(),1)

    # this solution should have more left at the origin than the first one
    assert pmf2[midpt2,midpt2] > firstsol[midpt,midpt]