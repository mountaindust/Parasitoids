# -*- coding: utf-8 -*-
"""
Drift-Diffusion Model
This module should implement the pieces of the model, including info about the
    spatial mesh. These functions will then be called from an external module,
    either for running the model or Bayesian inference.

Created on Sat Mar 07 20:18:32 2015

Author: Christopher Strickland  
Email: cstrickland@samsi.info  
"""

__author__ = "Christopher Strickland"
__email__ = "cstrickland@samsi.info"
__status__ = "Development"
__version__ = "1.1"
__copyright__ = "Copyright 2015, Christopher Strickland"

from math import floor
import numpy as np
from scipy.stats import mvn
from scipy import sparse
from CalcSol import r_small_vals

#we need to fix units for time. lets say t is in hours.

def emergence_data(site_name):
    """ Reads the observed emergence data from a text file.
    
    Arguments:
        - site_name -- string
        
    Returns:
        - dictionary of emergence data. Contains dicts of each field, whose keys
            are the dates of sampling."""
    em = {}

    file_name = site_name + 'emergence.txt'
    
    with open(file_name,'r') as em_file:
        #First line is column headers.
        comment_line = em_file.readline() # e.g. #date   0  22   25   ...
        split_comment_line = comment_line.split() #split by whitespace
        split_comment_line.pop(0) # remove '#date' label
        #Create a dict within em for each field
        for field in split_comment_line:
            em[field] = {}
        #Loop over each row of data
        for line in em_file.readlines(): # e.g. 01  2  0    0    ...
            split_line = line.split() #split by whitespace
            #first column is the day since release. pop off and covert to int
            date = int(split_line.pop(0))
            #cycle through each field (each column is a field)
            for ind in range(len(split_line)): # skips release field -- no it doesn't...
                field = split_comment_line[ind] # get this field's name
                # Assign the data to the date key within the field's dict
                em[field][date] = int(split_line[ind]) # em[field_name][date] = value

    return em

    
    
def read_wind_file(site_name):
    """ Reads the wind data from a text file
    
    Arguments:
        - site_name -- string
        
    Returns:
        - wind_data -- wind data as a dictionary of 2D ndarrays. 
                       Keys are date since release. In each ndarray, 
                       rows are times, columns are the windx,windy,windr
        - days -- sorted list of days found in wind file"""
    file_name = site_name + 'wind.txt'
    with open(file_name) as wind_file:
        wind_data = {}
        days = []
        for line in wind_file.readlines():
            # File has data like this: day x-component y-component
            splitline = line.split()
            day = int(splitline[0]) #date as an int
            windx = float(splitline[1]) #x-component of wind vector
            # Remove very small values
            if abs(windx) < 10e-5:
                windx = 0
            windy = float(splitline[2]) #y-component of wind vector
            # Remove very small values
            if abs(windy) < 10e-5: 
                windy = 0
            # Magnitude of wind vector, r
            windr = np.sqrt(windx**2+windy**2)
            # Remove very small values
            if abs(windr) < 10e-5: 
                windr = 0
                
            # Wind angle was unused, so this block commented.
            # Theta has also been removed from the output of this function
            # # Angle of wind vector, theta
            # if (windx == 0) and (windy == 0):
                # theta = 0
            # elif (windx == 0) and (windy > 0):
                # theta = np.pi/2
            # elif (windx == 0) and (windy < 0):
                # theta = -np.pi/2
            # else:
                # theta = np.arctan(windy/windx)
            # if windx < 0:
                # theta = theta+np.pi
                
            # Add to our wind_data dictionary.
            # Each date is an ordered list of wind data (ndarray) by hour.
            if day in wind_data:
                wind_data[day].append(np.array([windx,windy,windr]))
            else:
                wind_data[day] = [np.array([windx,windy,windr])]
                days.append(day)
    
    #convert each list of ndarrays to a single ndarray where rows are times,
    #  columns are the windx,windy,windr. This allows fancy slicing.
    for day in wind_data:
        wind_data[day] = np.array(wind_data[day]) #lists of ndarrays become 2D
        
    days.sort()

    return (wind_data,days)
    #this returns a dictionary of days, with each day pointing to
    #a 2-D ndarray. Rows are times, columns are the windx,windy,windr

    
    
#there is abiguity as to whether or not midnight belongs to one day or the other
#   In fact, one of our data sets starts at 00:00, the other at 00:30! Each,
#   however, keeps 48 30min periods in a day. So we will have the user specify
#   at which time the data set starts, and infer the convention used accordingly    
def get_wind_data(site_name,interp_num,start_time):
    '''Calls read_wind_file, interpolates the data linearly.
    This function will also handle the midnight vs. 00:30 starting problem.
    We need a convention for output, so we will say that the day starts at 00:00
    and runs until 23:59.
    
    This procedure will not be under MCMC, so it doesn't have to really fast.
    
    Args:
        site_name: string
        interp_num: number of time points to have in each data-point interval,
            [data1,data2), including the data point itself.
        start_time: string, '00:00' or '00:30', time of first data point
        
    Returns:
        wind_data: dictionary of wind arrays, one array for each day.
                   each row is one time point, each column is windx,windy,windr'''
                   
    wind_data_raw,days = read_wind_file(site_name)
    
    # No matter if the data starts at 00:00 or 00:30, we have a fencepost problem
    #   in linearly interpolating our data either at the beginning or the end.
    #   Fortunately, this occurs at the middle of the night, so it shouldn't
    #   make much of a difference.
    
    wind_data = {}
    time_pts = wind_data_raw[days[0]].shape[0]
    scaling = np.linspace(0,1,interp_num+1)[:-1]
    scaling_mat = np.tile(scaling,(3,1)).T
    scaling_mat_dec = 1 - scaling_mat
    
    if start_time == '00:00':
        for day in days[:-1]:
            interp_wind = np.zeros((time_pts*interp_num,3))
            for data_indx in range(time_pts-1):
                interp_wind[data_indx*interp_num:(data_indx+1)*interp_num,:] = (
                    scaling_mat_dec*wind_data_raw[day][data_indx,:] + 
                    scaling_mat*wind_data_raw[day][data_indx+1,:])
                # this calculation is incorrect for windr (triangle inequality).
            interp_wind[(time_pts-1)*interp_num:,:] = (
                scaling_mat_dec*wind_data_raw[day][-1,:] +
                scaling_mat*wind_data_raw[day+1][0,:])
            # recalculate windr before adding to wind_data
            interp_wind[:,2] = np.sqrt(interp_wind[:,0]**2 + interp_wind[:,1]**2)
            wind_data[day] = interp_wind
        
        # last day
        interp_wind = np.zeros((time_pts*interp_num,3))
        day = days[-1]
        for data_indx in range(time_pts-1):
            interp_wind[data_indx*interp_num:(data_indx+1)*interp_num,:] = (
                scaling_mat_dec*wind_data_raw[day][data_indx,:] + 
                scaling_mat*wind_data_raw[day][data_indx+1,:])
        # recalculate windr before adding to wind_data
        interp_wind[:,2] = np.sqrt(interp_wind[:,0]**2 + interp_wind[:,1]**2)
        # just repeat the last data point throughout the last interpolation period
        interp_wind[(time_pts-1)*interp_num:,:] = wind_data_raw[day][-1,:]
        wind_data[day] = interp_wind
        
    elif start_time == '00:30':
        # in this case, we assume that midnight is included in the previous day
        interp_wind = np.zeros((time_pts*interp_num,3))
        day = days[0]
        # just repeat backward the first data point for the interpolation period
        interp_wind[:interp_num,:] = wind_data_raw[day][0,:]
        for data_indx in range(time_pts-1):
            interp_wind[(data_indx+1)*interp_num:(data_indx+2)*interp_num,:] = (
                scaling_mat_dec*wind_data_raw[day][data_indx,:] + 
                scaling_mat*wind_data_raw[day][data_indx+1,:])
        # recalculate windr before adding to wind_data
        interp_wind[:,2] = np.sqrt(interp_wind[:,0]**2 + interp_wind[:,1]**2)
        wind_data[day] = interp_wind
        
        # after first day
        for day in days[1:]:
            interp_wind = np.zeros((time_pts*interp_num,3))
            interp_wind[:interp_num,:] = (
                scaling_mat_dec*wind_data_raw[day-1][-1,:] + 
                scaling_mat*wind_data_raw[day][0,:])
            for data_indx in range(time_pts-1):
                interp_wind[(data_indx+1)*interp_num:(data_indx+2)*interp_num,:] = (
                    scaling_mat_dec*wind_data_raw[day][data_indx,:] + 
                    scaling_mat*wind_data_raw[day][data_indx+1,:])
            # recalculate windr before adding to wind_data
            interp_wind[:,2] = np.sqrt(interp_wind[:,0]**2 + interp_wind[:,1]**2)
            wind_data[day] = interp_wind
            
    else:
        raise ValueError("start_time must be either '00:00' or '00:30'")
        
    return (wind_data,days)

##########    Model functions    ##########

def g_wind_prob(windr, aw, bw):
    """Returns probability of take-off under given wind conditions.
    If the wind has no effect on flight probability, returns 1.
    Otherwise, scales down from 1 to 0 as wind speed increases.
    
    Arguments:
        - windr -- wind speed
        - aw -- wind speed at which flight prob. is scaled by 0.5
        - bw -- steepness at which scaling changes (larger numbers = steeper)"""
    return 1.0 / (1. + np.exp(bw * (windr - aw)))

#Probability of flying at n discrete times of the day, equally spaced
def f_time_prob(n, a1, b1, a2, b2):
    """Returns probability mass function of take-off based on time at n equally
    spaced times.
    
    Arguments:
        - n -- number of wind data points per day available
        - a1 -- time of morning at which to return 0.5
        - b1 -- steepness of morning curve (larger numbers = steeper)
        - a2 -- time of afternoon at which to return 0.5
        - b2 -- steepness of afternoon curve (larger numbers = steeper)
        
    Returns:
        - A probability mass function for take-off in each of n intervals
            during the day"""

    # t is in hours, and denotes start time of flight.
    # By convention, we will start each day at 00:00:00.
    t_tild = np.linspace(0,24-24./n,n) #divide 24 hours into n equally spaced times
    # Calculate the likelihood of flight at each time of day, giving a number
    #   between 0 and 1. Combination of two logistic functions.
    likelihood = np.fmax(1.0 / (1. + np.exp(-b1 * (t_tild - a1))) - 
                    1.0 / (1. + np.exp(-b2 * (t_tild - a2))),
                    np.zeros_like(t_tild))
    # Scale the likelihood into a proper probability mass function, and return
    return likelihood/likelihood.sum()

def Dmat(sig_x, sig_y, rho):
    """Returns covarience matrix for diffusion process
    
    Arguments:
        - sig_x, sig_y -- Std. deviation in x and y direction respectively
        - rho -- Covariance"""
    
    assert sig_x > 0, 'sig_x must be positive'
    assert sig_y > 0, 'sig_y must be positive'
    assert -1 <= rho <= 1, 'correlation must be between -1 and 1'    
    return np.array([[sig_x**2, rho*sig_x*sig_y],\
                     [rho*sig_x*sig_y, sig_y**2]])
    
def h_flight_prob(day_wind, lam, aw, bw, a1, b1, a2, b2):
    """Returns probability density of flying (take-off) during a given day's wind.
    This is given by f times g times the constant lambda. Lambda can be thought
    of as the probability of flight during a day with constant ideal wind
    
    Arguments:
        - day_wind -- ndarray of wind directions
        - lam -- constant
        - aw,bw -- g function constants
        - a1,b1,a2,b2 -- f function constants
    
    Note: day_wind[0,:] = np.array([windx,windy,windr])"""
    
    n = day_wind.shape[0] #number of wind data entries in the day
    alpha_pow = 1 # new parameter?
    #get just the windr values
    try:
        windr = day_wind[:,2]
    except IndexError:
        windr = day_wind[2] # for testing prob_mass
        n = 1
    f_func = f_time_prob(n,a1,b1,a2,b2)
    g_func = g_wind_prob(windr,aw,bw)
    t_vec = np.linspace(1,n,n)
    integral_avg = f_func*g_func/t_vec/np.max(f_func)*np.cumsum(
        (1-np.cumsum(f_func)**alpha_pow)*(f_func-f_func*g_func))
    
    return f_func*g_func + integral_avg #np.array of length n

def get_mvn_cdf_values(cell_length,mu,S):
    """Get cdf values for a multivariate normal centered near (0,0) 
    inside regular cells. To do this fast, we use a secret Fortran mulivariate
    normal CDF (mvn) due to Dr. Alan Genz.
    
    This function will return a variable sized 2D array with its shape
    dependent on the support of the normal distribution.
    
    This function cannot be jit nopython compiled with numba due to mvn
    
    Args:
        cell_length: length of a side of each cell (in meters)
        mu: mean of the distribution (in meters)
        S: covariance matrix
        
    Returns:
        cdf_mat: 2D array of cdf values, one for each cell"""
    
    cdf_eps = 0.001    # integrate until the area of the square is within
                        #   cdf_eps of 1.0
    
    r = cell_length/2 # in meters. will want to integrate +/- this amount
    h = 0 # h*2+1 is the length of one side of the support in cells (int).
    
    cell_length_ary = np.array([cell_length,cell_length])
    
    # Integrate center cell
    low = np.array([-r,-r])
    upp = np.array([r,r])
    val, inform = mvn.mvnun(low,upp,mu,S)
    assert inform == 0 # integration finished with error < EPS
    # cdf_vals is a dict that takes x,y coordinate pairs (cell center locations)
    #   to probability mass values.
    cdf_vals = {(0,0):val}
    val_sum = val # keep track of the total sum of the integration
    
    # Start loop
    while 1 - val_sum >= cdf_eps:
        h += 1 # increase the size of the domain
        
        # Integrate the four corners of the square
        for ii in [-h,h]:
            for jj in [-h,h]:
                low = np.array([ii*cell_length-r,jj*cell_length-r])
                upp = low + cell_length_ary
                val, inform = mvn.mvnun(low,upp,mu,S)
                assert inform == 0 #integration finished with error < EPS
                cdf_vals[(ii,jj)] = val
                val_sum += val
                
        # Integrate the four sides of the square
        for ii in [-h,h]:
            for jj in range(-h+1,h):
                low = np.array([ii*cell_length-r,jj*cell_length-r])
                upp = low + cell_length_ary
                val, inform = mvn.mvnun(low,upp,mu,S)
                assert inform == 0 #integration finished with error < EPS
                cdf_vals[(ii,jj)] = val
                val_sum += val
                val, inform = mvn.mvnun(low[::-1],upp[::-1],mu,S)
                assert inform == 0 #integration finished with error < EPS
                cdf_vals[(jj,ii)] = val
                val_sum += val
        
    # We've now integrated to the required accuracy. Form an ndarray.
    # We need to translate x,y coordinate pairs to row and column numbers
    cdf_mat = np.array([[cdf_vals[(x,y)] for x in range(-h,h+1)] 
        for y in range(h,-h-1,-1)])
        
    return cdf_mat



class BndsError(Exception):
    def __str__(self):
        return 'Index error in calculating prob_mass.\n'+\
            'Most likely, wind results in spread that goes off the domain'+\
            ' in a single time period.'
    
    
    
def prob_mass(day,wind_data,hparams,Dparams,Dlparams,mu_r,mu_l_r,n_periods,
                rad_dist,rad_res,start_time=None):
    """Returns prob mass function for a given day as an ndarray.
    This function always is calculated based on an initial condition at the
    origin. The final position of all wasps based on the previous day's
    position can then be updated via convolution with this function.
    
    Arguments:
        - day -- day as specified in wind data
        - wind_data -- dictionary of wind data
        - hparams -- parameters for h_flight_prob(...). (lam,aw,bw,a1,b1,a2,b2)
        - Dparams -- parameters for Dmat(...). (sig_x,sig_y,rho)
        - Dlprams -- out-of-flow diffusion coefficients
        - mu_r -- parameter to scale distance vs. windspeed
        - n_periods -- number of time periods in flight duration. int
        - rad_dist -- distance from release point to side of the domain (m)
        - rad_res -- number of cells from center to side of the domain
        - start_time -- (optional) time at which release occurred (units=day)
        
    Returns:
        - pmf -- 2D spatial probability mass function of finding the parasitoid
                    in each spatial cell according to given resolution"""
        
    dom_len = rad_res*2+1 #number of cells along one dimension of domain
    cell_dist = rad_dist/rad_res #dist from one cell to neighbor cell.
        
    pmf = np.zeros((dom_len,dom_len))
    
    day_wind = wind_data[day] #alias the current day
    
    hprob = h_flight_prob(day_wind, *hparams)
    
    S = Dmat(*Dparams) #get diffusion covarience matrix
    Sl = Dmat(*Dlparams) # get out-of-flow diffusion covarience matrix
    
    # Check for single (primarily for testing) vs. multiple time periods
    if day_wind.ndim > 1:
        periods = day_wind.shape[0] # wind data is already interpolated.
                                    #   it starts at 00:00
        TEST_RUN = False
    else:
        periods = 1
        TEST_RUN = True
    
    if start_time is None:
        start_indx = 0
    else:
        start_indx = floor(start_time*periods)
        
    #################### Day Loop ####################
    for t_indx in range(start_indx,periods):
        ###                                                             ###
        ### Get the advection velocity and put in units = m/(unit time) ###
        ###                                                             ###
        if (not TEST_RUN) and n_periods > 1:
            if t_indx+n_periods-1 < periods:
                # effective flight advection over n periods in km/hr
                mu_v = np.sum(day_wind[t_indx:t_indx+n_periods,0:2],0)/n_periods
            elif day+1 in wind_data:
                # wrap sum into next day
                if t_indx != periods-1:
                    mu_v = np.sum(day_wind[t_indx:,0:2],0)
                else:
                    mu_v = np.array(day_wind[-1,0:2])
                wrap_periods = n_periods - (periods - t_indx)
                if wrap_periods != 1:
                    mu_v += np.sum(wind_data[day+1][:wrap_periods,0:2],0)
                else:
                    mu_v += wind_data[day+1][0,0:2]
                mu_v /= n_periods
            else:
                # last day in the data. Just extrapolate what's there to full time
                if t_indx != periods-1:
                    mu_v = np.sum(day_wind[t_indx:,0:2],0)/(periods-t_indx)
                else:
                    mu_v = np.array(day_wind[-1,0:2])
        elif not TEST_RUN:
            mu_v = np.array(day_wind[t_indx,0:2])
        else:
            # mu_v only has one entry. Typically a testing run to check behavior
            mu_v = np.array(day_wind[0:2])
        
        # mu_v is now in km/hr. convert to m/(n_periods)     
        mu_v *= 1000*24/(periods/n_periods) # m/(n_periods)
        
        # We also need to scale this by a constant which represents a scaling 
        # term that takes wind advection to flight advection.
        mu_v *= mu_r
        # Note: this is in (x,y) coordinates
        
        ###                                                         ###
        ### calculate spatial integral (in a spatially limited way) ###
        ###                                                         ###
        '''We know the distribution is centered around mu_v(t) at each t_indx, and
           that it has very limited support. Translate the normal distribution
           back to the origin and let get_mvn_cdf_values integrate until
           the support is exhausted.'''
        
        #   We will translate to the nearest cell center given by mu_v below.
        #   Pass the remainder of the translation to get_mvn_cdf_values as mu.
        cdf_mu = mu_v - np.round(mu_v/cell_dist)*cell_dist
        cdf_mat = get_mvn_cdf_values(cell_dist,cdf_mu,S)
        #translate mu_v from (x,y) coordinates to nearest cell-center location.
        #   [rad_res,rad_res] is the center cell of the domain.
        col_offset = int(np.round(mu_v[0]/cell_dist))
        row_offset = int(np.round(-mu_v[1]/cell_dist))
        row_cent = rad_res+row_offset
        col_cent = rad_res+col_offset
        adv_cent = np.array([row_cent,col_cent])
        
        ### now we want to plop the normal distribution around this center ###
        
        norm_r = int(cdf_mat.shape[0]/2) #shape[0] is odd, floor half of it.
        # Get indices in pmf array
        row_min, col_min = adv_cent - norm_r
        row_max, col_max = adv_cent + norm_r
        
        ### approximate integral over time ###
        
        # Return a unique error if the domain is not big enough so that it can
        # be handled in context.
        if row_max+1>pmf.shape[0] or col_max+1>pmf.shape[1] or \
            row_min < 0 or col_min < 0:
            raise BndsError
        try:
            assert -1e-9 <= hprob[t_indx] <= 1.000000001, \
                'hprob out of bounds at t_indx {}'.format(t_indx)
        except AssertionError as e:
            e.args += ('hprob[t_indx]={}'.format(hprob[t_indx]),
                       'day={}'.format(day),'hparams={}'.format(hparams),
                       'Dparams={}'.format(Dparams),'Dlparams={}'.format(Dlparams),
                       'mu_r={}'.format(mu_r),'n_periods={}'.format(n_periods),
                       'rad_dist={}'.format(rad_dist),'rad_res={}'.format(rad_res))
            raise
        pmf[row_min:row_max+1,col_min:col_max+1] += (hprob[t_indx]*
                cdf_mat)

    ###                                                             ###
    ###                     Find local spread                       ###
    ###                                                             ###
    '''pmf now has probabilities per cell of flying there.
       1-np.sum(pmf) is the probability of not flying.
       Add this probability to a distribution near the origin cell, if it isn't
       in roundoff territory.'''
       
    ### Check sum of pmf ###
    
    total_flight_prob = pmf.sum()
    try:
        assert pmf.min() >= -1e-8, 'pmf.min() less than zero, first block'
        assert total_flight_prob <= 1.00001, 'flight prob > 1, first block'
    except AssertionError as e:
        e.args += ('total_flight_prob = {}'.format(total_flight_prob),
            'pmf.min() = {}'.format(pmf.min()),
            'day={}'.format(day),'hparams={}'.format(hparams),
            'Dparams={}'.format(Dparams),'Dlparams={}'.format(Dlparams),
            'mu_r={}'.format(mu_r),'n_periods={}'.format(n_periods),
            'rad_dist={}'.format(rad_dist),'rad_res={}'.format(rad_res))
        raise
    if total_flight_prob < 0.99999:
    
        ### Shift by avg wind vel, where avg is weighted by f function ###
    
        # Get value of f function
        n = day_wind.shape[0] #number of wind data entries in the day
        fprob_vec = f_time_prob(n,*hparams[-4:]) #day long vector
        # Find weighted average of wind velocity
        wind_avg = np.average(day_wind[:,0:2],axis=0,weights=fprob_vec)
        # wind_avg is in km/hr. convert to m/(n_periods)
        # here, we are assuming that a parasitoid will on average
        #   be drifting an amount proportional to n_periods
        wind_avg *= 1000*24/(periods/n_periods) # m/(n_periods)
        # Scale this by a parameter.
        wind_avg *= mu_l_r
        
        ### Pass the remainder of the translation to get_mvn_cdf_values ###
        
        cdf_mu = wind_avg - np.round(wind_avg/cell_dist)*cell_dist
        cdf_mat = get_mvn_cdf_values(cell_dist,cdf_mu,Sl)
        
        ### translate wind_avg from (x,y) coords to nearest cell-center location. ###
        
        # [rad_res,rad_res] is the center cell of the domain.
        col_offset = int(np.round(wind_avg[0]/cell_dist))
        row_offset = int(np.round(-wind_avg[1]/cell_dist))
        row_cent = rad_res+row_offset
        col_cent = rad_res+col_offset
        adv_cent = np.array([row_cent,col_cent])
        # now we want to plop the normal distribution around this center
        norm_r = int(cdf_mat.shape[0]/2) #shape[0] is odd, floor half of it.
        # Get indices in pmf array
        row_min, col_min = adv_cent - norm_r
        row_max, col_max = adv_cent + norm_r
        # Return a unique error if the domain is not big enough so that it can
        # be handled in context.
        if row_max+1>pmf.shape[0] or col_max+1>pmf.shape[1] or \
            row_min < 0 or col_min < 0:
            raise BndsError
            
        ### Add local diffusion to model ###
        
        pmf[row_min:row_max+1,col_min:col_max+1] += \
            (1-total_flight_prob)*cdf_mat
            
        ###                                                              ###
        ### assure it sums to one by adding any error to the center cell ###
        ###                                                              ###
        
        total_flight_prob = pmf.sum()
        try:
            assert pmf.min() >= -1e-8, 'pmf.min() less than zero'
            assert total_flight_prob <= 1 + 1.00001, 'flight prob > 1'
        except AssertionError as e:
            e.args += ('total_flight_prob = {}'.format(total_flight_prob),
                'pmf.min() = {}'.format(pmf.min()),
                'cdf_mat.sum() = {}'.format(cdf_mat.sum()),
                'day={}'.format(day),'hparams={}'.format(hparams),
                'Dparams={}'.format(Dparams),'Dlparams={}'.format(Dlparams),
                'mu_r={}'.format(mu_r),'n_periods={}'.format(n_periods),
                'rad_dist={}'.format(rad_dist),'rad_res={}'.format(rad_res))
            raise
    
    ###                                                                    ###
    ### shrink the domain down as much as possible and return sparse array ###
    ###                                                                    ###
    
    # first, remove the really small data values from the array
    #   In some large pmf arrays, this can significantly reduce the size
    pmf_coo = r_small_vals(sparse.coo_matrix(pmf),prob_model=True)
    I = pmf_coo.row
    J = pmf_coo.col
    V = pmf_coo.data
    # now shrink domain and return a sparse array
    rad = int(max(np.fabs(I-rad_res).max(),np.fabs(J-rad_res).max()))
    I = I - rad_res + rad
    J = J - rad_res + rad
    return sparse.coo_matrix((V,(I,J)),shape=(rad*2+1,rad*2+1))
