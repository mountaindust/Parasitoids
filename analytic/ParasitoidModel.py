#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drift-Diffusion Model
This module should implement the pieces of the model, including info about the
    spatial mesh. These functions will then be called from an external module,
    either for running the model or Bayesian inference.

Created on Sat Mar 07 20:18:32 2015

:author: Christopher Strickland
:email: cstrickland@samsi.info
"""

import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats
from scipy.stats import mvn
from scipy import fftpack

__author__ = "Christopher Strickland"
__email__ = "cstrickland@samsi.info"
__status__ = "Development"
__version__ = "0.1"
__copyright__ = "Copyright 2015, Christopher Strickland"

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

# I'm guessing the units here are m/s
def read_wind_file(site_name):
    """ Reads the wind data from a text file
    
    Arguments:
        - site_name -- string
        
    Returns:
        - wind_data -- wind data as a dictionary of 2D ndarrays, 
                        keys are date since release"""
    file_name = site_name + 'wind.txt'
    with open(file_name) as wind_file:
        wind_data = {}
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
            # Angle of wind vector, theta
            if (windx == 0) and (windy == 0):
                theta = 0
            elif (windx == 0) and (windy > 0):
                theta = np.pi/2
            elif (windx == 0) and (windy < 0):
                theta = -np.pi/2
            else:
                theta = np.arctan(windy/windx)
            if windx < 0:
                theta = theta+np.pi
            # Add to our wind_data dictionary.
            # Each date is an ordered list of wind data (ndarray) by hour.
            if day in wind_data:
                wind_data[day].append(np.array([windx,windy,windr,theta]))
            else:
                wind_data[day] = [np.array([windx,windy,windr,theta])]
    
    #convert each list of ndarrays to a single ndarray where rows are times,
    #  columns are the windx,windy,windr,theta. This allows fancy slicing.
    for day in wind_data:
        wind_data[day] = np.array(wind_data[day]) #lists of ndarrays become 2D

    return wind_data
    #this returns a dictionary of days, with each day pointing to
    #a 2-D ndarray. Rows are times, columns are the windx,windy,windr,theta

##########    Model functions    ##########

def g_wind_prob(windr, aw, bw):
    """Returns probability of flying under given wind conditions.
    If the wind has no effect on flight probability, returns 1.
    Otherwise, scales down from 1 to 0 as wind speed increases.
    
    Arguments:
        - windr -- wind speed
        - aw -- wind speed at which flight prob. is scaled by 0.5
        - bw -- steepness at which scaling changes (larger numbers = steeper)"""
    return 1.0 / (1. + np.exp(bw * (windr - aw)))

#Probability of flying at n discrete times of the day, equally spaced
def f_time_prob(n, a1, b1, a2, b2):
    """Returns probability of flying based on time at n equally spaced times
    If the time of day as no effect on flight probability, returns 1.
    Otherwise, scales down from 1 to 0 as time is more unfavorable.
    
    Arguments:
        - n -- number of wind data points per day available
        - a1 -- time of morning at which to return 0.5
        - b1 -- steepness of morning curve (larger numbers = steeper)
        - a2 -- time of afternoon at which to return 0.5
        - b2 -- steepness of afternoon curve (larger numbers = steeper)"""

    #t is in hours, and denotes start time of flight.
    #(this is sort of weird, because it looks like wind was recorded starting
    #after the first 30 min)
    #Maybe we should shift this all 30 min...?
    t_tild = np.linspace(0,24-24./n,n) #divide 24 hours into n equally spaced times
    return 1.0 / (1. + np.exp(-b1 * (t_tild - a1))) - \
    1.0 / (1. + np.exp(-b2 * (t_tild - a2)))    

def Dmat(sig_x, sig_y, rho):
    """Returns covarience matrix for diffusion process
    
    Arguments:
        - sig_x, sig_y -- Std. deviation in x and y direction respectively
        - rho -- Covariance"""
        
    return np.array([[sig_x**2, rho*sig_x*sig_y],\
                     [rho*sig_x*sig_y, sig_y**2]])
    
def h_flight_prob(day_wind, lam, aw, bw, a1, b1, a2, b2):
    """Returns probability density of flying during a given day's wind.
    This is given by f times g times the constant lambda. Lambda can be thought
    of as the probability of flight under perfect conditions at an ideal time
    of day
    
    Arguments:
        - day_wind -- ndarray of wind directions
        - lam -- constant
        - aw,bw -- g function constants
        - a1,b1,a2,b2 -- f function constants
    
    Note: day_wind[0,:] = np.array([windx,windy,windr,theta])"""

    n = day_wind.shape[0] #number of wind data entries in the day
    #get just the windr values
    windr = day_wind[:,2]
    f_times_g = f_time_prob(n,a1,b1,a2,b2)*g_wind_prob(windr,aw,bw)
    # normalize f_times_g by the integral with respect to time.
    #   dt in hours can be had by dividing 24 hrs/day by samples/day
    return lam*f_times_g/(np.sum(f_times_g)*24/n) #np.array of length n


def get_mvn_cdf_values(centers,cell_length,S):
    """Get cdf values for a multivariate normal centered at (0,0) 
    inside regular cells. To do this fast, we use a secret Fortran mulivariate
    normal CDF (mvn) due to Dr. Alan Genz
    
    This function cannot be jit nopython compiled with numba
        - doesn't recognize mvn
    
    Args:
        centers: 2D ndarray with each row an x,y coordinate for the center of
                    the cell
        cell_length: length of each cell
        S: covariance matrix
        
    Returns:
        cdf_vals: array of cdf values, one for each row in centers
        inform: should be all zeros if normal completion with ERROR < EPS"""
    
    cdf_vals = np.empty(centers.shape[0])
    inform = np.empty(centers.shape[0])
    for ii in range(centers.shape[0]):
        low = centers[ii] - cell_length/2.
        upp = centers[ii] + cell_length/2.
        cdf_vals[ii],inform = mvn.mvnun(low,upp,np.array([0,0]),S)
        
    return cdf_vals,inform

    
def prob_mass(day,wind_data,hparams,Dparams,mu_r,rad_dist,rad_res):
    """Returns prob mass function for a given day as an ndarray.
    This function always is calculated based on an initial condition at the
    origin. The final position of all wasps based on the previous day's
    position can then be updated via convolution with this function.
    
    INT_RANGE is currently an arbitrary value representing how far we want to
    integrate around the origin of the mean-shifted normal distribution.
    TODO: In reality, this value should be based on Dmat.
    
    Arguments:
        - day -- day since release
        - wind_data -- dictionary of wind data
        - hparams -- parameters for h_flight_prob(...). (lam,aw,bw,a1,b1,a2,b2)
        - Dparams -- parameters for Dmat(...). (sig_x,sig_y,rho)
        - mu_r -- parameter to scale flight duration and distance vs. windspeed
        - rad_dist -- distance from release point to side of the domain (m)
        - rad_res -- number of cells from center to side of the domain
        
    Returns:
        - pmf -- 2D spatial probability mass function of finding the parasitoid
                    in each spatial cell according to given resolution"""
        
    dom_len = rad_res*2+1 #number of cells along one dimension of domain
    cell_dist = rad_dist/rad_res #dist from one cell to neighbor cell.
    
    INT_RANGE = 40
        
    stdnormal = stats.multivariate_normal(np.array([0,0]),Dmat(*Dparams))
    pmf = np.zeros((dom_len,dom_len))
    day_wind = wind_data[day]
    hprob = h_flight_prob(day_wind, *hparams)
    for t_indx in range(day_wind.shape[0]):
        # Get the advection velocity and put in units = m/(unit time)
        mu_v = day_wind[t_indx,0:2] # km/hr
        mu_v *= 1000*24/wind_data[day].shape[0] # m/(unit time)
        # We also need to scale this by a constant which represents the fraction
        #   of the unit time that the wasp spends flying times a scaling term
        #   that takes wind speed to advection speed.
        mu_v *= mu_r
        
        #calculate integral in an intelligent way.
        #we know the distribution is centered around mu(t) at each t_indx
        #translate into cell location. [rad_res,rad_res] is the center
        adv_cent = np.round(mu_v/cell_dist)+np.array([rad_res,rad_res])
        #now only worry about a normal distribution nearby this center
        
        #We will get the normal distribution +/- INT_RANGE from the center
        cellx_min = int(max(0,adv_cent[0]-INT_RANGE))
        cellx_max = int(min(dom_len,adv_cent[0]+INT_RANGE))
        celly_min = int(max(0,adv_cent[1]-INT_RANGE))
        celly_max = int(min(dom_len,adv_cent[1]+INT_RANGE))
        
        #want to pass list of [x,y] values to get_mvn_cdf_vals, 
        #   centered at origin
        xlist = np.arange(cellx_min,cellx_max+1)-adv_cent[0]
        ylist = np.arange(celly_min,celly_max+1)-adv_cent[1]
        
        #need list of all coordinates. This fancyness accomplishes it
        xylist = np.vstack(np.meshgrid(xlist,ylist)).reshape(2,-1).T
        
        #Get multivariate normal cdf at each cell under consideration
        cdf_vals,inform = \
        get_mvn_cdf_values(xylist*cell_dist,cell_dist,Dmat(*Dparams))
        #Check that the integration converged to less than EPS in each cell
        assert np.all(inform == 0)
        
        #approximate integral over time
        pmf[cellx_min:cellx_max+1,celly_min:celly_max+1] += (hprob[t_indx]*
            cdf_vals.reshape(xlist.size,ylist.size)*24/wind_data[day].shape[0])

    # pmf now has probabilities per cell of flying there.
    # 1-np.sum(ppdf) is the probability of not flying.
    # Add this probability to the origin cell.
    assert pmf.sum() <= 1
    pmf[rad_res,rad_res] += 1-pmf.sum()
    return pmf