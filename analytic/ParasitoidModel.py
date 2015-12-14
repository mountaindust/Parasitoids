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
    """Returns probability of flying under given wind conditions
    
    Arguments:
        - windr -- wind speed
        - aw, bw -- logistic parameters (shape and bias)"""
    return 1.0 / (1. + np.exp(bw * (windr - aw)))

#Probability of flying at n discrete times of the day, equally spaced
def f_time_prob(n, a1, b1, a2, b2):
    """Returns probability of flying at n discrete times of day, equally spaced
    
    Arguments:
        - n -- number of wind data points per day available
        - a1,b1,a2,b2 -- logistic parameters (shape and bias)"""

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
    """Returns probability of flying per unit time under given conditions
    This is given by f times g times the constant lambda. Lambda can be thought
    of as the probability of flight under perfect conditions at an ideal time
    of day.
    
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
    return lam*f_times_g/np.sum(f_times_g) #np.array of length n
    
def mu(t_indx,day_wind,r):
    """Returns distance traveled through advection
    
    Arguments:
        - t_indx -- index in wind data
        - day_wind -- ndarray of wind directions
        - r -- constant"""
    return r*day_wind[t_indx,0:2]
    
def prob_density(day,wind_data,hparams,Dparams,mu_r,rad_dist,rad_res):
    """Returns prob density for a given day as an ndarray.
    This function always is calculated based on an initial condition at the
    origin. The final position of all wasps based on the previous day's
    position is then updated via convolution with this function.
    
    Arguments:
        - day -- day since release
        - wind_data -- dictionary of wind data
        - hparams -- parameters for h_flight_prob(...). (lam,aw,bw,a1,b1,a2,b2)
        - Dparams -- parameters for Dmat(...). (sig_x,sig_y,rho)
        - mu_r -- parameter r in the function mu
        - rad_dist -- distance from release point to side of the domain (m)
        - rad_res -- number of cells from center to side of the domain"""
        
    dom_len = rad_res*2+1 #number of cells along one dimension of domain
    cell_dist = rad_dist/rad_res #dist from one cell to neighbor cell.
        
    stdnormal = stats.multivariate_normal(np.array([0,0]),Dmat(*Dparams))
    ppdf = np.zeros((dom_len,dom_len))
    day_wind = wind_data[day]
    hprob = h_flight_prob(day_wind, *hparams)
    for t_indx in range(day_wind.shape[0]):
        #calculate integral in an intelligent way.
        #we know the distribution is centered around mu(t) at each t_indx
        mu_vec = mu(t_indx,day_wind,mu_r)
        #translate into cell location. [rad_res,rad_res] is the center
        adv_cent = np.round(mu_vec/cell_dist)+np.array([rad_res,rad_res])
        #now only worry about a normal distribution nearby this center
        for ii in range(-40,40):
            for jj in range(-40,40):
                cellx = adv_cent[0]+ii
                celly = adv_cent[1]+jj
                #check boundaries (probably not necessary)
                if 0<=cellx<dom_len and 0<=celly<dom_len:
                    ppdf[cellx,celly] += hprob[t_indx]\
                    *stdnormal.pdf(np.array([ii*cell_dist,jj*cell_dist]))
    #1-np.sum(ppdf) is now the probability of not flying.
    #Place this probability at the origin
    ppdf[rad_res,rad_res] += 1-ppdf.sum()
    return ppdf