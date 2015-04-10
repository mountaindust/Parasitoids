# -*- coding: utf-8 -*-
"""
Drift-Diffusion Model with Bayesian inference

Created on Sat Mar 07 20:18:32 2015

@author: Christopher Strickland
"""

from __future__ import division
import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats
import pymc as pm

#we need to fix units for time. lets say t is in hours.

# Reading the observed emergence data from a text file.
def emergence_data(site_name):
    em = {}

    file_name = site_name + 'emergence.txt'
    em_file = open(file_name, 'r')

    comment_line = em_file.readline() # e.g. #date   0  22   25   ...
    split_comment_line = comment_line.split()
    split_comment_line.pop(0) # remove '#data'
    for line in em_file.readlines(): # e.g. 01  2  0    0    ...
        split_line = line.split()
        date = int(split_line.pop(0))
        for ind in range(0,len(split_line)): # skips release field
            field = split_comment_line[ind]
            if not em.has_key(field):
                em[field] = {}
            em[field][date] = int(split_line[ind]) # em[field_name,date] = value
    em_file.close()

    return em

# Read the wind information from a text file.
def read_wind_file(site_name):
    file_name = site_name + 'wind.txt'
    wind_file = open(file_name)
    wind_data = {}
    for line in wind_file.readlines():
        # File has data like this: day x-component y-component
        splitline = line.split()
        day = int(splitline[0])
        # x-component
        windx = float(splitline[1])
        if abs(windx) < 10e-5: # Remove very small values
            windx = 0
        # y-component
        windy = float(splitline[2])
        if abs(windy) < 10e-5: # Remove very small values
            windy = 0
        # r
        windr = np.sqrt(windx**2+windy**2)
        if abs(windr) < 10e-5: # Remove very small values
            windr = 0
        # theta
        if (windx == 0) & (windy == 0):
            theta = None
        elif (windx == 0) & (windy > 0):
            theta = np.pi/2
        elif (windx == 0) & (windy < 0):
            theta = -np.pi/2
        else:
            theta = np.arctan(windy/windx)
        if windx < 0:
            theta = theta+np.pi
        # Add to our wind_data dictionary
        if wind_data.has_key(day):
            wind_data[day].append(np.array([windx,windy,windr,theta]))
        else:
            wind_data[day] = [np.array([windx,windy,windr,theta])]
    wind_file.close()

    return wind_data
    #this returns a dictionary of days, with each day pointing to
    #an ordered list of numpy arrays. Each member of the list is a time.

##########    Model functions    ##########

#Probability of flying under given wind conditions
def g(windr, aw, bw):
    return 1.0 / (1. + np.exp(bw * (windr + aw)))

#Probability of flying at n discrete times of the day, equally spaced
def f(n, a1, b1, a2, b2):
    #t is in hours, and denotes start time of flight.
    #(this is sort of weird, because it looks like wind was recorded starting
    #after the first 30 min)
    t_tild = np.arange(0,24-24./n,n)
    return 1.0 / (1. + np.exp(b1 * (t_tild + a1))) - \
    1.0 / (1. + np.exp(b2 * (t_tild + a2)))    

#Covarience matrix for diffusion
def D(sig_x, sig_y, rho):
    return np.array([[sig_x^2, rho*sig_x*sig_y],\
                     [rho*sig_x*sig_y, sig_y^2]])
    
#Probability of flying under given conditions
def h(day_wind, Pf, aw, bw, a1, b1, a2, b2):
    #day_wind is a list of np.array wind directions
    #day_wind[0] = np.array([windx,windy,windr,theta])
    n = len(day_wind)
    #get just the windr values
    windary = np.array(day_wind)
    windr = windary[:,2]
    np.sum(f(n,a1,b1,a2,b2)*g(windr,aw,bw))