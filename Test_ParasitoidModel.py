#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for ParasitoidModel

Created on Fri May 08 12:12:19 2015

:author: Christopher Strickland
"""

from __future__ import division
import numpy as np
import ParasitoidModel as PM
import matplotlib.pyplot as plt

#load some emergence data
c_em = PM.emergence_data('carnarvonearl')
#load some wind data
wind_data = PM.read_wind_file('carnarvonearl')

#Here, we really want to solve two problems at the same time
# 1. test the functions in ParasitoidModel
# 2. play with them to get some reasonable dummy parameters

#### Test g function for prob. during different wind speeds ####
def test_g(aw=1.8,bw=6):
    windr_range = np.arange(0,3.1,0.1) #a range of wind speeds
    plt.figure()
    #first scalar centers the logistic. Second one stretches it.
    plt.plot(windr_range,PM.g(windr_range,aw,bw))
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
    plt.plot(day_time,PM.f(n,a1,b1,a2,b2))
    plt.xlabel('time of day (hrs)')
    plt.ylabel('probability density of flight')
    plt.title('f func for prob of flight during time of day')
    plt.show()
    
#### Test h function (and therefore g and f) with data ####
def test_h(day_wind=wind_data[1],lam=1.1):
    day_time = np.linspace(0,24,wind_data[1].shape[0])
    plt.figure()
    plt.plot(day_time,PM.h(day_wind,lam,1.8,6,7,1.5,19,1.5))
    plt.xlabel('time of day (hrs)')
    plt.ylabel('probability density of flight')
    plt.title('h func for prob of flight given wind')
    plt.show()