#! /usr/bin/env python3

import pstats, cProfile
from line_profiler import LineProfiler

from Run import Params
from ParasitoidModel import prob_mass, get_wind_data, get_mvn_cdf_values

params = Params()
wind_data, days = get_wind_data(*params.get_wind_params())
day = 8

cProfile.runctx("prob_mass(day,wind_data,*params.get_model_params())", 
    globals(), locals(), "Profile.prof")
    
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()


### This bit spits out a line by line profile of the functions passed to 
  # the LineProfiler class.
  
# profiler = LineProfiler(get_mvn_cdf_values)
# profiler.runctx("prob_mass(day,wind_data,*params.get_model_params())",
    # globals(),locals())
# profiler.print_stats()