#! /usr/bin/env python3

import pstats, cProfile
from line_profiler import LineProfiler

from Run import Params, main
from ParasitoidModel import prob_mass, get_wind_data, get_mvn_cdf_values

params = Params()
wind_data, days = get_wind_data(*params.get_wind_params())
day = days[0]
params.min_ndays = 40
params.ndays = 3
params.OUTPUT = False
params.PLOT = False
# day = 8

cProfile.runctx("prob_mass(day,wind_data,*params.get_model_params())", 
   globals(), locals(), "Profile.prof")
    
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()


### This bit spits out a line by line profile of the functions passed to 
  # the LineProfiler class.
  
# profiler = LineProfiler(main(params))
# profiler.runctx("main(params)",
    # globals(),locals())
# profiler.print_stats()
