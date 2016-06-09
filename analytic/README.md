# Analytic Parasitoids
### Parasitoid wasp drift-diffusion model with Bayesian data modeling framework
Please note: This code will only run in Python version 3.5 or greater.

**Acknowledements**:
This material was based upon work partially supported by the National Science 
Foundation under Grant DMS-1127914 to the Statistical and Applied Mathematical 
Sciences Institute. Any opinions, findings, and conclusions or recommendations 
expressed in this material are those of the author(s) and do not necessarily 
reflect the views of the National Science Foundation.

**Dependencies**  
Numpy  
Scipy  
Matplotlib  
Pillow (if plotting satellite imagery)  
py.test (if running tests)  
Reikna (if running on a GPU. Requres CUDA libraries and PyCUDA.)  
PyMC (if running Bayesian model fitting)
PyTables (if running Bayesian model fitting - used by hdf5 PyMC backend)
Pandas (if running Bayesian model fitting)

## Introduction

This code can consists of two parts: the first is the drift-diffusion model 
(which itself can be run either as a probabilistic model or as a population 
model), while the second is a Bayesian modeling framework for fitting the 
drift-diffusion model parameters to data. Which file you want to run depends on 
the functionality you would like to access.

### Drift-diffusion model

The drift-diffusion probability model can be run simply by typing "python Run.py" 
into a terminal window while within this directory. Given appropriate wind data, 
this model simulates the spatial probability of a single parasitoid's location 
after each day assuming that it was released from a central point at midnight.

The population model can be run by typing "python Run.py --pop" into a
terminal window. This model requires additional information about the number
of parasitoids released along with the duration and timing of the release. It 
simulates the expected value of the parasitoid population after each day inside 
within each spatial cell of the domain.

Both models accept flags and keyword arguments which will be described below. 
If a file named "config.txt" is present, Run.py will read it for keyword arguments
as well. These should be of the form "parameter = value", and anything following
a # on a line will be ignored as a comment. Any options passed via the terminal
will override the defaults specified in config.txt.

Saved model results can be plotted by typing "python Plot_Results.py <filename>" 
where <filename> is the path to the saved simulation data, with or without the 
file extension. This program features an interactive menu with options including 
plotting single simulation days, plotting all simulation days in succession, 
plotting in black and white, saving model visualizations and a user specified 
size and dpi, and saving all simulation days to an mp4 video file.

The plotting routine is capable of displaying satellite images from Bing or 
Google maps as a backdrop to the model results. For this functionality to 
operate, the release coordinates must be specified and a Bing or Google maps key 
must be provided in config.txt. You can obtain a Bing maps key for free at 
<https://www.bingmapsportal.com/> and a Google maps key for free at 
<https://developers.google.com/maps/documentation/static-maps/>. The syntax for 
specifying a map key in config.txt is "map_key = <key>". An internet connection 
is required; no maps are cached between calls to the plotting routine. You must
specify which service you are using via the maps_service variable - the default
is currently Google.

All tests can be run by calling py.test from the terminal.

The model makes use of the Reikna and PyCUDA libraries to run on the GPU.
If these libraries are not installed, or if the gloabl cuda flag has been set to
False, the model will run on the CPU only.

The model also makes use of Python's multiprocessing library to simulate each
day in parallel. Performance will depend on the specifics of your machine, and 
there is a variable one can set specifying a lower limit on the number of 
simulated flight days, below which parallelism will not be used.

### Bayesian model and parameter fitting

Due to the inevitable varience between datasets - including format, data collection 
techniques, etc. - it is impractical to expect a specific dataset structure from 
which code can automatically parse the necessary information. Instead, we leave 
it to the user to supply Pandas code that will properly parse the data and return 
expected data structures. This Pandas code should be pasted directly into 
Data_Import.py, which has been heavily commented to provide guidence on exactly 
what is needed where. Assuming everything is set up properly, Data_Import.py 
implements the LocInfo class which returns and object containing field data in a 
format that can be compared to model results through the Bayesian framework 
specified in Bayes_Run.py, which specifieds priors, runs the simulations, and 
does the MCMC sampling, and Bayes_funcs.py, which models data collection and 
parasitoid emergence based on the simulated population densities.

Once everything is properly specified, MCMC samples can be taken by typing 
"python Bayes_Run.py" into a terminal. These samples will be saved to an hdf5 
database allowing for later inspection or additional sampling.

## Parasitoid Drift-Diffusion Model

All model parameters for both the probability model and the population model 
are specified in the Params class whose implemenation is located in Run.py, 
except for the Bing/Google maps key which is always only specified in config.txt 
(this is for re-distribution purposes). The Params class implemenation can be
edited directly to specify different defaults, and machine specific defaults
can also be specified in config.txt. All parameters and flags can also be
specified on a per-run basis by using keyword options or flags at the command 
line. The following is a list of these options.

### Run.py command line options/parameters

Current defaults are specified with brackets where appropriate.  

***General flags and simulation options***
- --pop, --popmodel, --pop_model, or prob_model=False:  
  Run the drift-diffusion population model (rather than the probability model).
- --prob, --probmodel, --prob_model, or prob_model=True:  
  Run the drift-diffusion probability model. This is currently the default 
  behavior.
- --no_output, [--output], output=False/[True]:  
  Turn off/on saving of model results. If on, model results will be saved in 
  a [NumPy .npz ZipFile](http://docs.scipy.org/doc/numpy/neps/npy-format.html) 
  and the model parameters will be saved in a json file of the same name. 
  Saved simulation results can be visualized by typing 
  "python Plot_Results.py <filename>" where <filename> is the path to the saved 
  simulation file, with or without the file extension.
- --no_plot, [--plot], plot=False/[True]:
  Turn off/on plotting of results after the simulation is finished.
- --no_cuda, [--cuda], cuda=False/[True]:
  Whether or not to use CUDA libraries. If CUDA/PyCUDA is not installed, the 
  simulation will be run without CUDA automatically, regardless of this flag.
- --carnarvon, [--kalbar]:
  Load location specific parameters for a known location. These include:
  - dataset (name of location)
  - site_name (path to data ...clearly these were named backwards. oh well.)
  - start_time (time at which wind data started recording)
  - coord (release coordinates)
  - r_dur (release duration)
  - r_dist (release time distribution)
  - r_start (time of day release started)
  - r_number (number of wasps released)
  See below for information on each of these.
- ndays=<positive integer>:
  Number of days to run simulation. This should not exceed the number of days 
  you have wind data for. If you would like to run the simulation for all the 
  days for which you have data, you can enter a value of "-1".
- domain_info=(distance in meters, cell count):
  This defines the domain for the simulation. Specify the meters from the 
  release point to an edge of the domain, and the number of cells from release 
  to the edge. This will result in a domain with 2*(cells) + 1 number of cells 
  along each edge, with a cell area of (meters/cells)**2.
- outfile=<filename>:
  Name to use when saving simulation data (include path, if any). Default 
  behavior is to save in a subdirectory named "output" using the name specified 
  by the "dataset" parameter with the time information appended along with "pop" 
  if the population simulation was run.
- min_ndays=<integer>:
  Minimum number of days requried in a simulation for multiprocessing to kick in.
  Depending on your machine, there may be a threshold where very short 
  simulations run slower on multiple processors because of overhead.

***Location specific variables***  
- dataset=<name>: 
  name for this data
- site_name=<file path>: 
  path and name for all the necessary data to run the simulation. This usually 
  looks something like 'data/<yournamehere>'. The simulation will then try to 
  load a file named 'data/<yournamehere>wind.txt' to get the wind data. If 
  running Bayes, you will also need 'data/<yournamehere>releasegrid.txt' and
  'data/<yournamehere>fields.txt'.  
- coord=(latitude,longitude):
  Lat/long coordinates of the release point. Needed for satellite images.
- r_dur=<positive integer>:
  Number of days the release was conducted. Needed for population model.
- start_time=00:00/00:30:
  Time the wind data started. Either midnight (00:00) or 30 min after. Needed 
  for population model.
- r_start=<number between 0 and 1>:
  Time of day the release was started. Units are days. Needed for population 
  model.
- r_number=<positive integer>:
  Number of wasps released. Needed for population model.
  
***Model parameters***  
- g_params=(center,scale):
  Parameters for wind logistic function g. First one centers the logistic, the
  second on scales it.
- f_params=(center,scale,center,scale):
  Parameters for take-off probability mass function f, which is based on time of
  day. First two parameters are the morning logistic (center, scale), the second
  two are for the evening logistic (center, scale).
- Dparams=(sig_x,sig_y,correlation):
  Covarience parameters for diffusion in wind. Provide the standard deviation in
  the x-direction, y-direction, and the correlation.
- Dlparams=(sig_x,sig_y,correlation):
  Covarience parameters for local diffusion. Provide the standard deviation in
  the x-direction, y-direction, and the correlation.
- lam=lambda:
  lambda parameter describing the probability of wind-based flight during the 
  day assuming ideal conditions
- mu_r=mu_r:
  mu parameter scaling flight distance to average wind speed
- n_periods=<integer>:
  number of wind vector points per flight. This number times the time frequency 
  of the interpolated wind data gives the total flight time (see interp_num).

***Other***  
- interp_num=<positive integer>:
  Number of interpolation points to make between wind data collection points as 
  loaded from the data file. The resulting frequency of wind vector points is 
  crucial in determining how long each wasp will fly (see n_periods).
- maps_key=key:
  Bing or Google static maps key. Probably best to set this in config.txt.
- maps_service='Bing' or 'Google'
  Set in config.txt to match where you got your key.
  
### Plot_Result.py

Running Plot_Result.py in a terminal (python Plot_Result.py) will result in an
interactive menu allowing you to load saved simulations and visualize them in 
a variety of ways. Journal quality images may be saved using this menu (color or
black/white), and video output is also an option.

### Plot_SampleLocations.py

When run from the terminal, this module plots all grid sample points (using 
different colors depending on number of leaves sampled) and both the outline and
filled cells of each sentinel field. This is a great way to make sure that these
locations have been imported correctly.

### Plot_ParasitoidModel.py

This module is meant to be loaded into an IPython session and used interactively.
Its functions plot the various pieces of the analytic model so that one can
easily inspect what is happening "under the hood" for different parameter 
choices.
  
## Bayesian model

The exact form of the Bayesian model will depend on the specifics of your data
collection. LocInfo in Data_Import.py handles the loading and the parsing of the
data. Bayes_Run.py specifies priors, runs the model, and calls functions to 
connect the model results to data. It is also the file to run if you want to
do MCMC sampling or inspect the results. Bayes_MAP.py is similar to Bayes_Run.py
(in fact, most of it is just a direct copy - this due to limitations in the 
Python multiprocessing library), but runs pymc to find the maximum a posteriori
estimate instead. Bayes_funcs.py does the job of projecting parasitoid emergence 
from population model results, as well as gathering the expected number of wasps 
at grid points and cardinal direction sample points. Bayes_Plot.py contains 
functions for plotting traces and posterior distributions from mcmc samples.

All functionality should be provided via interactive menus in Bayes_Run.py and
Bayes_MAP.py. MCMC samples are saved in hdf5 databases which allow for later
inspection and resuming of sampling. It may be useful to import pymc and 
Bayes_Plot.py into an IPython session to load the database and inspect it 
directly (a menu item in Bayes_Run.py does this as well). Please see 
implementation for further details.

At this time, the Bayesian modules are only tested for the Kalbar dataset, and
thus Kalbar is hard coded into the modules Bayes_Run.py and Bayes_MAP.py.

While these modules have been tested and run as advertised, they should be 
considered under construction while results are examined, priors possibly 
refined, and additional functionality is added.