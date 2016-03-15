# Analytic Parasitoids
### Parasitoid wasp drift-diffusion model with Bayesian inference
Please note: This code will only run in Python version 3.5 or greater.

Major work in progress!!!

- Task 1: Put together a working model from function defs
- Task 2: Interface with the global functions through PyMC to do Bayesian inference on the parameters

**Acknowledements**
Project support from SAMSI for a working group on Physical Ecology.

**Dependencies**
Numpy
Scipy
Matplotlib
Pillow (if plotting satellite imagery)
py.test (if running tests)
Reikna (if running on a GPU)

## Introduction

The probability model can be run simply by typing "python Run.py" into a 
terminal window. This model gives the spatial probability of a parasitoid's
location assuming it was released at midnight on the first day.

The population model can be run by typing "python RunPopModel.py" into a
terminal window. This model requires additional information about the number
and duration of the initial parasitoid release and gives the expected value
of parasitoid population in each spatial cell.

Both models accept flags and keyword arguments - please see the Params class
definition in Run.py for details.

If a file named "config.txt" is present, Run.py will read it for keyword arguments
of the form "parameter = value". Anything following # on a line will be ignored.

The plotting routine is capable of displaying satellite images from Bing maps
behind the model results. For this functionality to operate, the release
coordinates must be specified and a Bing maps key provided in config.txt. You
can obtain a Bing maps key for free at https://www.bingmapsportal.com/. The 
syntax for specifying a map key is "map_key = <key>". An internet connection is
required; no maps are cached between calls to the plotting routine.

All tests can be run by calling py.test from the terminal.

This model makes use of the Reikna and PyCUDA libraries to run on the GPU.
If these libraries are not installed, or the gloabl cuda flag has been set to
False, the model will run on the CPU only.