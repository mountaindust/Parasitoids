# Analytic Parasitoids
Parasitoid wasp drift-diffusion model with Bayesian inference
===

Major work in progress!!!

- Task 1: Put together a working model from function defs
- Task 2: Interface with the global functions through PyMC to do Bayesian inference on the parameters

**Acknowledements**
Project support from SAMSI for a working group on Physical Ecology.

## Introduction

The model can be run simply by typing "python Run.py" into a terminal window.
Run.py accepts flags and keyword arguments - please see the Params class
definition in Run.py for details.

If a file named "config.txt" is present, Run.py will read it for keyword arguments
of the form "parameter = value". Anything following # on a line will be ignored.

All tests can be run by calling py.test from the terminal.

This model makes use of the Reikna and PyCUDA libraries to run on the GPU.
If these libraries are not installed, or the gloabl cuda flag has been set to
False, the model will run on the CPU only.