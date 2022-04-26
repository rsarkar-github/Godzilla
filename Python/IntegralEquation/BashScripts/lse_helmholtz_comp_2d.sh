#!/usr/bin/bash

bash
conda activate py38

python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run1 0 100 30

python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run2 0 10 30

python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run3 0 50 30

python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run4 1 50 30

python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run5 2 50 30

python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run6 11 50 30

python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run7 -1 50 30

python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run8 -1 50 10

python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run9 -1 50 5

python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run10 1 50 10

python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run11 1 50 15

python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run12 1 50 20

python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run13 11 50 10
