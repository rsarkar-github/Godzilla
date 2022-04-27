#!/usr/bin/bash
if [ $1 -eq 0 ]
then
    python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run1 0 10
    python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run2 0 15
    python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run3 0 20
    python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run4 1 10
    python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run5 1 15
    python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run6 1 20
    python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run7 2 10
    python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run8 2 15
    python -m Python.IntegralEquation.Scripts.lse_helmholtz_comp_2d run9 2 20
fi
if [ $1 -eq 1 ]
then
    str1="Python/IntegralEquation/Runs/test_linvel2d_integral_kernel_invertibility/run"
    str2="/Data/green_func.npz"
    for i in {1..9}
    do
       str3=$str1${i}$str2
       rm -rf $str3
    done
fi