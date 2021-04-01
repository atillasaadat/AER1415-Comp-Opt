#!/usr/bin/env python
"""
AER1415 Computer Optimization - Assignment 2

Author: Atilla Saadat
Submitted: Mar 31, 2021
Email: asaadat@utias-sfl.net

rosenbrock test function - supporting module

"""
from numpy import *

#rosenbrock function
def rosen(x,**kwargs):
	x = asarray_chkfinite(x)
	x0 = x[:-1]
	x1 = x[1:]
	return sum(100.0*(x1 - x0**2.0)**2.0 + (1 - x0)**2.0)

#rosenbrock gradient vector for n=2 and n=5
def rosen_gk(x):
	if len(x) == 2:
		#g = [400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2.,
		#	-200*x[0]**2 + 400*x[1]**3 + 202*x[1] - 2.]
		#g = [-400*(x[1]-x[0]**2)*x[0] - 2*(2.-x[0]),
		#	200*(x[1]-x[0]**2)]
		g = [2*x[0] - 400*x[0]*(-x[0]**2 + x[1]) - 2, 200*(x[1] -x[0]**2) ]
	elif len(x) == 5:
		'''
		g = [400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2.,
			-200*x[0]**2 + 400*x[1]**3 + x[1]*(202 - 400*x[2]) - 2.,
			-200*x[1]**2 + 400*x[2]**3 + x[2]*(202 - 400*x[3]) - 2.,
			-200*x[2]**2 + 400*x[3]**3 + x[3]*(202 - 400*x[4]) - 2.,
			-200*x[3]**2 + 400*x[4]**3 + 202*x[4] - 2.]
		'''
		g = [2*(x[0]-1) - 400*x[0]*(x[1] - x[0]**2),
			 2*(x[1]-1) - 400*x[1]*(x[2] - x[1]**2) + 200*(x[1] - x[0]**2),
			 2*(x[2]-1) - 400*x[2]*(x[3] - x[2]**2) + 200*(x[2] - x[1]**2),
			 2*(x[3]-1) - 400*x[3]*(x[4] - x[3]**2) + 200*(x[3] - x[2]**2),
			 200*(x[4] - x[3]**2),
		]
	return array(g)

#rosebrock hessiam matrix for n=2
def rosen_Hk_n2(x):
	#https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/quadratic-approximations/a/the-hessian
	#https://www.wolframalpha.com/input/?i=d%2Fdx+2*%28200*x%5E3+-+200*x*y+%2B+x+-+1%29
	#https://www.wolframalpha.com/input/?i=d%2Fdy+2*%28200*x%5E3+-+200*x*y+%2B+x+-+1%29
	#https://www.wolframalpha.com/input/?i=d%2Fdx+200*%28y-x%5E2%29
	#https://www.wolframalpha.com/input/?i=d%2Fdy+200*%28y-x%5E2%29
	h_xx = 1200*x[0]**2 - 400*x[1] + 2.
	h_xy = -400*x[0]
	h_yx = -400*x[0]
	h_yy = 200.
	return array([[h_xx,h_yx],[h_xy,h_yy]])