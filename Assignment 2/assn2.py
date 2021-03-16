#!/usr/bin/env python
"""
AER1415 Computer Optimization - Assignment 2

Author: Atilla Saadat
Submitted: Mar 31, 2021
Email: asaadat@utias-sfl.net

"""

from numpy import *
import os
from matplotlib import pyplot as plt
from IPython import embed
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib
import cmath

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

matplotlib.use('TkAgg')

def l2(x):
	return linalg.norm(x)**2

def P1(x,**kwargs):
	x0 = x[:-1]
	x1 = x[1:]
	return sum(100.0*(x1 - x0**2.0)**2.0 + (1 - x0)**2.0)

# P1
def runPS1(xSize,params):
	print('PS1:\n')
	bounds = array([(-5,5)]*xSize)
	x0 = random.uniform(-5,5,xSize)
	return PSO(P1,x0,bounds,numParticles=max([200,xSize*10]),maxRepeat=10,maxIter=100,params=params).optimize(verbose=False)


def ps1_gk(x):
	g_x = 2*(200*x[0]**3 - 200*x[0]*x[1] + x[0] - 1)
	#https://www.wolframalpha.com/input/?i=d%2Fdy+100%28y%E2%88%92x%5E2%29%5E2%2B%281%E2%88%92x%29%5E2
	#d/dy -> 200*(x[1]-x[0]**2)
	g_y = 200*(x[1]-x[0]**2)
	return array([g_x, g_y])

def ps1_Hk(x):
	#https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/quadratic-approximations/a/the-hessian
	#https://www.wolframalpha.com/input/?i=d%2Fdx+2*%28200*x%5E3+-+200*x*y+%2B+x+-+1%29
	h_xx = 1200*x[0]**2 - 400*x[1] + 2
	#https://www.wolframalpha.com/input/?i=d%2Fdy+2*%28200*x%5E3+-+200*x*y+%2B+x+-+1%29
	h_xy = -400*x[0]
	#https://www.wolframalpha.com/input/?i=d%2Fdx+200*%28y-x%5E2%29
	h_yx = -400*x[0]
	#https://www.wolframalpha.com/input/?i=d%2Fdy+200*%28y-x%5E2%29
	h_yy = 200.
	return array([[h_xx,h_yx],[h_xy,h_yy]])


def normal_newton(fn):
	#https://www.wolframalpha.com/input/?i=d%2Fdx+100%28y%E2%88%92x%5E2%29%5E2%2B%281%E2%88%92x%29%5E2
	#d/dx -> 2*(200*x1**3 - 200*x1*x2 + x1 - 1)
	x0 = array([-1., -2.])
	e_r = 0.1
	e_g = 0.1
	e_a = 0.1
	x = [x0]
	vals = [fn(x0)]
	passConditions = array([])
	for iterNum in range(5):
		x_k = x[-1]
		g_k = ps1_gk(x_k)
		H_k = ps1_Hk(x_k)
		
		#slides method
		p_k = -dot(linalg.inv(H_k),g_k)
		x_k1 = x_k + p_k
		val_k1 = fn(x_k) + dot(g_k.T,p_k) + 0.5*dot(dot(p_k.T,H_k),p_k)

		x.append(x_k1)
		vals.append(val_k1)

	for idx,x_k in x:
		print('x*: {}, val:: {}',format(x_k,vals[idx]))

def quasi_newton(fn):
	#https://www.wolframalpha.com/input/?i=d%2Fdx+100%28y%E2%88%92x%5E2%29%5E2%2B%281%E2%88%92x%29%5E2
	#d/dx -> 2*(200*x1**3 - 200*x1*x2 + x1 - 1)
	x0 = array([-1., -2.])
	e_r = 0.1
	e_g = 0.1
	e_a = 0.1
	u_1 = 10e-4
	alpha_rho = 0.5
	x = [x0]
	vals = [fn(x0)]
	passConditions = []
	I = eye(len(x0))
	B_inv = [I[:]]
	for iterNum in range(15):
		x_k = x[-1]
		val_k = vals[-1]
		B_k_inv = B_inv[-1]
		g_k = ps1_gk(x_k)
		H_k = ps1_Hk(x_k)

		if l2(g_k) <= e_g:
			break

		p_k = -dot(B_k_inv,g_k)

		#Armijo sufficient decrease condition
		a_k = 0.5
		t1 = fn(x_k + a_k*p_k)
		t2 = fn(x_k) + u_1*a_k*dot(g_k.T,p_k)
		while not t1 <= t2:
			a_k = alpha_rho*a_k
			t1 = fn(x_k + a_k*p_k)
			t2 = fn(x_k) + u_1*a_k*dot(g_k.T,p_k)


		x_k1 = x_k + a_k*p_k

		val_k1 = fn(x_k1)

		passConditions.append((abs(val_k1 - val_k) <= (e_a + e_r*abs(val_k))))

		if len(passConditions) >= 2 and all(passConditions[-2:]):
			break

		#BFGS
		s_k = a_k*p_k
		y_k = ps1_gk(x_k1) - g_k

		B_k1_inv = (I - (dot(s_k,y_k.T)/dot(s_k.T,y_k))) * B_k_inv * (I - (dot(y_k,s_k.T)/dot(s_k.T,y_k))) + (dot(s_k,s_k.T)/dot(s_k.T,y_k))
		B_inv.append(B_k1_inv)
		print(x_k1,val_k1)
		x.append(x_k1)
		vals.append(val_k1)
	
	for idx,x_k in x:
		print('x*: {}, val:: {}',format(x_k,vals[idx]))

def modified_newton(f):
	#https://www.wolframalpha.com/input/?i=d%2Fdx+100%28y%E2%88%92x%5E2%29%5E2%2B%281%E2%88%92x%29%5E2
	#d/dx -> 2*(200*x1**3 - 200*x1*x2 + x1 - 1)
	x0 = array([-1., -2.])
	e_r = 0.1
	e_g = 0.1
	e_a = 0.1
	u_1 = 10e-4
	alpha_rho = 0.5
	x_k = x0
	val_k = f(x_k)
	passConditions = array([])
	for iterNum in range(100):
		g_x = 2*(200*x_k[0]**3 - 200*x_k[0]*x_k[1] + x_k[0] - 1)
		#https://www.wolframalpha.com/input/?i=d%2Fdy+100%28y%E2%88%92x%5E2%29%5E2%2B%281%E2%88%92x%29%5E2
		#d/dy -> 200*(x[1]-x[0]**2)
		g_y = 200*(x_k[1]-x_k[0]**2)
		gradientVector = array([g_x, g_y])

		#https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/quadratic-approximations/a/the-hessian
		#https://www.wolframalpha.com/input/?i=d%2Fdx+2*%28200*x%5E3+-+200*x*y+%2B+x+-+1%29
		h_xx = 1200*x_k[0]**2 - 400*x_k[1] + 2
		#https://www.wolframalpha.com/input/?i=d%2Fdy+2*%28200*x%5E3+-+200*x*y+%2B+x+-+1%29
		h_xy = -400*x_k[0]
		#https://www.wolframalpha.com/input/?i=d%2Fdx+200*%28y-x%5E2%29
		h_yx = -400*x_k[0]
		#https://www.wolframalpha.com/input/?i=d%2Fdy+200*%28y-x%5E2%29
		h_yy = 200.
		hessianMatrix = array([[h_xx,h_yx],[h_xy,h_yy]])

		if l2(gradientVector) <= e_g:
			break

		delta_k = -linalg.inv(hessianMatrix)*gradientVector
		if delta_k.T*gradientVector < 0:
			p_k = delta_k
		else:
			p_k = -delta_k

		#Armijo sufficient decrease condition
		alpha_k = 0.5
		t1 = f(x_k + alpha_k*p_k)
		t2 = f(x_k) + u_1*a_k*g_k.T*p_k
		while not t1 <= t2:
			alpha_k = alpha_rho*alpha_k
			t1 = f(x_k + alpha_k*p_k)
			t2 = f(x_k) + u_1*a_k*g_k.T*p_k

		x_k += alpha_k*p_k

		val_k1 = f(x_k)

		passConditions.append((val_k1 - val_k) <= (e_a + e_r*abs(f(x_k))))

		if len(passConditions) >= 2 and all(passConditions[-2:]):
			break

			
		embed()
	print('x*: {}'.format(x))

#normal_newton(P1)
quasi_newton(P1)