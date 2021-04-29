#!/usr/bin/env python
""" 
AER1415 Computer Optimization - Assignment 3

Author: Atilla Saadat
Submitted: April 28, 2021
Email: asaadat@utias-sfl.net

Descripton: Question 5

"""

from numpy import *
import os
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from PSO import *
from PSOHybrid import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

#define P4 objective function
def P4(x,**kwargs):
	try:
		t1 = sum(cos(x)**4)
	except TypeError:
		print ('TypeError')
		#embed()
	t2 = prod(cos(x)**2)
	t3 = sum((arange(len(x))+1)*x**2)
	gx1 = 0.75 - prod(x)
	gx2 = sum(x) - (7.5*len(x))
	fx = divide(-abs(t1-2*t2),sqrt(t3))
	if isnan(fx):
		fx = 0
	psi = max([0,gx1])**2 + max([0,gx2])**2
	if 'onlyFx' in kwargs.keys() and kwargs['onlyFx']:
		return fx
	return fx + kwargs['penaltyParam']*psi

#initialize P4 variables and algorithm type handler
def runPS4(xSize, params, method):
	print('PS4 - {}:\n'.format(method))
	bounds = array([(0,10)]*xSize)
	x0 = random.uniform(0.5,5.5,xSize)
	if method == 'PSO':
		return PSO(P4,x0,bounds,numParticles=max([200,xSize*10]),maxRepeat=10,maxIter=100,params=params).optimize(verbose=False)
	elif method == 'PSO Hybrid':
		return PSOHybrid(P4,x0,bounds,numParticles=max([200,xSize*10]),maxRepeat=10,maxIter=100,params=params).optimize(verbose=False)

def plotPS4_iter(dim,paramsCombo):
	fig, ax = plt.subplots()
	#plot hybrid and normal PSO algorithm
	for method in ['PSO Hybrid','PSO']:
		ps4Results = runPS4(dim,paramsCombo,method)
		penalty = 'rho: {}'.format(paramsCombo['penalty']) if  paramsCombo.get('penalty',False) else 'rho_static: {}'.format(paramsCombo['penaltyStatic'])
		bound = 'boundMethod: {}'.format(paramsCombo['boundMethod'])
		ax.plot(ps4Results['meanRepeatValues'],label='Method: {}, Mean: {:.8f}'.format(method,ps4Results['minMeanVal']))
		ax.fill_between(range(len(ps4Results['meanRepeatValues'])), ps4Results['meanRepeatValues']-ps4Results['stdRepeatValues'],  ps4Results['meanRepeatValues']+ps4Results['stdRepeatValues'], alpha=0.2)
	ax.legend()
	ax.set_title('PS4 Bump Test n={} - Objective Func. Val (w/ Quad. Penalty) vs. Iteration Number'.format(dim))
	ax.set_xlabel('Iteration #')
	ax.set_ylabel(r'$\pi(x,\rho)$')
	ax.minorticks_on()
	ax.grid(True, which='both')
	plt.show()


#run question 5
param = {'w': 0.5, 'c1': 1.5, 'c2': 1.5, 'rel_tol': 1e-9, 'penalty': 5, 'boundMethod': 'reflect'}
plotPS4_iter(20,param)