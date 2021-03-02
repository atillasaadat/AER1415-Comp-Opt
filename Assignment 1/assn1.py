#!/usr/bin/env python
""" AER1415 Computer Optimization - Assignment 1

Author: Atilla Saadat
Submitted: Feb 25, 2021
Email: asaadat@utias-sfl.net

Descripton:

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

class Particle:
	def __init__(self, x0, bounds, params):
		self.pos = array(x0[:])
		self.vel = array([random.uniform(*i) for i in bounds])
		self.posBest = None
		self.valBest = None
		self.val = None
		self.params = params
		self.bounds = bounds
		self.penaltyParam = 1

	def calc(self,costFunc,iterNum):
		#setup exponential penalty function
		if self.params.get('penalty',False):
			self.penaltyParam = self.params['penalty']**iterNum
		elif self.params.get('penaltyStatic',False):
			self.penaltyParam = self.params['penaltyStatic']
		#Call cost function with current particle position
		self.val = costFunc(self.pos,iterNum=iterNum,penaltyParam=self.penaltyParam,params=self.params)
		#if new val is less than stored minimum particle val, update value and position
		if self.valBest is None or self.val < self.valBest:
			self.posBest = self.pos
			self.valBest = self.val

	def update_position(self,posGlobBest):
		r1, r2 = random.uniform(size=2)
		vel_cognitive = self.params['c1']*r1*(self.posBest-self.pos.copy())
		vel_social = self.params['c2']*r2*(posGlobBest-self.pos.copy())
		#calcualte new particle velocity
		self.vel = self.params['w']*self.vel + vel_cognitive + vel_social
		
		#Hyperbolic bound approach
		if self.params['boundMethod'] == 'hyperbolic':
			for idx,xNew in enumerate(self.pos):
				if self.vel[idx] > 0:
					self.vel[idx] = self.vel[idx] / (1. + abs(self.vel[idx]/(self.bounds[idx][1]-self.pos[idx])))
				else:
					self.vel[idx] = self.vel[idx] / (1. + abs(self.vel[idx]/(self.pos[idx]-self.bounds[idx][0])))

		#set new particle position from velcotiy addition
		self.pos = add(self.pos,self.vel)

		#Reflect bound approach
		if self.params['boundMethod'] == 'reflect':
			for idx,xNew in enumerate(self.pos):
				if xNew < self.bounds[idx][0]:
					self.pos[idx] = self.bounds[idx][0] + (self.bounds[idx][0] - self.pos[idx])
				elif xNew > self.bounds[idx][1]:
					self.pos[idx] = self.bounds[idx][1] - (self.bounds[idx][1] - self.pos[idx])
		#Nearest bound approach
		if self.params['boundMethod'] == 'nearest':
			for idx,xNew in enumerate(self.pos):
				if xNew < self.bounds[idx][0]:
					self.pos[idx] = self.bounds[idx][0]
					#self.vel[idx] = 0
				elif xNew > self.bounds[idx][1]:
					self.pos[idx] = self.bounds[idx][1]
					#self.vel[idx] = 0
		

class PSO:
	def __init__(self,costFunc,x0,bounds,numParticles,maxRepeat,maxIter,params):
		self.costBestVal = None
		self.posGlobBest = None
		self.iterGlobBest = None
		self.costFunc = costFunc
		self.numParticles = numParticles
		self.maxIter = maxIter
		self.maxRepeat = maxRepeat
		self.bounds = bounds
		self.params = params
		self.x0 = x0

	def optimize(self,verbose=False):
		allResultsDict = {}
		print(self.params)
		#repeat M times to get mean values for parameter set
		for repeatIter in range(self.maxRepeat):
			self.swarm = [Particle(self.x0,self.bounds,self.params) for i in range(self.numParticles)]
			self.currentDiff = None
			self.costBestVal = None
			self.posGlobBest = None
			iterResults = []
			#for N number of iterations per independent run
			for idx in range(self.maxIter):
				for particle in self.swarm:
					#for every particle, calculate new particle val and position at current iteration step
					particle.calc(self.costFunc, idx)
					#update gloval cost function value and position based on all the new particle positions and values
					if self.costBestVal is None or particle.val < self.costBestVal:
						#calcualte current iterations gloval best value differential for convergence
						self.currentDiff = abs(subtract(self.costBestVal,particle.val)) if idx != 0 else None
						self.costBestVal = particle.val
						self.posGlobBest = particle.pos
				iterResults.append(append(self.posGlobBest,[self.costBestVal,self.currentDiff]))
				#store index at which differntial convergence first happens
				try:
					if idx != 0 and self.currentDiff is not None and abs(self.currentDiff) <= self.params['rel_tol'] and self.iterGlobBest is None:
						self.iterGlobBest = idx
				except:
					embed()
				#update all particles with new global best value
				for particle in self.swarm:
					particle.update_position(self.posGlobBest)
				if verbose:
					print('Iter: {}/{} - CostFunc: {}, val: {}, df: {}'.format(idx,self.maxIter,self.posGlobBest,self.costBestVal,self.currentDiff))
			print('{} / {} - CostFunc: {}, val: {}'.format(repeatIter,self.maxRepeat,self.posGlobBest,self.costBestVal))
			allResultsDict[repeatIter] = array(iterResults)
		#calculate mean and std values for later plotting
		repeatResults = array([v.T[-2].T for v in allResultsDict.values()]).T.astype(float)
		bestRun, bestIter = divmod(repeatResults.T.argmin(),repeatResults.T.shape[1])
		meanVals = mean(repeatResults,axis=1)
		meanPos = array([mean(i,axis=1) for i in array([v.T[:-2].T for v in allResultsDict.values()]).T]) 
		meanPosVal = vstack([meanPos,meanVals]).astype(float)
		results = {'minVal': float(repeatResults.T[bestRun][bestIter]), 'x*': (allResultsDict[bestRun][bestIter][:-2]).astype(float), 'iter': bestIter, 'relTolPass': True, 'meanPosVal': meanPosVal,'meanRepeatValues': meanVals, 'stdRepeatValues': std(repeatResults,axis=1)}
		if self.iterGlobBest is None:
			results['iter'] = idx
			results['relTolPass'] = False
		#if getAllResults:
		#	results['allResultsDict'] = allResultsDict
		return results

def P1(x,**kwargs):
	x0 = x[:-1]
	x1 = x[1:]
	return sum(100.0*(x1 - x0**2.0)**2.0 + (1 - x0)**2.0)

def P2(x,**kwargs):
	return (10*len(x) + sum(x**2 - 10*cos(2*pi*x)))

def P3(x,**kwargs):
	#https://www.wolframalpha.com/input/?i=extrema+%5B%2F%2Fmath%3Ax%5E2+%2B+0.5*x+%2B+3*x*y+%2B+5*y%5E2%2F%2F%5D+subject+to+%5B%2F%2Fmath%3A3*x%2B2*y%2B2%3C%3D0%2C15*x-3*y-1%3C%3D0%2C-1%3C%3Dx%3C%3D1%2C-1%3C%3Dy%3C%3D1%2F%2F%5D++
	#(-0.8064516129032258, 0.20967741935483872)
	fx = x[0]**2 + 0.5*x[0] + 3*x[0]*x[1] + 5*x[1]**2
	gx1 = 3*x[0] + 2*x[1] + 2
	gx2 = 15*x[0] - 3*x[1] - 1
	psi = max([0,gx1])**2 + max([0,gx2])**2
	return (fx + kwargs['penaltyParam']*psi)

def P4(x,**kwargs):
	t1 = sum(cos(x)**4)
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

class P5:
	def __init__(self,**kwargs):
		self.time, self.displacement =  loadtxt('MeasuredResponse.dat').T
		self.m = 1.0
		self.F0 = 1.0
		self.w = 0.1

	def groundTruth(self):
		return array([self.time,self.displacement])

	def evaluate(self,x):
		c,k = x
		alpha = arctan2(c*self.w,(k-self.m*self.w**2))
		C = sqrt((k-self.m*self.w**2)**2 + (c*self.w)**2)
		w_d  = cmath.sqrt((k/self.m)-(c/(2*self.m))**2).real
		A = -self.F0/C*cos(alpha)
		B = -(divide(self.F0,(C*w_d)))*(self.w*sin(alpha) + (c/(2*self.m))*cos(alpha))
		u_t = (A*cos(w_d*self.time) + B*sin(w_d*self.time))*exp(divide(-(c*self.time),(2*(self.m)))) + (self.F0/C)*cos(self.w*self.time - alpha)
		if isnan(u_t).any():
			u_t = 0
		return u_t

	def costFuncVal(self,x,**kwargs):
		u_t = self.evaluate(x)
		RMSE = sqrt(square(subtract(u_t,self.displacement)).mean())
		return RMSE

# c1 – cognitive parameter (confidence in itself)
# c2 – social parameter (confidence in the swarm)
# w – inertial weight (inertia of a particle)

# P1
def runPS1(xSize,params):
	print('PS1:\n')
	bounds = array([(-5,5)]*xSize)
	x0 = random.uniform(-5,5,xSize)
	return PSO(P1,x0,bounds,numParticles=max([200,xSize*10]),maxRepeat=10,maxIter=100,params=params).optimize(verbose=False)

def runPS2(xSize,params):
	print('PS2:\n')
	bounds = array([(-5,5)]*xSize)
	x0 = random.uniform(-5,5,xSize)
	return PSO(P2,x0,bounds,numParticles=max([200,xSize*10]),maxRepeat=10,maxIter=100,params=params).optimize(verbose=False)

#(-0.8064516129032258, 0.20967741935483872)
def runPS3(xSize,params):
	print('PS3:\n')
	bounds = array([(-1,1)]*xSize)
	x0 = random.uniform(-1,1,xSize)
	return PSO(P3,x0,bounds,numParticles=max([200,xSize*10]),maxRepeat=10,maxIter=100,params=params).optimize(verbose=False)

def runPS4(xSize, params):
	print('PS4:\n')
	bounds = array([(0,10)]*xSize)
	x0 = random.uniform(0.5,4,xSize)
	return PSO(P4,x0,bounds,numParticles=max([200,xSize*10]),maxRepeat=10,maxIter=100,params=params).optimize(verbose=False)

def runPS5(xSize, params):
	print('PS5:\n')
	bounds = array([(0,100)]*xSize)
	x0 = random.uniform(0,20,xSize)
	oscil = P5()
	return PSO(oscil.costFuncVal,x0,bounds,numParticles=max([200,xSize*10]),maxRepeat=10,maxIter=100,params=params).optimize(verbose=False)


#runPS1()
#runPS3()

# P2
#n = 2
#bounds = [(-5,5)]*n

# P3
def plotPS4_3D(ps4Results):
	x = linspace(0, 10, 100)
	y = linspace(0, 10, 100)
	X, Y = meshgrid(x, y)
	Z = []
	xx = X.flatten()
	yy = Y.flatten()
	for idx in range(xx.size): 	
		Z.append(P4(x=array([xx[idx], yy[idx]]),onlyFx=True))
	Z = array(Z).reshape(X.shape)
	
	fig = plt.figure()
	ax = plt.axes(projection="3d")
	#ax.plot_wireframe(X, Y, Z, color='green')
	ax = plt.axes(projection='3d')
	ax.plot3D(*ps4Results['meanPosVal'],'ro',alpha=0.7,zorder=2)
	ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
	                cmap='viridis', edgecolor='none', antialiased=True, zorder=1, alpha=0.5)

	ax.set_xlabel('$x_1$')
	ax.set_ylabel('$x_2$')
	ax.set_zlabel('$f (x_n)$')
	ax.set_title('P4: Bump Function')
	
	plt.show()


#grid searched parameter combinations
paramCombinations = [
	{'w': 0.3, 'c1': 0.5, 'c2': 1.5, 'rel_tol': 1e-9, 'penalty': 5, 'boundMethod': 'reflect'},
	{'w': 0.3, 'c1': 3.5, 'c2': 0.5, 'rel_tol': 1e-9, 'penalty': 5, 'boundMethod': 'reflect'},
	{'w': 0.3, 'c1': 1.5, 'c2': 1.5, 'rel_tol': 1e-9, 'penalty': 5, 'boundMethod': 'reflect'},
	{'w': 0.5, 'c1': 0.5, 'c2': 3.5, 'rel_tol': 1e-9, 'penalty': 5, 'boundMethod': 'reflect'},
	{'w': 0.5, 'c1': 3.5, 'c2': 0.5, 'rel_tol': 1e-9, 'penalty': 5, 'boundMethod': 'reflect'},
	{'w': 0.5, 'c1': 1.5, 'c2': 1.5, 'rel_tol': 1e-9, 'penalty': 5, 'boundMethod': 'reflect'},
	{'w': 0.7, 'c1': 0.5, 'c2': 3.5, 'rel_tol': 1e-9, 'penalty': 5, 'boundMethod': 'reflect'},
	{'w': 0.7, 'c1': 3.5, 'c2': 0.5, 'rel_tol': 1e-9, 'penalty': 5, 'boundMethod': 'reflect'},
	{'w': 0.7, 'c1': 1.5, 'c2': 1.5, 'rel_tol': 1e-9, 'penalty': 5, 'boundMethod': 'reflect'},
	{'w': 0.9, 'c1': 0.5, 'c2': 3.5, 'rel_tol': 1e-9, 'penalty': 5, 'boundMethod': 'reflect'},
	{'w': 0.9, 'c1': 3.5, 'c2': 0.5, 'rel_tol': 1e-9, 'penalty': 5, 'boundMethod': 'reflect'},
	{'w': 0.9, 'c1': 1.5, 'c2': 1.5, 'rel_tol': 1e-9, 'penalty': 5, 'boundMethod': 'reflect'},
	]

def plotPS4_iter(dim,paramCombinations):
	fig, ax = plt.subplots()
	for paramsCombo in paramCombinations:
		ps4Results = runPS4(dim,paramsCombo)
		penalty = 'rho: {}'.format(paramsCombo['penalty']) if  paramsCombo.get('penalty',False) else 'rho_static: {}'.format(paramsCombo['penaltyStatic'])
		bound = 'boundMethod: {}'.format(paramsCombo['boundMethod'])
		ax.plot(ps4Results['meanRepeatValues'],label='minVal:{:.8f}, cIter: {} - w: {:.5f}, c1: {:.5f}, c2: {:.5f}, {}, {}'.format(ps4Results['minVal'],ps4Results['iter'],paramsCombo['w'],paramsCombo['c1'],paramsCombo['c2'],penalty,bound))
		ax.fill_between(range(len(ps4Results['meanRepeatValues'])), ps4Results['meanRepeatValues']-ps4Results['stdRepeatValues'],  ps4Results['meanRepeatValues']+ps4Results['stdRepeatValues'], alpha=0.2)
	ax.legend()
	ax.set_title('PS4 Bump Test n={} - Objective Func. Val (w/ Quad. Penalty) vs. Iteration Number'.format(dim))
	ax.set_xlabel('Iteration #')
	ax.set_ylabel(r'$\pi(x,\rho)$')
	ax.minorticks_on()
	ax.grid(True, which='both')
	plt.show()

def plotPS5(results):
	fig, ax = plt.subplots()
	oscil = P5()
	ax.plot(*oscil.groundTruth(),label='Measured Response')
	ax.plot(oscil.time,oscil.evaluate(results['x*']),label='Estimated Response - c: {:.6f}, k: {:.6f}'.format(*results['x*']))
	ax.legend()
	ax.set_title('PS5 Dynamic Response - Displacement Response vs. Time')
	ax.set_xlabel('Time [t]')
	ax.set_ylabel('Displace response [u]')
	ax.minorticks_on()
	ax.grid(True, which='both')
	plt.show()
	return results

#--------------------------------------------------------------------------------------------------------------
plotPS4_iter(2,paramCombinations)
#plotPS4_iter(50,paramCombinations)
selectParams = {'w': 0.5, 'c1': 1.5, 'c2': 1.5, 'rel_tol': 1e-9, 'penalty': 5, 'boundMethod': 'reflect'}
ps4_n2 = runPS4(2,selectParams)
print('\nPS4 N=2 Soln:\nx* = {}\nf(x*) = {:.6f}'.format(ps4_n2['x*'],ps4_n2['minVal']))
plotPS4_3D(ps4_n2)
ps4_n10 = runPS4(10,selectParams)
print('\nPS4 N=10 Soln:\nx* = {}\nf(x*) = {:.6f}'.format(ps4_n10['x*'],ps4_n10['minVal']))
ps4_n50 = runPS4(50,selectParams)
print('\nPS4 N=50 Soln:\nx* = {}\nf(x*) = {:.6f}'.format(ps4_n50['x*'],ps4_n50['minVal']))
#--------------------------------------------------------------------------------------------------------------
for n in [2,5]:
	ps1 = runPS1(n,selectParams)
	print('\nPS1 N={} Soln:\nx* = {}\nf(x*) = {:.6f}\n'.format(n,ps1['x*'],ps1['minVal']))
#--------------------------------------------------------------------------------------------------------------
for n in [2,5]:
	ps2 = runPS2(n,selectParams)
	print('\nPS2 N={} Soln:\nx* = {}\nf(x*) = {:.6f}\n'.format(n,ps2['x*'],ps2['minVal']))
#--------------------------------------------------------------------------------------------------------------
ps3 = runPS3(2,selectParams)
print('\nPS3 N={} Soln:\nx* = {}\nf(x*) = {:.6f}\n'.format(2,ps3['x*'],ps3['minVal']))
#--------------------------------------------------------------------------------------------------------------
ps5 = plotPS5(runPS5(2,selectParams))
print('\nPS5 N={} Soln:\nx* = {}\nf(x*) = {:.6f}\n'.format(2,ps5['x*'],ps5['minVal']))
embed()
#
