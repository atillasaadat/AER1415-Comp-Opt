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
import timeit

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
		self.val = costFunc(self.pos,iterNum=iterNum,penaltyParam=self.penaltyParam,params=self.params)
		if 'penalty' in self.params.keys():
			self.penaltyParam *= self.params['penalty']

		if self.valBest is None or self.val < self.valBest:
			self.posBest = self.pos
			self.valBest = self.val

	def update_velocity(self,posGlobBest):
		r1, r2 = random.uniform(size=2)

		vel_cognitive = self.params['c1']*r1*(self.posBest-self.pos)
		vel_social = self.params['c2']*r2*(posGlobBest-self.pos)
		self.vel = self.params['w']*self.vel + vel_cognitive + vel_social

	def update_position(self):
		self.pos = add(self.pos,self.vel)

		for idx,xNew in enumerate(self.pos):
			if xNew < self.bounds[idx][0]:
				self.pos[idx] = self.bounds[idx][0]
			if xNew > self.bounds[idx][1]:
				self.pos[idx] = self.bounds[idx][1]

class PSO:
	def __init__(self,costFunc,x0,bounds,numParticles,maxIter,params):
		self.costBestVal = None
		self.posGlobBest = None
		self.iterGlobBest = None
		self.costFunc = costFunc
		self.maxIter = maxIter
		self.bounds = bounds
		self.swarm = [Particle(x0,bounds,params) for i in range(numParticles)]
		self.params = params

	def optimize(self,verbose=False,getResults=False):
		for idx in range(self.maxIter):
			for particle in self.swarm:
				particle.calc(self.costFunc, idx)
				if self.costBestVal is None or particle.val < self.costBestVal:
					self.costBestVal = particle.val
					self.posGlobBest = particle.pos
			
			if self.costBestVal <= self.params['rel_tol'] and self.iterGlobBest is None:
				self.iterGlobBest = idx

			for particle in self.swarm:
				particle.update_velocity(self.posGlobBest)
				particle.update_position()
			if verbose:
				print('Iter: {}/{} - CostFunc: {}, val: {}'.format(idx,self.maxIter,self.posGlobBest,self.costBestVal))
		print('Finished PSO!\nCostFunc: {}, val: {}'.format(self.posGlobBest,self.costBestVal))
		if getResults:
			results = {'val': self.costBestVal, 'x*': self.posGlobBest, 'iter': self.iterGlobBest, 'relTolPass': True}
			if self.iterGlobBest is None:
				results['iter'] = idx
				results['relTolPass'] = False
			return results

def P1(x,**kwargs):
	x0 = x[:-1]
	x1 = x[1:]
	return sum(100.0*(x1 - x0**2.0)**2.0 + (1 - x0)**2.0)

# ^ better function call for Rosen Func than Scipy.optimize.rosen! try it out
'''
from scipy.optimize import rosen
import timeit
print(timeit.Timer(lambda: P1(x)).timeit(1000))
print(timeit.Timer(lambda: rosen(x)).timeit(1000))

'''

def P2(x,**kwargs):
	return 10*len(x) + sum(x**2 - 10*cos(2*pi*x))

def P3(x,**kwargs):
	#https://www.wolframalpha.com/input/?i=extrema+%5B%2F%2Fmath%3Ax%5E2+%2B+0.5*x+%2B+3*x*y+%2B+5*y%5E2%2F%2F%5D+subject+to+%5B%2F%2Fmath%3A3*x%2B2*y%2B2%3C%3D0%2C15*x-3*y-1%3C%3D0%2C-1%3C%3Dx%3C%3D1%2C-1%3C%3Dy%3C%3D1%2F%2F%5D++
	#(-0.8064516129032258, 0.20967741935483872)
	fx = x[0]**2 + 0.5*x[0] + 3*x[0]*x[1] + 5*x[1]**2
	gx1 = 3*x[0] + 2*x[1] + 2
	gx2 = 15*x[0] - 3*x[1] - 1
	psi = max([0,gx1])**2 + max([0,gx2])**2
	#phi = params['phiC']*iterNum**params['phiAlpha']
	#phi = iterNum
	#print(kwargs['penaltyParam'])
	return (fx + kwargs['penaltyParam']*psi)

def P4(x,**kwargs):
	t1 = sum(cos(x)**4)
	t2 = prod(cos(x)**2)
	t3 = sum((arange(len(x))+1)*x**2)
	gx1 = 0.75 - prod(x)
	gx2 = prod(x) - (15*len(x)/2.)
	fx = -abs(t1-2*t2)/sqrt(t3)
	psi = max([0,gx1])**2 + max([0,gx2])**2
	return (fx + kwargs['penaltyParam']*psi)

class P5:
	def __init__(self):
		self.time, self.displacement =  loadtxt('MeasuredResponse.dat').T
		self.m = 1.0
		self.m = 1.0
		self.F0 = 1.0
		self.w = 0.1

	def evaluate(self,c,k):
		costFuncVal = 0
		for idx,t in enumerate(self.time):
			alpha = arctan2(c*self.w,(k-self.m*self.w**2))
			C = sqrt((k-self.m*self.w**2)**2 + (c*self.w)**2)
			w_d  = sqrt((k/self.m)-(c/(2*self.m))**2)
			A = self.F0/C
			B = -(self.F0/(C*w_d))*(self.w*sin(alpha) + (c/(2*self.m))*cos(alpha))
			u_t = (A*cos(w_d*t) + B*sin(w_d*t))*exp(-(c*t)/(2(self.m))) + (self.F0/C)*cos(w*t - alpha)
			costFuncVal += (u_t - self.displacement) #residual
		return costFuncVal


# c1 – cognitive parameter (confidence in itself)
# c2 – social parameter (confidence in the swarm)
# w – inertial weight (inertia of a particle)

# P1
def runPS1():
	print('PS1:\n')
	params = {'w': 0.7289, 'c1': 2.05*0.7289, 'c2': 2.05*0.7289,'rel_tol': 1e-9}
	for n in [2,5]:
		bounds = array([(-5,5)]*n)
		x0 = random.uniform(-5,5,n)
		P1opt = PSO(P1,x0,bounds,numParticles=800,maxIter=100,params=params)
		#P1opt.optimize(verbose=True)
		print(P1opt.optimize(verbose=True,getResults=True))

#(-0.8064516129032258, 0.20967741935483872)
def runPS3():
	print('PS3:\n')
	n=2
	bounds = array([(-1,1)]*n)
	x0 = random.uniform(-1,1,n)
	params = {'w': 0.7289, 'c1': 2.05*0.7289, 'c2': 2.05*0.7289, 'rel_tol': 1e-9, 'penalty': 5.}
	P3opt = PSO(P3,x0,bounds,numParticles=800,maxIter=100,params=params)
	P3opt.optimize(verbose=True)

def runPS4():
	print('PS4:\n')
	for n in [2,10]:
		bounds = array([(0,10)]*n)
		x0 = random.uniform(0,10,n)
		params = {'w': 0.7289, 'c1': 2.05*0.7289, 'c2': 2.05*0.7289, 'rel_tol': 1e-9, 'penalty': 5.}
		P3opt = PSO(P4,x0,bounds,numParticles=500,maxIter=200,params=params)
		P3opt.optimize(verbose=True,getResults=True)

#runPS1()
#runPS3()
runPS4()
# P2
#n = 2
#bounds = [(-5,5)]*n

# P3