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
	def __init__(self, x0, bounds, w, c1, c2):
		self.pos = array(x0[:])
		self.vel = random.uniform(-1,1,len(x0))
		self.posBest = None
		self.errBest = None
		self.err = None
		self.w = w
		self.c1 = c1
		self.c2 = c2
		self.bounds = bounds

	def calc(self,costFunc,iterNum,phi_C,phi_alpha):
		try:
			self.err = costFunc(self.pos)
		except TypeError:
			self.err = costFunc(self.pos,iterNum,phi_C,phi_alpha)

		if self.errBest is None or self.err < self.errBest:
			self.posBest = self.pos
			self.errBest = self.err

	def update_velocity(self,posGlobBest):
		r1, r2 = random.random(2)

		vel_cognitive = self.c1*r1*(self.posBest-self.pos)
		vel_social = self.c2*r2*(posGlobBest-self.pos)
		self.vel = self.w*self.vel + vel_cognitive + vel_social

	def update_position(self):
		self.pos = add(self.pos,self.vel)

		for idx,xNew in enumerate(self.pos):
			if xNew < self.bounds[idx][0]:
				self.pos[idx] = self.bounds[idx][0]
			if xNew > self.bounds[idx][1]:
				self.pos[idx] = self.bounds[idx][1]

class PSO:
	def __init__(self,costFunc,x0,bounds,numParticles,maxIter,w=0.4, c1=1.75, c2=2.25,phi_C=0.5,phi_alpha=1.5):
		self.errGlobBest = None
		self.posGlobBest = None
		self.costFunc = costFunc
		self.maxIter = maxIter
		self.bounds = bounds
		self.swarm = [Particle(x0,bounds,w,c1,c2) for i in range(numParticles)]
		self.phi_C = phi_C
		self.phi_alpha = phi_alpha

	def optimize(self,verbose=False):
		for idx in range(self.maxIter):
			for particle in self.swarm:
				particle.calc(self.costFunc,idx, self.phi_C, self.phi_alpha)
				if self.errGlobBest is None or particle.err < self.errGlobBest:
					self.errGlobBest = particle.err
					self.posGlobBest = particle.pos
			
			for particle in self.swarm:
				particle.update_velocity(self.posGlobBest)
				particle.update_position()
			if verbose:
				print('Iter: {}/{} - CostFunc: {}, err: {}'.format(idx,self.maxIter,self.posGlobBest,self.errGlobBest))
		print('Finished PSO!\nCostFunc: {}, err: {}'.format(self.posGlobBest,self.errGlobBest))

def P1(x):
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

def P2(x):
	return 10*len(x) + sum([(i**2 - 10*cos(2*pi*i)) for i in x])

def P3(x,iterNum,phi_C,phi_alpha):

	fx = x[0]**2 + 0.5*x[0] + 3*x[0]*x[1] + 5*x[1]**2
	gx1 = 3*x[0] + 2*x[1] + 2
	gx2 = 15*x[0] - 3*x[1] - 1
	psi = max([0,gx1])**2 + max([0,gx2])**2
	phi = phi_C*iterNum**phi_alpha
	return (fx + psi*phi)

def P4(x):
	t1 = sum([cos(i)**4 for i in x])
	t2 = prod([cos(i)**2 for i in x])
	t3 = sum([(idx+1)*i**2 for idx,i in enumerate(x)])
	return -abs(t1-2*t2)/sqrt(t3)

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


# P1
'''
print('PS1:\n')
for n in [2,5]:
	bounds = [(-5,5)]*n
	x0 = random.uniform(-5,5,n)
	P1opt = PSO(P1,x0,bounds,numParticles=100,maxIter=100,w=0.4, c1=3.5, c2=2)
	P1opt.optimize(verbose=True)
'''
print('PS1:\n')
n=2
bounds = [(-1,1)]*2
x0 = random.uniform(-1,1,n)
P1opt = PSO(P3,x0,bounds,numParticles=100,maxIter=100,w=0.4, c1=2, c2=1.5,phi_C=0.5,phi_alpha=1.9)
P1opt.optimize(verbose=True)

# P2
#n = 2
#bounds = [(-5,5)]*n

# P3