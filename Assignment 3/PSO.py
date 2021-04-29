#!/usr/bin/env python
""" 
AER1415 Computer Optimization - Assignment 3

Author: Atilla Saadat
Submitted: April 28, 2021
Email: asaadat@utias-sfl.net

Descripton: Question 5 - Original PSO Algorithm

"""

from numpy import *

class Particle:
	def __init__(self, costFunc, x0, bounds, params):
		self.costFunc = costFunc
		self.pos = array(x0[:])
		self.vel = array([random.uniform(*i) for i in bounds])
		self.posBest = None
		self.valBest = None
		self.val = None
		self.params = params
		self.bounds = bounds
		self.penaltyParam = 1

	def calc(self,iterNum):
		#setup exponential penalty function
		if self.params.get('penalty',False):
			self.penaltyParam = self.params['penalty']**iterNum
		elif self.params.get('penaltyStatic',False):
			self.penaltyParam = self.params['penaltyStatic']
		#Call cost function with current particle position
		self.val = self.costFunc(self.pos,iterNum=iterNum,penaltyParam=self.penaltyParam,params=self.params)
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
		self.x0 = x0
		self.posGlobBest = None
		self.iterGlobBest = None
		self.costFunc = costFunc
		self.numParticles = numParticles
		self.maxIter = maxIter
		self.maxRepeat = maxRepeat
		self.bounds = bounds
		self.params = params

	def optimize(self,verbose=False):
		allResultsDict = {}
		print(self.params)
		#repeat M times to get mean values for parameter set
		for repeatIter in range(self.maxRepeat):
			self.swarm = [Particle(self.costFunc,self.x0,self.bounds,self.params) for i in range(self.numParticles)]
			self.currentDiff = None
			self.costBestVal = None
			self.posGlobBest = None
			iterResults = []
			#for N number of iterations per independent run
			for idx in range(self.maxIter):
				for particle in self.swarm:
					#for every particle, calculate new particle val and position at current iteration step
					particle.calc(idx)
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
		#results = {'minVal': float(repeatResults.T[bestRun][bestIter]), 'x*': (allResultsDict[bestRun][bestIter][:-2]).astype(float), 'iter': bestIter, 'relTolPass': True, 'meanPosVal': meanPosVal,'meanRepeatValues': meanVals, 'stdRepeatValues': std(repeatResults,axis=1)}
		results = {'minMeanVal': meanVals[-1], 'x*': (allResultsDict[bestRun][bestIter][:-2]).astype(float), 'iter': bestIter, 'relTolPass': True, 'meanPosVal': meanPosVal,'meanRepeatValues': meanVals, 'stdRepeatValues': std(repeatResults,axis=1)}
		if self.iterGlobBest is None:
			results['iter'] = idx
			results['relTolPass'] = False
		#if getAllResults:
		#	results['allResultsDict'] = allResultsDict
		return results
