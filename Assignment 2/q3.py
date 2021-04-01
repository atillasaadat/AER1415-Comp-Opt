#!/usr/bin/env python
"""
AER1415 Computer Optimization - Assignment 2

Author: Atilla Saadat
Submitted: Mar 31, 2021
Email: asaadat@utias-sfl.net

Question 3

"""

from numpy import *
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.colors as colors
import scipy.io
from IPython import embed

from rosen import *


#Convex Quadratic functions
def CQ(x,A,b):
	x_t = x.reshape([x.shape[0], 1])
	#val = (0.5*x_t*A*x - x_t*b)
	val = 0.5*dot(dot(x.T,A),x) - dot(x,b)
	return val

#Convex Quadratic gradient
def CQ_g(x,A,b):
	return (dot(A,x) - b)

def BB(consts,verbose=False,plot=False):
	#BB - steepest descent

	#import convex quadratic dataset
	mat = scipy.io.loadmat('ConvexQuadratic.mat')
	A = mat['A'].toarray()
	b = mat['b'].flatten()

	#initialize parameters
	x0 = random.uniform(-1,1,len(b))
	x = [x0]
	vals = [CQ(x0,A,b)]
	g = [CQ_g(x0,A,b)]
	passConditions = []
	I = eye(len(x0))
	for iterNum in range(10000):
		x_k = x[-1]
		val_k = vals[-1]
		g_k = g[-1]

		#l2-norm break criterion
		if not linalg.norm(g_k) > consts['e_g'] :
			print('l2 break - {} <= {}'.format(linalg.norm(g_k),consts['e_g']))
			break
		
		#solve for search direction from gradient vector
		p_k = -g_k

		#Q4 - http://www.princeton.edu/~aaa/Public/Teaching/ORF363_COS323/F14/ORF363_COS323_F14_Lec8.pdf
		if iterNum == 0:
			#exact line search for first iteration
			a1 = dot(g_k.T,g_k)
			a2 = dot(dot(g_k.T,A),g_k)
			a_k = divide(a1,a2)
		else:
			#BB method for k>1
			s_k = x_k - x[-2]
			y_k = g_k - g[-2]
			a_k = dot(s_k,s_k)/dot(s_k,y_k)

		#multiply a_k to calculate new step values and position
		x_k1 = x_k + a_k*p_k
		val_k1 = CQ(x_k1,A,b)
		g_k1 = CQ_g(x_k1,A,b)

		#check pass condition
		#numpy isclose: absolute(a - b) <= (atol + rtol * absolute(b))
		passCond = isclose(val_k1,val_k,atol=consts['e_a'],rtol=consts['e_r'])
		if len(passConditions) >= 2 and all(passConditions[-2:]):
			break
		
		passConditions.append(passCond)
		#if passCond:
		#	embed()
		x.append(x_k1)
		vals.append(val_k1)
		g.append(g_k1)
	
	#compute steepest descent for current run for comparision
	sd_x, sd_vals, sd_l2_g = steepest_descent(consts)

	if verbose:
		for idx,val in enumerate(vals):
			print('Iter: {}, val: {}'.format(idx,val))

	#plot convergence and gradient graphs
	if plot:
		fig, ax = plt.subplots(2,1)
		fig.set_size_inches(18.5, 10.5)
		ax[0].plot(vals,label=r'BB Last Iter: {} - $f(x_k)$: {:.6f}'.format(len(vals),vals[-1]))
		ax[0].plot(sd_vals,label=r'SD Last Iter: {} - $f(x_k)$: {:.6f}'.format(len(sd_vals),sd_vals[-1]))
		ax[0].set_ylabel(r'$f(x_k)$')
		ax[0].set_title('Barzilai & Borwein (BB) and Steepest Descent methods - Convex Quadratic function')
		ax[0].minorticks_on()
		ax[0].grid(True, which='both')
		ax[0].legend()

		ax[1].plot(linalg.norm(g,axis=1),label='BB method')
		ax[1].plot(sd_l2_g,label='SD method')
		ax[1].set_ylabel(r'${\vert \vert \nabla f(x_k) \| \|}_2$')
		ax[1].set_xlabel('Iteration Number, $k$')
		ax[1].minorticks_on()
		ax[1].grid(True, which='both')
		plt.savefig('Q34_BBandSD_ConvexQuadratic.svg',dpi=1000, format = 'svg', bbox_inches = 'tight')
		plt.show()
	print('BB method: iter: {}, val: {}'.format(len(x),vals[-1]))
	return [array(x),array(vals),linalg.norm(g,axis=1)]

def steepest_descent(consts):
	#import convex quadratic dataset
	mat = scipy.io.loadmat('ConvexQuadratic.mat')
	A = mat['A'].toarray()
	b = mat['b'].flatten()

	#initialize parametsd
	x0 = random.uniform(-1,1,len(b))
	x = [x0]
	vals = [CQ(x0,A,b)]
	g = [CQ_g(x0,A,b)]
	passConditions = []
	I = eye(len(x0))
	for iterNum in range(10000):
		x_k = x[-1]
		val_k = vals[-1]
		g_k = g[-1]

		#l2-norm break criterion
		if not linalg.norm(g_k) > consts['e_g'] :
			print('l2 break - {} <= {}'.format(linalg.norm(g_k),consts['e_g']))
			break

		#steepest gradient descent search direction
		p_k = -g_k

		#the objective function is quadratic, line search can be carried out exactly
		a1 = dot(g_k.T,g_k)
		a2 = dot(dot(g_k.T,A),g_k)
		a_k = divide(a1,a2)

		x_k1 = x_k + a_k*p_k
		val_k1 = CQ(x_k1,A,b)
		g_k1 = CQ_g(x_k1,A,b)

		#check pass condition
		#numpy isclose: absolute(a - b) <= (atol + rtol * absolute(b))
		passCond = isclose(val_k1,val_k,atol=consts['e_a'],rtol=consts['e_r'])		
		if len(passConditions) >= 2 and all(passConditions[-2:]):
			break
		
		passConditions.append(passCond)
		x.append(x_k1)
		vals.append(val_k1)
		g.append(g_k1)
	print('Steepest-Descent:  iter: {}, val: {}'.format(len(x),vals[-1]))
	return [array(x),array(vals),linalg.norm(g,axis=1)]

'''
	- e_r : relative tolerance
	- e_a : absolute tolerance
	- e_g : absolute gradient tolerance
'''
consts = {'e_r': 1e-5, 'e_g': 1e-5, 'e_a': 1e-8}

bb_x = []
bb_vals = []
bb_iters = []
sd_x = []
sd_vals = []
sd_iters = []
for i in range(20):
	print('\nRound {} / {}'.format(i+1,20))
	bb = BB(consts)
	sd = steepest_descent(consts)
	bb_x.append(bb[0][-1])
	bb_vals.append(bb[1][-1])
	bb_iters.append(len(bb[0]))
	sd_x.append(sd[0][-1])
	sd_vals.append(sd[1][-1])
	sd_iters.append(len(sd[0]))
print('\n{}'.format('-'*50))
print('BB Avgs - iter: {}, x*: {}, val: {}'.format(round(mean(bb_iters)),repr(mean(bb_x,axis=0)),mean(bb_vals)))
print('SD Avgs - iter: {}, x*: {}, val: {}'.format(round(mean(sd_iters)),repr(mean(sd_x,axis=0)),mean(sd_vals)))

BB(consts,verbose=True,plot=True)


