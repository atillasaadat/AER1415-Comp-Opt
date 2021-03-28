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

from rosen import *

#Convex Quadratic
def CQ(x,A,b):
	x_t = x.reshape([x.shape[0], 1])
	#val = (0.5*x_t*A*x - x_t*b)
	val = 0.5*dot(dot(x.T,A),x) - dot(x,b)
	return val

def CQ_g(x,A,b):
	return (dot(A,x) - b)

def BB(consts,verbose=False,plot=False):
	#BB - steepest descent

	mat = scipy.io.loadmat('ConvexQuadratic.mat')
	A = mat['A'].toarray()
	b = mat['b'].flatten()

	x0 = random.uniform(-1,1,len(b))
	x = [x0]
	vals = [CQ(x0,A,b)]
	g = [CQ_g(x0,A,b)]
	passConditions = []
	I = eye(len(x0))
	#http://bicmr.pku.edu.cn/~wenzw/courses/WenyuSun_YaxiangYuan_BB.pdf
	for iterNum in range(10000):
		x_k = x[-1]
		val_k = vals[-1]
		g_k = g[-1]

		if not linalg.norm(g_k) > consts['e_g'] :
			print('l2 break - {} <= {}'.format(linalg.norm(g_k),consts['e_g']))
			break
		
		p_k = -g_k

		#Q4 - http://www.princeton.edu/~aaa/Public/Teaching/ORF363_COS323/F14/ORF363_COS323_F14_Lec8.pdf
		if iterNum == 0:
			#exact line search for first iteration
			a1 = dot(g_k.T,g_k)
			a2 = dot(dot(g_k.T,A),g_k)
			a_k = divide(a1,a2)
		else:
			s_k = x_k - x[-2]
			y_k = g_k - g[-2]
			a_k = dot(s_k,s_k)/dot(s_k,y_k)


		x_k1 = x_k + a_k*p_k
		val_k1 = CQ(x_k1,A,b)
		g_k1 = CQ_g(x_k1,A,b)

		#numpy isclose: absolute(a - b) <= (atol + rtol * absolute(b))
		passCond = isclose(val_k1,val_k,atol=consts['e_a'],rtol=consts['e_r'])
		#passCond = abs(val_k1 - val_k)<=(e_a + e_r*abs(val_k))
		#if passCond:
		#	embed()
		#print('{} + {}*{}'.format(x_k,a_k,g_k))
		
		#print('{} passCon: {} - {} <= {}'.format(iterNum,passCond,abs(val_k1 - val_k),(consts['e_a'] + consts['e_r']*abs(val_k))))
		if len(passConditions) >= 2 and all(passConditions[-2:]):
			#print('passCond break - {}'.format(passConditions))
			break
		
		passConditions.append(passCond)
		#if passCond:
		#	embed()
		x.append(x_k1)
		vals.append(val_k1)
		g.append(g_k1)
	
	sd_x, sd_vals, sd_l2_g = steepest_descent(consts)

	if verbose:
		for idx,val in enumerate(vals):
			print('Iter: {}, val: {}'.format(idx,val))

	if plot:
		fig, ax = plt.subplots(2,1)
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
		plt.show()

def steepest_descent(consts):
	mat = scipy.io.loadmat('ConvexQuadratic.mat')
	A = mat['A'].toarray()
	b = mat['b'].flatten()

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

		if not linalg.norm(g_k) > consts['e_g'] :
			print('l2 break - {} <= {}'.format(linalg.norm(g_k),consts['e_g']))
			break

		#Q4 - http://www.princeton.edu/~aaa/Public/Teaching/ORF363_COS323/F14/ORF363_COS323_F14_Lec8.pdf
		p_k = -g_k

		#the objective function is quadratic, line search can be carried out exactly
		a1 = dot(g_k.T,g_k)
		a2 = dot(dot(g_k.T,A),g_k)
		a_k = divide(a1,a2)


		x_k1 = x_k + a_k*p_k
		val_k1 = CQ(x_k1,A,b)
		g_k1 = CQ_g(x_k1,A,b)

		#numpy isclose: absolute(a - b) <= (atol + rtol * absolute(b))
		passCond = isclose(val_k1,val_k,atol=consts['e_a'],rtol=consts['e_r'])
		
		#print('{} passCon: {} - {} <= {}'.format(iterNum,passCond,abs(val_k1 - val_k),(consts['e_a'] + consts['e_r']*abs(val_k))))
		if len(passConditions) >= 2 and all(passConditions[-2:]):
			#print('passCond break - {}'.format(passConditions))
			break
		
		passConditions.append(passCond)
		x.append(x_k1)
		vals.append(val_k1)
		g.append(g_k1)

	return [array(x),array(vals),linalg.norm(g,axis=1)]

'''
	- e_r : relative tolerance
	- e_a : absolute tolerance
	- e_g : absolute gradient tolerance
'''
consts = {'e_r': 1e-8, 'e_g': 1e-5, 'e_a': 1e-8}

BB(consts,verbose=True,plot=True)


