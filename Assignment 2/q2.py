#!/usr/bin/env python
"""
AER1415 Computer Optimization - Assignment 2

Author: Atilla Saadat
Submitted: Mar 31, 2021
Email: asaadat@utias-sfl.net

Question 2

"""

from numpy import *
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.colors as colors
from IPython import embed
'''
import os
from matplotlib import pyplot as plt
from IPython import embed
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib
import cmath
import matplotlib.colors as colors
import scipy.io
'''

from rosen import *

def quasi_newton(n,consts,plot=False,verbose=False):
	#https://www.wolframalpha.com/input/?i=d%2Fdx+100%28y%E2%88%92x%5E2%29%5E2%2B%281%E2%88%92x%29%5E2
	#d/dx -> 2*(200*x1**3 - 200*x1*x2 + x1 - 1)
	#x0 = array([2,3])
	x0 = random.uniform(-5,5,n)
	x = [x0]
	vals = [rosen(x0)]
	g = [rosen_gk(x0)]
	passConditions = []
	I = eye(len(x0))
	B_inv = [I]
	for iterNum in range(1000):
		x_k = x[-1]
		val_k = vals[-1]
		B_k_inv = B_inv[-1]
		g_k = g[-1]

		#if isclose(l2(g_k),e_g,atol=e_g,rtol=0):
		if not linalg.norm(g_k) > consts['e_g'] :
			print('l2 break - {} <= {}'.format(linalg.norm(g_k),consts['e_g']))
			break

		#p_k = -dot(B_k_inv,g_k)
		p_k = -linalg.solve(B_k_inv,g_k)
		#p_k = -B_k_inv.T.dot(g_k)
		#p_k = -dot(B_k,g_k)

		#Armijo sufficient decrease condition
		a_k = consts['a_k_init']
		t1 = rosen(x_k + a_k*p_k)
		t2 = rosen(x_k) + consts['u_1']*a_k*dot(g_k.T,p_k)
		#print('a_k: {} , x* = {} -> {} <= {} - diff: {}'.format(a_k,x_k + a_k*p_k,t1,t2, abs(t1-t2)))
		while rosen(x_k + a_k*p_k) > (rosen(x_k) + consts['u_1']*a_k*dot(g_k.T,p_k)):
		#while not (isclose(t1,t2,atol=e_a,rtol=e_r)):
			if a_k < 1e-6:
				print("Problem converging Armijo conditions")
			a_k *= consts['alpha_rho']
			#print('a_k: {} , x* = {} -> {} <= {} - diff: {}'.format(a_k,x_k + a_k*p_k,t1,t2, abs(t1-t2)))
			#g1 = abs(dot(ps1_gk(x_k1).T,p_k))
			#g2 = u_2*abs(dot(g_k.T,p_k))
			#print('{} - {} <= {}'.format(a_k,t1,t2))

		x_k1 = x_k + a_k*p_k
		val_k1 = rosen(x_k1)

		#numpy isclose: absolute(a - b) <= (atol + rtol * absolute(b))
		passCond = isclose(val_k1,val_k,atol=consts['e_a'],rtol=consts['e_r'])

		#print('{} + {}*{}'.format(x_k,a_k,p_k))
		#print('{} passCon: {} - {} <= {}'.format(iterNum,passCond,abs(val_k1 - val_k),(consts['e_a'] + consts['e_r']*abs(val_k))))

		passConditions.append(passCond)
		if len(passConditions) >= 2 and all(passConditions[-2:]):
			print('passCond break - {}'.format(passConditions))
			break

		#BFGS

		g_k1 = rosen_gk(x_k1)
		s_k = x_k1 - x_k
		y_k = g_k1 - g_k
		x_len = x_k.shape[0]

		y_t = y_k.reshape([x_len, 1])
		Bk = dot(B_k_inv, s_k)
		k_t_B = dot(s_k, B_k_inv)
		kBk = dot(dot(s_k, B_k_inv), s_k)
		B_k1_inv = B_k_inv + y_t*y_k/dot(y_k, s_k) - Bk.reshape([x_len, 1]) * k_t_B / kBk

		if any(isnan(B_k1_inv)):
			print('B_k1_inv is nan')
			break
		
		B_inv.append(B_k1_inv)
		x.append(x_k1)
		vals.append(val_k1)
		g.append(g_k1)

	x = array(x)
	vals = array(vals)
	g = array(g)

	if verbose:
		for idx,x_k in enumerate(x):
			print('Iter: {} - x*: {}, val:: {}'.format(idx+1,x_k,vals[idx]))
	
	if plot:
		if n==2:
			print('Creating Contour Plot - Please Wait')
			xx = linspace(min(min(x.T[0]),-5), max(max(x.T[0]),5), consts['contourQuality'])
			yy = linspace(min(min(x.T[1]),-5), max(max(x.T[1]),5), consts['contourQuality'])
			X, Y = meshgrid(xx,yy)
			Z = array([rosen([j,i]) for i,j in zip(X.flatten(),Y.flatten())]).reshape(X.shape)
			fig, ax = plt.subplots()
			pcm = ax.pcolormesh(X, Y, Z,norm=colors.SymLogNorm(linthresh=0.03,vmin=Z.min(), vmax=Z.max()),cmap='RdBu_r')
			ax.plot(*x.T,'-o',color='#16F41B',markersize=2)
			ax.set_xlabel(r'$x_1$')
			ax.set_ylabel(r'$x_2$')
			ax.set_title('Quasi-Newton method w/ BFGS & Armijio Cond. - Rosenbrock Test Function - n={}'.format(n))
			#pcm = ax.pcolormesh(X, Y, Z, cmap='RdBu_r')
			fig.colorbar(pcm, ax=ax, extend='both')
			#ax.contourf(X, Y, Z)
			#plt.contour(X, Y, Z, 20, cmap='RdGy');
			print('Contour Plot Done!')

		sd_x, sd_vals, sd_l2_g = steepest_descent(n,consts)

		fig2, ax2 = plt.subplots(2,1)
		ax2[0].plot(vals, label=r'Last Iter: {} - $x^\star$: {}, $f(x_k)$: {:.6f}'.format(len(vals),around(x[-1],6),vals[-1]))
		ax2[0].plot(sd_vals,label=r'SD Last Iter: {} - $x^\star$: {}, $f(x_k)$: {:.6f}'.format(len(sd_vals),around(sd_x[-1],6),sd_vals[-1]))
		ax2[0].set_ylabel(r'$f(x_k)$')
		ax2[0].set_title('Quasi-Newton method w/ BFGS & Armijio Cond. - Rosenbrock Test Function - n={}'.format(n))
		ax2[0].minorticks_on()
		ax2[0].grid(True, which='both')
		ax2[0].legend()
		ax2[0].set_yscale('log')

		ax2[1].plot(linalg.norm(g,axis=1))
		ax2[1].plot(sd_l2_g,label='SD method')
		ax2[1].set_ylabel(r'${\vert \vert \nabla f(x_k) \| \|}_2$')
		ax2[1].set_xlabel('Iteration Number, $k$')
		ax2[1].minorticks_on()
		ax2[1].grid(True, which='both')
		ax2[1].set_yscale('log')
		plt.show()


def steepest_descent(n,consts):

	x0 = random.uniform(-5,5,n)
	x = [x0]
	vals = [rosen(x0)]
	g = [rosen_gk(x0)]
	passConditions = []
	I = eye(len(x0))
	B_inv = [I]
	for iterNum in range(10000):
		x_k = x[-1]
		val_k = vals[-1]
		B_k_inv = B_inv[-1]
		g_k = g[-1]

		#if isclose(l2(g_k),e_g,atol=e_g,rtol=0):
		if not linalg.norm(g_k) > consts['e_g'] :
			print('l2 break - {} <= {}'.format(linalg.norm(g_k),consts['e_g']))
			break

		p_k = -g_k

		#Armijo sufficient decrease condition
		a_k = consts['a_k_init']
		t1 = rosen(x_k + a_k*p_k)
		t2 = rosen(x_k) + consts['u_1']*a_k*dot(g_k.T,p_k)
		#print('a_k: {} , x* = {} -> {} <= {} - diff: {}'.format(a_k,x_k + a_k*p_k,t1,t2, abs(t1-t2)))
		while rosen(x_k + a_k*p_k) > (rosen(x_k) + consts['u_1']*a_k*dot(g_k.T,p_k)):
		#while not (isclose(t1,t2,atol=e_a,rtol=e_r)):
			if a_k < 1e-6:
				print("Problem converging Armijo conditions")
			a_k *= consts['alpha_rho']

		x_k1 = x_k + a_k*p_k
		val_k1 = rosen(x_k1)
		g_k1 = rosen_gk(x_k1)

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
consts = {'e_g': 1e-4, 'e_a': 1e-8, 'e_r': 1e-5, 'u_1': 1e-4, 'alpha_rho': 0.5, 'a_k_init': 0.95, 'contourQuality': 500}

quasi_newton(n=2,consts=consts,plot=True,verbose=True)
quasi_newton(n=5,consts=consts,plot=True,verbose=True)