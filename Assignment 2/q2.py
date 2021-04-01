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

from rosen import *

def quasi_newton(n,consts,plot=False,verbose=False):
	#Initialize parameters
	x0 = random.uniform(-5,5,n)
	x = [x0]
	vals = [rosen(x0)]
	g = [rosen_gk(x0)]
	passConditions = []
	I = eye(len(x0))
	B_inv = [I]
	for iterNum in range(1000):
		#Get current iterate values
		x_k = x[-1]
		val_k = vals[-1]
		B_k_inv = B_inv[-1]
		g_k = g[-1]

		#l2-norm break criterion
		if not linalg.norm(g_k) > consts['e_g'] :
			break

		#solve for search direction from BFGS hessian approximate
		p_k = -linalg.solve(B_k_inv,g_k)

		#Armijo sufficient decrease condition
		a_k = consts['a_k_init']
		t1 = rosen(x_k + a_k*p_k)
		t2 = rosen(x_k) + consts['u_1']*a_k*dot(g_k.T,p_k)
		while rosen(x_k + a_k*p_k) > (rosen(x_k) + consts['u_1']*a_k*dot(g_k.T,p_k)):
			a_k *= consts['alpha_rho']

		#after armijios condition passes, multiply a_k
		x_k1 = x_k + a_k*p_k
		val_k1 = rosen(x_k1)

		#check pass condition
		#numpy isclose: absolute(a - b) <= (atol + rtol * absolute(b))
		passCond = isclose(val_k1,val_k,atol=consts['e_a'],rtol=consts['e_r'])
		passConditions.append(passCond)
		if len(passConditions) >= 2 and all(passConditions[-2:]):
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
	
	#Plot contour, convergence and gradient vector plots
	if plot:
		if n==2:
			print('Creating Contour Plot - Please Wait')
			xx = linspace(min(min(x.T[0]),-5), max(max(x.T[0]),5), consts['contourQuality'])
			yy = linspace(min(min(x.T[1]),-5), max(max(x.T[1]),5), consts['contourQuality'])
			X, Y = meshgrid(xx,yy)
			Z = array([rosen([j,i]) for i,j in zip(X.flatten(),Y.flatten())]).reshape(X.shape)
			fig, ax = plt.subplots()
			fig.set_size_inches(18.5, 10.5)
			pcm = ax.pcolormesh(X, Y, Z,norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),cmap='RdBu_r',rasterized=True)
			ax.plot(*x.T,'-o',color='#16F41B',markersize=2)
			ax.set_xlabel(r'$x_1$')
			ax.set_ylabel(r'$x_2$')
			ax.set_title('Quasi-Newton method w/ BFGS & Armijio Cond. - Rosenbrock Test Function - n={}'.format(n))
			fig.colorbar(pcm, ax=ax, extend='max',label=r'$f(x)$')
			plt.savefig('Q2_QuasiNewtonContour.svg', format = 'svg', bbox_inches = 'tight')
			print('Contour Plot Done!')

		sd_x, sd_vals, sd_l2_g = steepest_descent(n,consts)

		fig2, ax2 = plt.subplots(2,1)
		fig2.set_size_inches(18.5, 10.5)
		ax2[0].plot(vals, label=r'QN Last Iter: {} - $x^\star$: {}, $f(x_k)$: {:.6f}'.format(len(vals),around(x[-1],6),vals[-1]))
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
		plt.savefig('Q2_QuasiNewtonPlot_n{}.svg'.format(n),dpi=1000, format = 'svg', bbox_inches = 'tight')
		plt.show()
	print('Quasi-Newton: iter: {}, x*: {}, val: {}'.format(len(x),x[-1],vals[-1]))
	return [array(x),array(vals),linalg.norm(g,axis=1)]

#steepest descent algorithm for comparision
def steepest_descent(n,consts):
	#Initialize parameters
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

		#l2-norm break criterion
		if not linalg.norm(g_k) > consts['e_g'] :
			break

		#steepest gradient descent search direction
		p_k = -g_k

		#Armijo sufficient decrease condition
		a_k = consts['a_k_init']
		t1 = rosen(x_k + a_k*p_k)
		t2 = rosen(x_k) + consts['u_1']*a_k*dot(g_k.T,p_k)
		while rosen(x_k + a_k*p_k) > (rosen(x_k) + consts['u_1']*a_k*dot(g_k.T,p_k)):
			a_k *= consts['alpha_rho']
		
		#after armijios condition passes, multiply a_k
		x_k1 = x_k + a_k*p_k
		val_k1 = rosen(x_k1)
		g_k1 = rosen_gk(x_k1)

		#check pass condition
		#numpy isclose: absolute(a - b) <= (atol + rtol * absolute(b))
		passCond = isclose(val_k1,val_k,atol=consts['e_a'],rtol=consts['e_r'])
		if len(passConditions) >= 2 and all(passConditions[-2:]):
			break
		
		passConditions.append(passCond)
		x.append(x_k1)
		vals.append(val_k1)
		g.append(g_k1)
	print('Steepest-Descent:  iter: {}, x*: {}, val: {}'.format(len(x),x[-1],vals[-1]))
	return [array(x),array(vals),linalg.norm(g,axis=1)]


'''
	- e_r : relative tolerance
	- e_a : absolute tolerance
	- e_g : absolute gradient tolerance
'''
consts = {2: {'e_g': 1e-5, 'e_a': 1e-8, 'e_r': 1e-5, 'u_1': 1e-4, 'alpha_rho': 0.5, 'a_k_init': 0.95, 'contourQuality': 500},
		  5: {'e_g': 1e-7, 'e_a': 1e-10, 'e_r': 1e-7, 'u_1': 1e-4, 'alpha_rho': 0.5, 'a_k_init': 0.95, 'contourQuality': 500},	}

#20 runs of n=2 and n=5
for n in [2,5]:
	qn_x = []
	qn_vals = []
	qn_iters = []
	sd_x = []
	sd_vals = []
	sd_iters = []
	for i in range(20):
		print('\nRound {} / {}'.format(i+1,20))
		qn = quasi_newton(n,consts[n])
		sd = steepest_descent(n,consts[n])
		qn_x.append(qn[0][-1])
		qn_vals.append(qn[1][-1])
		qn_iters.append(len(qn[0]))
		sd_x.append(sd[0][-1])
		sd_vals.append(sd[1][-1])
		sd_iters.append(len(sd[0]))
	print('\nN={}\n{}'.format(n,'-'*50))
	print('QN Avgs - iter: {}, x*: {}, val: {}'.format(round(mean(qn_iters)),mean(qn_x,axis=0),mean(qn_vals)))
	print('SD Avgs - iter: {}, x*: {}, val: {}'.format(round(mean(sd_iters)),mean(sd_x,axis=0),mean(sd_vals)))


quasi_newton(n=2,consts=consts,plot=True,verbose=True)
quasi_newton(n=5,consts=consts,plot=True,verbose=True)