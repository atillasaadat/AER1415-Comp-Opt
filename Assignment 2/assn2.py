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
import matplotlib.colors as colors
import scipy.io

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

matplotlib.use('TkAgg')

def l2(x,ord=2):
	#return sum(abs(x)**ord, axis=0)**(1.0 / ord)
	return linalg.norm(x)

def P1(x,**kwargs):
	#x = asarray_chkfinite(x)
	x = array(x)
	x0 = x[:-1]
	x1 = x[1:]
	return sum(100.0*(x1 - x0**2.0)**2.0 + (1 - x0)**2.0)
	#return 100.0*(x[1] - x[0]**2)**2. + (1. - x[0])**2.

# P1
def runPS1(xSize,params):
	print('PS1:\n')	
	bounds = array([(-5,5)]*xSize)
	x0 = random.uniform(-5,5,xSize)
	return PSO(P1,x0,bounds,numParticles=max([200,xSize*10]),maxRepeat=10,maxIter=100,params=params).optimize(verbose=False)

#http://web.cse.ohio-state.edu/~parent.1/classes/788/Au10/OptimizationPapers/mathematicalOptimization.pdf
def ps1_gk(x):
	#https://www.math.purdue.edu/~wang838/notes/HW/CS20_HW.pdf
	#g_x = -400*(x[1]-x[0]**2)**x[0] - 2*(1-x[0])
	#g_x = 2*(200*x[0]**3 - 200*x[0]*x[1] + x[0] - 1)
	#https://www.wolframalpha.com/input/?i=d%2Fdy+100%28y%E2%88%92x%5E2%29%5E2%2B%281%E2%88%92x%29%5E2
	#d/dy -> 200*(x[1]-x[0]**2)
	if len(x) == 2:
		#g = [400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2.,
		#	-200*x[0]**2 + 400*x[1]**3 + 202*x[1] - 2.]
		#g = [-400*(x[1]-x[0]**2)*x[0] - 2*(2.-x[0]),
		#	200*(x[1]-x[0]**2)]
		g = [2*x[0] - 400*x[0]*(-x[0]**2 + x[1]) - 2, -200*x[0]**2 + 200*x[1]]
	#https://www.wolframalpha.com/input/?i=D%5B%28100%28b-a%5E2%29%5E2%2B%281-a%29%5E2%2B100%28x-b%5E2%29%5E2%2B%281-b%29%5E2%2B100%28y-x%5E2%29%5E2%2B%281-x%29%5E2%2B100%28z-y%5E2%29%5E2%2B%281-y%29%5E2%2B100%280-z%5E2%29%5E2%2B%281-z%29%5E2%29%2C+a%5D
	elif len(x) == 5:
		'''
		g = [400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2.,
			-200*x[0]**2 + 400*x[1]**3 + x[1]*(202 - 400*x[2]) - 2.,
			-200*x[1]**2 + 400*x[2]**3 + x[2]*(202 - 400*x[3]) - 2.,
			-200*x[2]**2 + 400*x[3]**3 + x[3]*(202 - 400*x[4]) - 2.,
			-200*x[3]**2 + 400*x[4]**3 + 202*x[4] - 2.]
		'''
		g = [2*(x[0]-1) - 400*x[0]*(x[1] - x[0]**2),
			 2*(x[1]-1) - 400*x[1]*(x[2] - x[1]**2) + 200*(x[1] - x[0]**2),
			 2*(x[2]-1) - 400*x[2]*(x[3] - x[2]**2) + 200*(x[2] - x[1]**2),
			 2*(x[3]-1) - 400*x[3]*(x[4] - x[3]**2) + 200*(x[3] - x[2]**2),
			 200*(x[4] - x[3]**2),
		]
	return array(g)

def ps1_Hk_n2(x):
	#https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/quadratic-approximations/a/the-hessian
	#https://www.wolframalpha.com/input/?i=d%2Fdx+2*%28200*x%5E3+-+200*x*y+%2B+x+-+1%29
	#https://www.wolframalpha.com/input/?i=d%2Fdy+2*%28200*x%5E3+-+200*x*y+%2B+x+-+1%29
	#https://www.wolframalpha.com/input/?i=d%2Fdx+200*%28y-x%5E2%29
	#https://www.wolframalpha.com/input/?i=d%2Fdy+200*%28y-x%5E2%29
	h_xx = 1200*x[0]**2 - 400*x[1] + 2.
	h_xy = -400*x[0]
	h_yx = -400*x[0]
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
	for iterNum in range(5):
		x_k = x[-1]
		g_k = ps1_gk(x_k)
		H_k = ps1_Hk_n2(x_k)

		#slides method
		p_k = -dot(linalg.inv(H_k),g_k)
		x_k1 = x_k + p_k
		val_k1 = fn(x_k1)

		x.append(x_k1)
		vals.append(val_k1)

	x = array(x)
	vals = array(vals)

	X, Y = meshgrid(linspace(-3, 3, 100), linspace(-3, 3, 100))
	Z = array([P1([j,i]) for i,j in zip(X.flatten(),Y.flatten())]).reshape(X.shape)
	fig, ax = plt.subplots()
	#pcm = ax.pcolormesh(X, Y, Z,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=-1.0, vmax=1.0),cmap='RdBu_r')
	ax.plot(*x.T,'y-o')
	#pcm = ax.pcolormesh(X, Y, Z, cmap='RdBu_r')
	#fig.colorbar(pcm, ax=ax, extend='both')
	#ax.contourf(X, Y, Z)
	plt.contour(X, Y, Z, 20, cmap='RdGy');

	for idx,x_k in enumerate(x):
		print('x*: {}, val:: {}'.format(x_k,vals[idx]))
	plt.show()


def quasi_newton(fn):
	#https://www.wolframalpha.com/input/?i=d%2Fdx+100%28y%E2%88%92x%5E2%29%5E2%2B%281%E2%88%92x%29%5E2
	#d/dx -> 2*(200*x1**3 - 200*x1*x2 + x1 - 1)
	n = 2
	#x0 = array([2,3])
	x0 = random.uniform(-5,5,n)
	e_g = 1e-4
	e_a = 1e-8
	e_r = 1e-5
	u_1 = 1e-4
	alpha_rho = 0.5
	a_k_init = 0.95
	x = [x0]
	vals = [fn(x0)]
	g = [ps1_gk(x0)]
	passConditions = []
	I = eye(len(x0))
	B_inv = [I]
	for iterNum in range(10000):
		x_k = x[-1]
		val_k = vals[-1]
		B_k_inv = B_inv[-1]
		g_k = g[-1]

		if isclose(l2(g_k),e_g,atol=e_g,rtol=0):
			print('l2 break - {} <= {}'.format(l2(g_k),e_g))
			break

		#p_k = -dot(B_k_inv,g_k)
		p_k = -linalg.solve(B_k_inv,g_k)
		#p_k = -B_k_inv.T.dot(g_k)
		#p_k = -dot(B_k,g_k)

		#Armijo sufficient decrease condition
		#'''
		a_k = a_k_init
		t1 = fn(x_k + a_k*p_k)
		t2 = fn(x_k) + u_1*a_k*dot(g_k.T,p_k)
		#g1 = abs(dot(ps1_gk(x_k1).T,p_k))
		#g2 = u_2*abs(dot(g_k.T,p_k))
		print('a_k: {} , x* = {} -> {} <= {} - diff: {}'.format(a_k,x_k + a_k*p_k,t1,t2, abs(t1-t2)))
		while fn(x_k + a_k*p_k) > (fn(x_k) + u_1*a_k*dot(g_k.T,p_k)):
		#while not (isclose(t1,t2,atol=e_a,rtol=e_r)):
			if isnan(t1) or isnan(t2):
				print('is nan')
				embed()
			if a_k < 1e-5:
				print("issue with Armijo")
				embed()
			a_k *= alpha_rho
			print('a_k: {} , x* = {} -> {} <= {} - diff: {}'.format(a_k,x_k + a_k*p_k,t1,t2, abs(t1-t2)))
			#g1 = abs(dot(ps1_gk(x_k1).T,p_k))
			#g2 = u_2*abs(dot(g_k.T,p_k))
			#print('{} - {} <= {}'.format(a_k,t1,t2))
		#'''
		x_k1 = x_k + a_k*p_k

		val_k1 = fn(x_k1)

		#numpy isclose: absolute(a - b) <= (atol + rtol * absolute(b))
		#passCond = isclose(val_k1,val_k,atol=e_a,rtol=e_r)
		passCond = abs(val_k1 - val_k)<=(e_a + e_r*abs(val_k))
		#if passCond:
		#	embed()
		print('{} + {}*{}'.format(x_k,a_k,p_k))
		print('{} passCon: {} - {} <= {}'.format(iterNum,passCond,abs(val_k1 - val_k),(e_a + e_r*abs(val_k))))
		passConditions.append(passCond)
		#if passCond:
		#	embed()

		if len(passConditions) >= 2 and all(passConditions[-2:]):
			print('passCond break - {}'.format(passConditions))
			break

		#BFGS
		method = 1 

		g_k1 = ps1_gk(x_k1)
		s_k = x_k1 - x_k
		y_k = g_k1 - g_k
		x_len = x_k.shape[0]
		if method == 1:
			y_t = y_k.reshape([x_len, 1])
			Bk = dot(B_k_inv, s_k)
			k_t_B = dot(s_k, B_k_inv)
			kBk = dot(dot(s_k, B_k_inv), s_k)
					 # Update B positive definite matrix. Calculate exactly according to the formula
			B_k1_inv = B_k_inv + y_t*y_k/dot(y_k, s_k) - Bk.reshape([x_len, 1]) * k_t_B / kBk

		elif method == 2:
			rhok_inv = dot(s_k.T,y_k)
			# this was handled in numeric, let it remaines for more safety
			if rhok_inv == 0.:
				rhok = 1000.0
				print("***Divide-by-zero encountered: rhok assumed large****")
				#embed()
			else:
				rhok = 1. / rhok_inv
			A1 = I - (dot(s_k,y_k.T)*rhok)
			#A1 = I - dot(s_k,y_k.T)*rhok
			A2 = I - (dot(y_k,s_k.T)*rhok)
			#A2 = I - dot(y_k,s_k.T)*rhok
			#B_k1 = (I - dot(s_k,y_k.T)/dot(s_k.T,y_k)) * B_k * (I - dot(y_k,s_k.T)/dot(s_k.T,y_k)) + dot(s_k,s_k.T)/dot(s_k.T,y_k)
			B_k1_inv = dot(dot(A1,B_k_inv),A2) + dot(s_k,s_k.T)*rhok
			#B_k1 = dot() + s_k*s_k*rhok
		elif method == 3:
			s_t = s_k.reshape([x_len, 1])
			y_t = y_k.reshape([x_len, 1])
			den = dot(s_k,y_k)
			A1 = I - s_k*y_t/den
			A2 = I - y_k*s_t/den
			A3 = s_k*s_t/den
			B_k1_inv = A1*B_k_inv*A2 + A3
			#embed()
		elif method == 4:
			#s_t = s_k.reshape([x_len, 1])
			s_t = s_k.T
			#y_t = y_k.reshape([x_len, 1])
			y_t = y_k.T
			B_k1_inv = B_k_inv + ((s_t*y_k + y_t*B_k_inv*y_k)*(s_k*s_t))/(s_t*y_k)**2 - (B_k_inv*y_k*s_t + s_k*y_t*B_k_inv)/(s_t*y_k)


		#embed()
		#'''

		#B_k1 = B_k - divide(B_k*s_k*s_k.T*B_k,s_k.T*B_k*s_k) + divide(y_k*y_k.T,y_k.T*s_k)
		

		#if nan B_k1_inv then break 
		if any(isnan(B_k1_inv)):
			print('B_k1_inv is nan')
			break#embed()
		
		B_inv.append(B_k1_inv)
		print(x_k1,val_k1)
		x.append(x_k1)
		vals.append(val_k1)
		g.append(g_k1)
	x = array(x)
	vals = array(vals)	
	X, Y = meshgrid(linspace(-5, 5, 100), linspace(-5, 5, 100))
	Z = array([P1([j,i]) for i,j in zip(X.flatten(),Y.flatten())]).reshape(X.shape)
	fig, ax = plt.subplots()
	#pcm = ax.pcolormesh(X, Y, Z,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=-1.0, vmax=1.0),cmap='RdBu_r')
	ax.plot(*x.T,'y-o')
	#pcm = ax.pcolormesh(X, Y, Z, cmap='RdBu_r')
	#fig.colorbar(pcm, ax=ax, extend='both')
	#ax.contourf(X, Y, Z)
	plt.contour(X, Y, Z, 20, cmap='RdGy')
	
	for idx,x_k in enumerate(x):
		print('x*: {}, val:: {}'.format(x_k,vals[idx]))
	embed()
	plt.show()	

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

def Q3():
	#BB - steepest descent
	x0 = random.uniform(-5,5,n)
	e_r = 1e-8
	e_g = 1e-5
	e_a = 1e-8
	mat = scipy.io.loadmat('ConvexQuadratic.mat')
	A = mat['A'].toarray()
	b = mat['b'].flatten()
	x = [ones(len(b))]
	fn = lambda i: 0.5*dot(dot(i.T,A),x)-dot(i.T,b)
	val_k = fn(x[0])
	#for iterNum in range(100):
				


#normal_newton(P1)
quasi_newton(P1)