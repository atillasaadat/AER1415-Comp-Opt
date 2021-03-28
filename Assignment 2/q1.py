#!/usr/bin/env python
"""
AER1415 Computer Optimization - Assignment 2

Author: Atilla Saadat
Submitted: Mar 31, 2021
Email: asaadat@utias-sfl.net

Question 1

"""

from numpy import *
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.colors as colors
'''
import os
from IPython import embed
import cmath
import scipy.io
'''

from rosen import *


def newtons_method(plot=False,verbose=False):
	#https://www.wolframalpha.com/input/?i=d%2Fdx+100%28y%E2%88%92x%5E2%29%5E2%2B%281%E2%88%92x%29%5E2
	#d/dx -> 2*(200*x1**3 - 200*x1*x2 + x1 - 1)
	x0 = array([-1., -2.])
	x = [x0]
	vals = [rosen(x0)]
	for iterNum in range(5):
		x_k = x[-1]
		g_k = rosen_gk(x_k)
		H_k = rosen_Hk_n2(x_k)

		#slides method
		p_k = -dot(linalg.inv(H_k),g_k)
		x_k1 = x_k + p_k
		val_k1 = rosen(x_k1)

		x.append(x_k1)
		vals.append(val_k1)

	x = array(x)
	vals = array(vals)

	if verbose:
		for idx,x_k in enumerate(x):
			print('Iter: {} - x*: {}, val:: {}'.format(idx,x_k,vals[idx]))
	
	if plot:
		print('Creating Contour Plot - Please Wait')
		X, Y = meshgrid(linspace(-1.5, 2, 500), linspace(-3, 3, 1000))
		Z = array([rosen([j,i]) for i,j in zip(X.flatten(),Y.flatten())]).reshape(X.shape)
		fig, ax = plt.subplots()
		pcm = ax.pcolormesh(X, Y, Z,norm=colors.SymLogNorm(linthresh=0.03,vmin=Z.min(), vmax=Z.max()),cmap='RdBu_r')
		ax.plot(*x.T,'-o',color='#16F41B',markersize=2)
		ax.set_xlabel(r'$x_1$')
		ax.set_ylabel(r'$x_2$')
		ax.set_title('Newtons method - Rosenbrock Test Function - n=2')
		#pcm = ax.pcolormesh(X, Y, Z, cmap='RdBu_r')
		fig.colorbar(pcm, ax=ax, extend='both')
		#ax.contourf(X, Y, Z)
		#plt.contour(X, Y, Z, 20, cmap='RdGy');
		print('Contour Plot Done!')
		plt.show()

	return {'x*': x[-1], 'val': vals[-1]}

newtons_method(plot=True,verbose=True)
