#!/usr/bin/env python
""" 
AER1415 Computer Optimization - Assignment 3

Author: Atilla Saadat
Submitted: April 28, 2021
Email: asaadat@utias-sfl.net

Descripton: Question 2

"""

from numpy import *
import os
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

#define analytical solution to plot 
x_opt = [0.5,0.5,0.75]

#define objective function and limits
vRange=1.5
x = arange(-vRange,vRange+.1,0.1)
y = arange(-vRange,vRange+.1,0.1)
def fn(x1,x2):
	return (x1**3 + x2**3) #+ (1.-x1-x2)

#plot contour
X,Y = meshgrid(x,y)
Z = fn(X,Y)
contours = plt.contour(X, Y, Z, 75, colors='black')
plt.clabel(contours, inline=True, fontsize=8)

plt.imshow(Z, extent=[-vRange, vRange, -vRange, vRange], origin='lower',
           cmap='viridis', alpha=0.5)
cbar = plt.colorbar()
cbar.set_label(r'$f(x)$')
plt.xlim([-vRange,vRange])
plt.ylim([-vRange,vRange])
plt.plot(x,1.-x,label=r'$x_1 + x_2 = 1$ constraint')
plt.scatter(x_opt[0],x_opt[1],label=r'$[x^\star,\lambda^\star]$ @ [{:.2f},{:.2f},{:.2f}]'.format(*x_opt),c='r')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Q2 KKT Solution - $f(x) = x_1^3 + x_2^3$')
plt.legend()
plt.minorticks_on()
plt.grid(True, which='both')
plt.show()
