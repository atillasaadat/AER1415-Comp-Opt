#!/usr/bin/env python
""" 
AER1415 Computer Optimization - Assignment 3

Author: Atilla Saadat
Submitted: April 28, 2021
Email: asaadat@utias-sfl.net

Descripton: Question 3

"""

from numpy import *

#define KKT conditions and KKT matrix
G = array([[6,2,1],[2,5,2],[1,2,4]])
d = array([-1,-1,-1])
A = array([[1,0,1],[0,1,1]])
b = array([3,0])
#define full system of linear equations
AA = block([[G,-A.T],[A,zeros((2,2))]])
bb = block([-d,b])

#solve KKT conditions to get optimial solution
x_opt = linalg.solve(AA,bb)
print('x* = [{:.5f},{:.5f},{:.5f}], \u05BB* = [{:.5f},{:.5f}]'.format(*x_opt))