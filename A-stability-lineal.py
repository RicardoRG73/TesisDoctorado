#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:23:58 2024

@author: ricardo
"""
# libraries
import numpy as np
import matplotlib.pyplot as plt

# complex plane discretization
N = 301
limits_x = [-3, 1]
limits_y = [-2, 2]
x = np.linspace(limits_x[0], limits_x[1], N)
y = np.linspace(limits_y[0], limits_y[1], N)
x,y = np.meshgrid(x,y)

# condition Euler |1+z| <= 1
condition = np.sqrt( (1 + x)**2 + y**2 )
condition = condition <= 1
condition = condition * 1

# figure
plt.figure()
plt.contourf(x,y,condition, levels=[-1,0,1], cmap="Blues")
plt.axhline(0, color="k")
plt.axvline(0, color="k")
plt.colorbar()
plt.xlim(limits_x)
plt.ylim(limits_y)
plt.axis("equal")
plt.show()