#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:23:58 2024

@author: ricardo
"""
# libraries
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Forward Euler
# =============================================================================
# complex plane discretization
N = 301
limits_x = [-3, 1]
limits_y = [-2, 2]
x = np.linspace(limits_x[0], limits_x[1], N)
y = np.linspace(limits_y[0], limits_y[1], N)
x,y = np.meshgrid(x,y)

z = x + 1j*y    # complex plane

# condition |1+z| <= 1
condition = np.abs(1+z)
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
plt.title("Forward Euler")
plt.show()


# =============================================================================
# Backward Euler
# =============================================================================
# complex plane discretization
N = 301
limits_x = [-1, 3]
limits_y = [-2, 2]
x = np.linspace(limits_x[0], limits_x[1], N)
y = np.linspace(limits_y[0], limits_y[1], N)
x,y = np.meshgrid(x,y)

z = x + 1j*y    # complex plane

# condition |1/(1-z)| <= 1
condition = np.abs( 1 / (1-z) )
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
plt.title("Backward Euler")
plt.show()


# =============================================================================
# Crank-Nicolson
# =============================================================================
# complex plane discretization
N = 301
limits_x = [-2, 2]
limits_y = [-2, 2]
x = np.linspace(limits_x[0], limits_x[1], N)
y = np.linspace(limits_y[0], limits_y[1], N)
x,y = np.meshgrid(x,y)

z = x + 1j*y    # complex plane

# condition | (1 + z/2) / (1 - z/2) | <= 1
condition = np.abs( (1 + z/2) / (1 - z/2) )
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
plt.title("Crank-Nicolson")
plt.show()