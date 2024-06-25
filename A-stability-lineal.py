#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:23:58 2024

@author: ricardo
"""
save_figures = False
# libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(["seaborn-v0_8"])

# =============================================================================
# Forward Euler
# =============================================================================
# complex plane discretization
N = 301
limits_x = [-3.5, 1.5]
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
if save_figures:
    plt.savefig("figuras/A-Stability-FE.pdf")
plt.show()


# =============================================================================
# Backward Euler
# =============================================================================
# complex plane discretization
N = 301
limits_x = [-1.5, 3.5]
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
if save_figures:
    plt.savefig("figuras/A-Stability-BE.pdf")
plt.show()


# =============================================================================
# Crank-Nicolson
# =============================================================================
# complex plane discretization
N = 301
limits_x = [-2.5, 2.5]
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
if save_figures:
    plt.savefig("figuras/A-Stability-CN.pdf")
plt.show()


# =============================================================================
# RK2
# =============================================================================
# complex plane discretization
N = 301
limits_x = [-3.5, 1.5]
limits_y = [-2, 2]
x = np.linspace(limits_x[0], limits_x[1], N)
y = np.linspace(limits_y[0], limits_y[1], N)
x,y = np.meshgrid(x,y)

z = x + 1j*y    # complex plane

# condition
condition = 1 + z + (z**2)/2
condition = np.abs(condition) <= 1
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
plt.title("Runge-Kutta 2")
if save_figures:
    plt.savefig("figuras/A-Stability-RK2.pdf")
plt.show()


# =============================================================================
# RKF45
# =============================================================================
# complex plane discretization
N = 1001
limits_x = [-7, 5.5]
limits_y = [-4, 4]
x = np.linspace(limits_x[0], limits_x[1], N)
y = np.linspace(limits_y[0], limits_y[1], N)
x,y = np.meshgrid(x,y)

z = x + 1j*y    # complex plane

# condition
# condition = z * ( 0.00048*z**5 - 0.00128*z**4 )

# RK5
cond_RK5 = np.array([
    0.000480769230769231*z**6,
    0.00833333333333333*z**5,
    0.0416666666666667*z**4,
    0.166666666666667*z**3,
    0.5*z**2,
    z,
    0*z + 1
])
cond_RK5 = np.sum(cond_RK5, axis=0)
cond_RK5 = np.abs(cond_RK5) <= 1
cond_RK5 = cond_RK5 * 1

# RK4
cond_RK4 = np.array([
    0.00961538461538462*z**5,
    0.0416666666666667*z**4,
    0.166666666666667*z**3,
    0.5*z**2,
    z,
    0*z + 1
])
cond_RK4 = np.sum(cond_RK4, axis=0)
cond_RK4 = np.abs(cond_RK4) <= 1
cond_RK4 = cond_RK4 * 1

# figure
plt.figure()
contrk5 = plt.contourf(x,y,cond_RK5, levels=[-1,0,1], alpha=0.5,cmap="Reds")
contrk4 = plt.contourf(x,y,cond_RK4, levels=[-1,0,1], alpha=0.5, cmap="Blues")

coords = contrk5.allsegs[1][0]
plt.plot(coords[:,0],coords[:,1], color="red", label="RK5")
coords = contrk4.allsegs[1][0]
plt.plot(coords[:,0],coords[:,1], color="blue", label="RK4")

# plt.contourf(x,y,cond_RK4*cond_RK5, levels=[-1,0,1], alpha=0.5, cmap="Greys")


plt.axhline(0, color="k")
plt.axvline(0, color="k")
plt.xlim(limits_x)
plt.ylim(limits_y)
plt.axis("equal")
plt.title("Runge-Kutta-Fehlberg")
plt.legend()
if save_figures:
    plt.savefig("figuras/A-Stability-RKF45.pdf")
plt.show()