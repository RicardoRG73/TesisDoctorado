#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:43:33 2024

@author: ricardo
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-talk"])

# =============================================================================
# La ecuacion diferencial es
# y' = y * (1-y)
# =============================================================================

x = np.linspace(-3,4,31)
y = np.linspace(-3,4,31)
x,y = np.meshgrid(x,y)

u = np.ones(x.shape)
v = y * (1-y)

plt.figure()
plt.contourf(x,y,v, cmap="Blues", levels=21)
plt.streamplot(x,y,u,v, color="k")
plt.axis("equal")

# =============================================================================
# Otra ecuacion
# Considerando los circulos
# x**2 + y**2 = r ** 2
# donde el radio r es constante.
# Derivando respecto a x
# 2*x + 2*y*y' = 0
# despejando y'
# y' = - x / y
# =============================================================================
x = np.linspace(-5,5,31)
y = x.copy()
x,y = np.meshgrid(x,y)

u = y
v = -x

plt.figure()
plt.contourf(x,y,x**2 + y**2, cmap="Blues", levels=11)
plt.streamplot(x,y,u,v, color="k")
plt.axis("equal")

# =============================================================================
# y' = 1 + y^2
# =============================================================================
x = np.linspace(-5,5,101)
y = x.copy()
x,y = np.meshgrid(x,y)
u = np.ones(x.shape)
v = 1 + y**2
plt.figure()
plt.contourf(x,y,v, cmap="Blues", levels=21)
plt.streamplot(x,y,u,v, color="k")
plt.axis("equal")

plt.show()