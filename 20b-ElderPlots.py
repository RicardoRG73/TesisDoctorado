#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:03:27 2024

@author: ricardo
"""
save_figures = True
#%%
# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import pickle
import matplotlib.pyplot as plt

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-talk"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = False
plt.rcParams["legend.framealpha"] = 0.1
mapa_de_color = "plasma"

#%%
# =============================================================================
# Load files
# a) N = 533
# b) N = 1430
# c) N = 5453
# d) N = 6820
# =============================================================================
path = "figuras/Elder/solN533.pkl"
Ua,ta,pa,DxPa,DyPa = pickle.load(open(path, "rb"))
Na = 533

path = "figuras/Elder/solN1430.pkl"
Ub,tb,pb,DxPb,DyPb = pickle.load(open(path, "rb"))
Nb = 1430

path = "figuras/Elder/solN5453.pkl"
Uc,tc,pc,DxPc,DyPc = pickle.load(open(path, "rb"))
Nc = 5453

path = "figuras/Elder/solN6820.pkl"
Ud,td,pd,DxPd,DyPd = pickle.load(open(path, "rb"))
Nd = 6820

#%%
# =============================================================================
# Psi and C at different times, N=6820
# =============================================================================
levelsP = 16
levelsC = 9

fig, axes = plt.subplots(5, 2, sharex="col", sharey="row", figsize=(9,8), constrained_layout=True)

fig.suptitle("Solution at different times with $N=%d$" %Nd, fontsize=20)

ax1 = axes[0,0]
ax2 = axes[0,1]
ax3 = axes[1,0]
ax4 = axes[1,1]
ax5 = axes[2,0]
ax6 = axes[2,1]
ax7 = axes[3,0]
ax8 = axes[3,1]
ax9 = axes[4,0]
ax10 = axes[4,1]

ax1.set_aspect("equal", "box")
ax2.set_aspect("equal", "box")
ax3.set_aspect("equal", "box")
ax4.set_aspect("equal", "box")
ax5.set_aspect("equal", "box")
ax6.set_aspect("equal", "box")
ax7.set_aspect("equal", "box")
ax8.set_aspect("equal", "box")
ax9.set_aspect("equal", "box")
ax10.set_aspect("equal", "box")

ax1.tricontourf(pd[:,0], pd[:,1], Ud[:Nd,0], cmap=mapa_de_color, levels=levelsP)
ax1.set_title("$\Psi$ at $t=%1.3f$" %td[0])

ax2.tricontourf(pd[:,0], pd[:,1], Ud[Nd:,0], cmap=mapa_de_color, levels=levelsC)
ax2.set_title("$C$ at $t=%1.3f$" %td[0])

ax3.tricontourf(pd[:,0], pd[:,1], Ud[:Nd,1], cmap=mapa_de_color, levels=levelsP)
ax3.set_title("$\Psi$ at $t=%1.3f$" %td[1])

ax4.tricontourf(pd[:,0], pd[:,1], Ud[Nd:,1], cmap=mapa_de_color, levels=levelsC)
ax4.set_title("$C$ at $t=%1.3f$" %td[1])

ax5.tricontourf(pd[:,0], pd[:,1], Ud[:Nd,2], cmap=mapa_de_color, levels=levelsP)
ax5.set_title("$\Psi$ at $t=%1.3f$" %td[2])

ax6.tricontourf(pd[:,0], pd[:,1], Ud[Nd:,2], cmap=mapa_de_color, levels=levelsC)
ax6.set_title("$C$ at $t=%1.3f$" %td[2])

ax7.tricontourf(pd[:,0], pd[:,1], Ud[:Nd,3], cmap=mapa_de_color, levels=levelsP)
ax7.set_title("$\Psi$ at $t=%1.3f$" %td[3])

ax8.tricontourf(pd[:,0], pd[:,1], Ud[Nd:,3], cmap=mapa_de_color, levels=levelsC)
ax8.set_title("$C$ at $t=%1.3f$" %td[3])

ax9.tricontourf(pd[:,0], pd[:,1], Ud[:Nd,4], cmap=mapa_de_color, levels=levelsP)
ax9.set_title("$\Psi$ at $t=%1.3f$" %td[4])

ax10.tricontourf(pd[:,0], pd[:,1], Ud[Nd:,4], cmap=mapa_de_color, levels=levelsC)
ax10.set_title("$C$ at $t=%1.3f$" %td[4])

if save_figures:
    plt.savefig("figuras/Elder/Psi_C_diff_t.pdf")

# =============================================================================
# C at t=0.05 using different N
# =============================================================================
levels = [0, 0.2, 0.6, 1]

fig, axes = plt.subplots(2,2,sharex="col",sharey="row", figsize=(13,4), constrained_layout=True)

fig.suptitle("$C$ at $t=0.05$ for different N", fontsize=20)

ax0 = axes[0,0]
ax1 = axes[0,1]
ax2 = axes[1,0]
ax3 = axes[1,1]

ax0.set_aspect("equal", "box")
ax1.set_aspect("equal", "box")
ax2.set_aspect("equal", "box")
ax3.set_aspect("equal", "box")

cont = ax0.tricontourf(pa[:,0], pa[:,1], Ua[Na:,2], cmap=mapa_de_color, levels=levels)
ax0.set_title("$N=%d$" %Na)
fig.colorbar(cont)

cont = ax1.tricontourf(pb[:,0], pb[:,1], Ub[Nb:,2], cmap=mapa_de_color, levels=levels)
ax1.set_title("$N=%d$" %Nb)
fig.colorbar(cont)

cont = ax2.tricontourf(pc[:,0], pc[:,1], Uc[Nc:,2], cmap=mapa_de_color, levels=levels)
ax2.set_title("$N=%d$" %Nc)
fig.colorbar(cont)

cont = ax3.tricontourf(pd[:,0], pd[:,1], Ud[Nd:,2], cmap=mapa_de_color, levels=levels)
ax3.set_title("$N=%d$" %Nd)
fig.colorbar(cont)

if save_figures:
    plt.savefig("figuras/Elder/C_diff_N.pdf")

#%%
# =============================================================================
# Psi, C, vel, stationary t=1.239, N=6820
# =============================================================================
fig, axes = plt.subplots(3,1,sharex="col",sharey="row", figsize=(9,8),constrained_layout=True)

fig.suptitle("Solution at $t=%1.3f$ using $N=%d$" %(td[4], Nd), fontsize=20)

ax0 = axes[0]
ax1 = axes[1]
ax2 = axes[2]

ax0.set_aspect("equal", "box")
ax1.set_aspect("equal", "box")
ax2.set_aspect("equal", "box")

cont = ax0.tricontourf(pd[:,0], pd[:,1], Ud[:Nd,4], cmap=mapa_de_color, levels=levelsP)
fig.colorbar(cont)
ax0.set_title("$\Psi$")

cont = ax1.tricontourf(pd[:,0], pd[:,1], Ud[Nd:,4], cmap=mapa_de_color, levels=levelsC)
fig.colorbar(cont)
ax1.set_title("$C$")

vx = DyPd @ Ud[:Nd,4]
vy = -DxPd @ Ud[:Nd,4]
norm_v = np.sqrt(vx**2 + vy**2)
cont = ax2.tricontourf(pd[:,0], pd[:,1], norm_v, cmap=mapa_de_color, levels=levelsC)
fig.colorbar(cont)
ax2.clear()
# ax2.quiver(pd[:,0], pd[:,1], vx, vy, alpha=0.5)
from scipy.interpolate import griddata
x = np.linspace(0, 4, 81)
y = np.linspace(0, 1, 21)
x,y = np.meshgrid(x,y)
velx = griddata(pd, vx, (x,y))
vely = griddata(pd, vy, (x,y))
norm_vel = np.sqrt(velx**2 + vely**2)
cont = ax2.streamplot(x, y, velx, vely, color=norm_vel, cmap="plasma")
ax2.set_title("Velocity")

if save_figures:
    plt.savefig("figuras/Elder/stationary.pdf")
