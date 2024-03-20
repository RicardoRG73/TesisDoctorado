#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:46:36 2024

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
color_map = "plasma"

#%%
# =============================================================================
# Load files
# =============================================================================
path = "figuras/CNDD/solN451.pkl"
Ua,ta,pa,DxPa,DyPa = pickle.load(open(path, "rb"))
Na = 451

path = "figuras/CNDD/solN976.pkl"
Ub,tb,pb,DxPb,DyPb = pickle.load(open(path, "rb"))
Nb = 976

#%%
# =============================================================================
# Psi, T, C. Different times
# =============================================================================
levelsP = 20
levelsT = 20
levelsC = 20

fig, axes = plt.subplots(3, 3, sharex="col", sharey="row", figsize=(13,10), constrained_layout=True)

ax1 = axes[0,0]
ax2 = axes[0,1]
ax3 = axes[1,0]
ax4 = axes[1,1]
ax5 = axes[2,0]
ax6 = axes[2,1]
ax7 = axes[0,2]
ax8 = axes[1,2]
ax9 = axes[2,2]

ax1.set_aspect("equal", "box")
ax2.set_aspect("equal", "box")
ax3.set_aspect("equal", "box")
ax4.set_aspect("equal", "box")
ax5.set_aspect("equal", "box")
ax6.set_aspect("equal", "box")
ax7.set_aspect("equal", "box")
ax8.set_aspect("equal", "box")
ax9.set_aspect("equal", "box")

lines = ax1.tricontourf(pb[:,0], pb[:,1], Ub[:Nb,0], cmap=color_map, levels=levelsP)
ax1.set_title("$\Psi$ at $t=%1.3f" %tb[0] + "$")
fig.colorbar(lines)

lines = ax2.tricontourf(pb[:,0], pb[:,1], Ub[Nb:2*Nb,0], cmap=color_map, levels=levelsT)
ax2.set_title("$T$ at $t=%1.3f" %tb[0] + "$")
fig.colorbar(lines)

lines = ax7.tricontourf(pb[:,0], pb[:,1], Ub[2*Nb:,0], cmap=color_map, levels=levelsC)
ax7.set_title("$C$ at $t=%1.3f" %tb[0] + "$")
fig.colorbar(lines)

t_index = tb.shape[0]//8
lines = ax3.tricontourf(pb[:,0], pb[:,1], Ub[:Nb,t_index], cmap=color_map, levels=levelsP)
# ax3.streamplot(pb[:,0], pb[:,1], DyP@Ub[:Nb,t_index], -DxP@Ub[:Nb,t_index])
ax3.set_title("$\Psi$ at $t=%1.3f" %tb[t_index] + "$")
fig.colorbar(lines)

lines = ax4.tricontourf(pb[:,0], pb[:,1], Ub[Nb:2*Nb,t_index], cmap=color_map, levels=levelsT)
ax4.set_title("$T$ at $t=%1.3f" %tb[t_index] + "$")
fig.colorbar(lines)

lines = ax8.tricontourf(pb[:,0], pb[:,1], Ub[2*Nb:,t_index], cmap=color_map, levels=levelsC)
ax8.set_title("$C$ at $t=%1.3f" %tb[t_index] + "$")
fig.colorbar(lines)

lines = ax5.tricontourf(pb[:,0], pb[:,1], Ub[:Nb,-1], cmap=color_map, levels=levelsP)
ax5.set_title("$\Psi$ at $t=%1.3f" %tb[-1] + "$")
fig.colorbar(lines)

lines = ax6.tricontourf(pb[:,0], pb[:,1], Ub[Nb:2*Nb,-1], cmap=color_map, levels=levelsT)
ax6.set_title("$T$ at $t=%1.3f" %tb[-1] + "$")
fig.colorbar(lines)

lines = ax9.tricontourf(pb[:,0], pb[:,1], Ub[2*Nb:,-1], cmap=color_map, levels=levelsC)
ax9.set_title("$C$ at $t=%1.3f" %tb[-1] + "$")
fig.colorbar(lines)

fig.suptitle(r"Solution with $N_{\text{nodes}}=%d$" %pb.shape[0], fontsize=20)

# =============================================================================
# C at t=0.05 using different N
# =============================================================================
levels = [0, 0.2, 0.6, 1]

fig, axes = plt.subplots(1,2,sharex="col",sharey="row", figsize=(9,4), constrained_layout=True)

t_wanted = 0.03
fig.suptitle("$C$ at $t=%1.2f$ for different N" %t_wanted, fontsize=20)

ax0 = axes[0]
ax1 = axes[1] 

ax0.set_aspect("equal", "box")
ax1.set_aspect("equal", "box")

t_index = np.argmin((ta - t_wanted)**2)

lines = ax0.tricontourf(pa[:,0], pa[:,1], Ua[2*Na:,t_index], cmap=color_map, levels=levels)
ax0.set_title("$N=%d$, $t=%1.2f$" %(Na,ta[t_index]))
fig.colorbar(lines)

t_index = np.argmin((tb - t_wanted)**2)

lines = ax1.tricontourf(pb[:,0], pb[:,1], Ub[2*Nb:,t_index], cmap=color_map, levels=levels)
ax1.set_title("$N=%d$, $t=%1.2f$" %(Nb,tb[t_index]))
fig.colorbar(lines)


plt.show()

# =============================================================================
# T, C, Vel. Stationary
# =============================================================================
# fig, axes = plt.subplots(3,1,sharex="col",sharey="row", figsize=(5,10), constrained_layout=True)

# fig.suptitle("Solution at $t=%1.2f$ using $N=%d$" %(tb[-1], Nb), fontsize=20)

# ax0 = axes[0]
# ax1 = axes[1]
# ax2 = axes[2]

# ax0.set_aspect("equal", "box")
# ax1.set_aspect("equal", "box")
# ax2.set_aspect("equal", "box")

# lines = ax0.tricontourf(pb[:,0], pb[:,1], Ub[Nb:2*Nb,-1], cmap=color_map, levels=levelsT)
# ax0.set_title("$T$ at $t=%1.2f" %tb[-1] + "$")
# fig.colorbar(lines)

# lines = ax1.tricontourf(pb[:,0], pb[:,1], Ub[2*Nb:,-1], cmap=color_map, levels=levelsC)
# ax1.set_title("$C$ at $t=%1.2f" %tb[-1] + "$")
# fig.colorbar(lines)

# vx = DyPb @ Ub[:Nb,4]
# vy = -DxPb @ Ub[:Nb,4]
# norm_v = np.sqrt(vx**2 + vy**2)
# lineas = ax2.tricontourf(pb[:,0], pb[:,1], norm_v, cmap=color_map, levels=levelsC)
# fig.colorbar(lineas)
# ax2.clear()
# from scipy.interpolate import griddata
# x = np.linspace(0, 1, 81)
# y = np.linspace(0, 0.9, 21)
# x,y = np.meshgrid(x,y)
# velx = griddata(pb, vx, (x,y))
# vely = griddata(pb, vy, (x,y))
# norm_vel = np.sqrt(velx**2 + vely**2)
# lineas = ax2.streamplot(x, y, velx, vely, color=norm_vel, cmap="plasma")
# ax2.set_title("Velocity")

