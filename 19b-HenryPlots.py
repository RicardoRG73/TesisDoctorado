#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:09:55 2024

@author: ricardo
"""
# =============================================================================
# Libraries
# =============================================================================
import pickle as pk
import matplotlib.pyplot as plt

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-talk"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1
mapa_de_color = "plasma"

# =============================================================================
# Load files
# a) N = 274
# b) N = 998
# c) N = 5969
# =============================================================================

path = "figuras/Henry/solN274.pkl"
Ua,ta,pa = pk.load(open(path, "rb"))
Na = pa.shape[0]

path = "figuras/Henry/solN998.pkl"
Ub,tb,pb = pk.load(open(path, "rb"))
Nb = pb.shape[0]

path = "figuras/Henry/solN5969.pkl"
Uc,tc,pc = pk.load(open(path, "rb"))
Nc = pc.shape[0]

# test plot
# fig, axes = plt.subplots(1,3,sharey="row")

# ax0 = axes[0]
# ax1 = axes[1]
# ax2 = axes[2]

# ax0.set_aspect("equal", "box")
# ax1.set_aspect("equal", "box")
# ax2.set_aspect("equal", "box")

# ax0.tricontourf(pa[:,0], pa[:,1], Ua[Na:,-1], cmap=mapa_de_color)
# ax1.tricontourf(pb[:,0], pb[:,1], Ub[Nb:,-1], cmap=mapa_de_color)
# ax2.tricontourf(pc[:,0], pc[:,1], Uc[Nc:,-1], cmap=mapa_de_color)

# =============================================================================
# Psi and C at different times
# =============================================================================
levelsP = 20
levelsC = 20

fig, axes = plt.subplots(4, 2, sharex="col", sharey="row", figsize=(9,10), constrained_layout=True)

fig.suptitle("Solution at different times with $N=%d$" %Nc, fontsize=20)

ax1 = axes[0,0]
ax2 = axes[0,1]
ax3 = axes[1,0]
ax4 = axes[1,1]
ax5 = axes[2,0]
ax6 = axes[2,1]
ax7 = axes[3,0]
ax8 = axes[3,1]

ax1.set_aspect("equal", "box")
ax2.set_aspect("equal", "box")
ax3.set_aspect("equal", "box")
ax4.set_aspect("equal", "box")
ax5.set_aspect("equal", "box")
ax6.set_aspect("equal", "box")
ax7.set_aspect("equal", "box")
ax8.set_aspect("equal", "box")

ax1.tricontourf(pc[:,0], pc[:,1], Uc[:Nc,0], cmap=mapa_de_color, levels=levelsP)
ax1.set_title("$\Psi$ at $t=%1.3f$" %tc[0])

ax2.tricontourf(pc[:,0], pc[:,1], Uc[Nc:,0], cmap=mapa_de_color, levels=levelsC)
ax2.set_title("$C$ at $t=%1.3f$" %tc[0])

ax3.tricontourf(pc[:,0], pc[:,1], Uc[:Nc,1], cmap=mapa_de_color, levels=levelsP)
ax3.set_title("$\Psi$ at $t=%1.3f$" %tc[1])

ax4.tricontourf(pc[:,0], pc[:,1], Uc[Nc:,1], cmap=mapa_de_color, levels=levelsC)
ax4.set_title("$C$ at $t=%1.3f$" %tc[1])

ax5.tricontourf(pc[:,0], pc[:,1], Uc[:Nc,2], cmap=mapa_de_color, levels=levelsP)
ax5.set_title("$\Psi$ at $t=%1.3f$" %tc[2])

ax6.tricontourf(pc[:,0], pc[:,1], Uc[Nc:,2], cmap=mapa_de_color, levels=levelsC)
ax6.set_title("$C$ at $t=%1.3f$" %tc[2])

ax7.tricontourf(pc[:,0], pc[:,1], Uc[:Nc,3], cmap=mapa_de_color, levels=levelsP)
ax7.set_title("$\Psi$ at $t=%1.3f$" %tc[3])

ax8.tricontourf(pc[:,0], pc[:,1], Uc[Nc:,3], cmap=mapa_de_color, levels=levelsC)
ax8.set_title("$C$ at $t=%1.3f$" %tc[3])

