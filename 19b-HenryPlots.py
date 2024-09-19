#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:09:55 2024

@author: ricardo
"""
save_figures=False
# =============================================================================
# Libraries
# =============================================================================
import pickle
import matplotlib.pyplot as plt

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-talk"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = False
plt.rcParams["legend.framealpha"] = 0.1
mapa_de_color = "plasma"

# =============================================================================
# Load files
# a) N = 274
# b) N = 998
# c) N = 5969
# d) N = 2748
# =============================================================================

path = "figuras/Henry/solN274.pkl"
Ua,ta,pa,Dxpsia,Dypsia = pickle.load(open(path, "rb"))
Na = pa.shape[0]

path = "figuras/Henry/solN998.pkl"
Ub,tb,pb,Dxpsib,Dypsib = pickle.load(open(path, "rb"))
Nb = pb.shape[0]

path = "figuras/Henry/solN5969.pkl"
Uc,tc,pc,Dxpsic,Dypsic = pickle.load(open(path, "rb"))
Nc = pc.shape[0]

path = "figuras/Henry/PinderSolN274.pkl"
PUa,Pta,Ppa,PDxpsia,PDypsia = pickle.load(open(path, "rb"))
PNa = Ppa.shape[0]

path = "figuras/Henry/PinderSolN998.pkl"
PUb,Ptb,Ppb,PDxpsib,PDypsib = pickle.load(open(path, "rb"))
PNb = Ppb.shape[0]

path = "figuras/Henry/PinderSolN5969.pkl"
PUc,Ptc,Ppc,PDxpsic,PDypsic = pickle.load(open(path, "rb"))
PNc = Ppc.shape[0]

path = "figuras/Henry/ModifiedSolN274.pkl"
MUa,Mta,Mpa,MDxpsia,MDypsia = pickle.load(open(path, "rb"))
MNa = Mpa.shape[0]

path = "figuras/Henry/ModifiedSolN998.pkl"
MUb,Mtb,Mpb,MDxpsib,MDypsib = pickle.load(open(path, "rb"))
MNb = Mpb.shape[0]

path = "figuras/Henry/ModifiedSolN5969.pkl"
MUc,Mtc,Mpc,MDxpsic,MDypsic = pickle.load(open(path, "rb"))
MNc = Mpc.shape[0]

path = "figuras/Henry/solN2748.pkl"
Ud,td,pd,Dxpsid,Dypsid = pickle.load(open(path, "rb"))
Nd = pd.shape[0]

path = "figuras/Henry/PinderSolN2748.pkl"
PUd,Ptd,Ppd,PDxpsid,PDypsid = pickle.load(open(path, "rb"))
PNd = Ppd.shape[0]

path = "figuras/Henry/ModifiedSolN2748.pkl"
MUd,Mtd,Mpd,MDxpsid,MDypsid = pickle.load(open(path, "rb"))
MNd = Mpd.shape[0]

# =============================================================================
# Psi and C at different times
# =============================================================================
levelsP = 16
levelsC = 9

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

if save_figures:
    plt.savefig("figuras/Henry/Psi_C_diff_t.pdf")

# =============================================================================
# Psi and C t=0.21 with different N
# =============================================================================
fig, axes = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(9,8), constrained_layout=True)

fig.suptitle("Solution with differetn $N$ at $t=0.21$", fontsize=20)

ax1 = axes[0,0]
ax2 = axes[0,1]
ax3 = axes[1,0]
ax4 = axes[1,1]
ax5 = axes[2,0]
ax6 = axes[2,1]

ax1.set_aspect("equal", "box")
ax2.set_aspect("equal", "box")
ax3.set_aspect("equal", "box")
ax4.set_aspect("equal", "box")
ax5.set_aspect("equal", "box")
ax6.set_aspect("equal", "box")

ax1.tricontourf(pa[:,0], pa[:,1], Ua[:Na,-1], cmap=mapa_de_color, levels=levelsP)
ax1.set_title("$\Psi$, $N=%d$" %Na)

ax2.tricontourf(pa[:,0], pa[:,1], Ua[Na:,-1], cmap=mapa_de_color, levels=levelsP)
ax2.set_title("$C$, $N=%d$" %Na)

ax3.tricontourf(pb[:,0], pb[:,1], Ub[:Nb,-1], cmap=mapa_de_color, levels=levelsP)
ax3.set_title("$\Psi$, $N=%d$" %Nb)

ax4.tricontourf(pb[:,0], pb[:,1], Ub[Nb:,-1], cmap=mapa_de_color, levels=levelsP)
ax4.set_title("$C$, $N=%d$" %Nb)

ax5.tricontourf(pc[:,0], pc[:,1], Uc[:Nc,-1], cmap=mapa_de_color, levels=levelsP)
ax5.set_title("$\Psi$, $N=%d$" %Nc)

ax6.tricontourf(pc[:,0], pc[:,1], Uc[Nc:,-1], cmap=mapa_de_color, levels=levelsP)
ax6.set_title("$C$, $N=%d$" %Nc)

if save_figures:
    plt.savefig("figuras/Henry/Psi_C_diff_N.pdf")

# =============================================================================
# Psi and C t=0.21 with different N, Pinder version (a=0.2637, b=0.035)
# =============================================================================
fig, axes = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(9,8), constrained_layout=True)

fig.suptitle(
    "Pinder $(a=0.2637, b=0.035)$ Solution with differetn $N$ at $t=0.21$",
    fontsize=20
)

ax1 = axes[0,0]
ax2 = axes[0,1]
ax3 = axes[1,0]
ax4 = axes[1,1]
ax5 = axes[2,0]
ax6 = axes[2,1]

ax1.set_aspect("equal", "box")
ax2.set_aspect("equal", "box")
ax3.set_aspect("equal", "box")
ax4.set_aspect("equal", "box")
ax5.set_aspect("equal", "box")
ax6.set_aspect("equal", "box")

ax1.tricontourf(Ppa[:,0], Ppa[:,1], PUa[:PNa,-1], cmap=mapa_de_color, levels=levelsP)
ax1.set_title("$\Psi$, $N=%d$" %PNa)

ax2.tricontourf(Ppa[:,0], Ppa[:,1], PUa[PNa:,-1], cmap=mapa_de_color, levels=levelsP)
ax2.set_title("$C$, $N=%d$" %PNa)

ax3.tricontourf(Ppb[:,0], Ppb[:,1], PUb[:PNb,-1], cmap=mapa_de_color, levels=levelsP)
ax3.set_title("$\Psi$, $N=%d$" %PNb)

ax4.tricontourf(Ppb[:,0], Ppb[:,1], PUb[PNb:,-1], cmap=mapa_de_color, levels=levelsP)
ax4.set_title("$C$, $N=%d$" %PNb)

ax5.tricontourf(Ppc[:,0], Ppc[:,1], PUc[:PNc,-1], cmap=mapa_de_color, levels=levelsP)
ax5.set_title("$\Psi$, $N=%d$" %PNc)

ax6.tricontourf(Ppc[:,0], Ppc[:,1], PUc[PNc:,-1], cmap=mapa_de_color, levels=levelsP)
ax6.set_title("$C$, $N=%d$" %PNc)

if save_figures:
    plt.savefig("figuras/Henry/Psi_C_diff_N_Pinder.pdf")

# =============================================================================
# Isochlor C=0.5, Pinder and Modified versions
# Pinder: a=0.2637, b=0.035
# Modified: a=0.1315, b=0.2
# =============================================================================
fig = plt.figure()
ax = plt.axes()

ax.tricontour(pa[:,0], pa[:,1], Ua[Na:,-1], levels=[0.5])
ax.tricontour(pb[:,0], pb[:,1], Ub[Nb:,-1], levels=[0.5])
ax.tricontour(pc[:,0], pc[:,1], Uc[Nc:,-1], levels=[0.5])
ax.tricontour(pd[:,0], pd[:,1], Ud[Nd:,-1], levels=[0.5])

ax.tricontour(Ppa[:,0], Ppa[:,1], PUa[PNa:,-1], levels=[0.5])
ax.tricontour(Ppb[:,0], Ppb[:,1], PUb[PNb:,-1], levels=[0.5])
ax.tricontour(Ppc[:,0], Ppc[:,1], PUc[PNc:,-1], levels=[0.5])
ax.tricontour(Ppd[:,0], Ppd[:,1], PUd[Nd:,-1], levels=[0.5])

ax.tricontour(Mpa[:,0], Mpa[:,1], MUa[MNa:,-1], levels=[0.5])
ax.tricontour(Mpb[:,0], Mpb[:,1], MUb[MNb:,-1], levels=[0.5])
ax.tricontour(Mpc[:,0], Mpc[:,1], MUc[MNc:,-1], levels=[0.5])
ax.tricontour(Mpd[:,0], Mpd[:,1], MUd[Nd:,-1], levels=[0.5])

lineOa = ax.collections[0].get_paths()[0].vertices
lineOb = ax.collections[1].get_paths()[0].vertices
lineOc = ax.collections[2].get_paths()[0].vertices
lineOd = ax.collections[3].get_paths()[0].vertices

linePa = ax.collections[4].get_paths()[0].vertices
linePb = ax.collections[5].get_paths()[0].vertices
linePc = ax.collections[6].get_paths()[0].vertices
linePd = ax.collections[7].get_paths()[0].vertices

lineMa = ax.collections[8].get_paths()[0].vertices
lineMb = ax.collections[9].get_paths()[0].vertices
lineMc = ax.collections[10].get_paths()[0].vertices
lineMd = ax.collections[11].get_paths()[0].vertices

# Original vs Pinder vs Modified isoclor C=0.5
plt.figure()
plt.plot(lineOd[:,0], lineOd[:,1], label="Original")
plt.plot(linePd[:,0], linePd[:,1], label="Pinder")
plt.plot(lineMd[:,0], lineMd[:,1], label="Modified")
plt.axis("equal")
plt.legend()
plt.title("Isochlor $C=0.5$")

if save_figures:
    plt.savefig("figuras/Henry/C=0.5_versions.pdf")

# All isochlors C=0.5
fig, axes= plt.subplots(1,2, sharey="row", sharex="col", constrained_layout=True)

ax1 = axes[0]
ax2 = axes[1]

ax1.set_aspect("equal", "box")
ax2.set_aspect("equal", "box")

ax1.plot(linePa[:,0], linePa[:,1], "--", label="$N=274$")
ax1.plot(linePb[:,0], linePb[:,1], label="$N=998$")
ax1.plot(linePd[:,0], linePd[:,1], "--", label="$N=2748$")
ax1.plot(linePc[:,0], linePc[:,1], label="$N=5969$")
ax1.legend()
ax1.set_title("Pinder $(a=0.2637, b=0.035)$")

ax2.plot(lineMa[:,0], lineMa[:,1], "--", label="$N=274$")
ax2.plot(lineMb[:,0], lineMb[:,1], label="$N=998$")
ax2.plot(lineMd[:,0], lineMd[:,1], "--", label="$N=2748$")
ax2.plot(lineMc[:,0], lineMc[:,1], label="$N=5969$")
ax2.legend()
ax2.set_title("Modified $(a=0.1315, b=0.2)$")

if save_figures:
    plt.savefig("figuras/Henry/C=0.5_diff_N.pdf")

# =============================================================================
# Xtoe position table, Original, Pinder, Modified
# d) el_size=0.03, N=2748
# =============================================================================
import pandas
import numpy as np

num_nodes = [Na,Nb,Nd,Nc]
OriginalxToe = np.array([
    lineOa[ lineOa[:,1] == 0. , 0][0],
    lineOb[ lineOb[:,1] == 0. , 0][0],
    lineOc[ lineOc[:,1] == 0. , 0][0],
    lineOd[ lineOd[:,1] == 0. , 0][0]
])
OriginalxToe = np.round(OriginalxToe, decimals=4)
PinderxToe = np.array([
    linePa[ linePa[:,1] == 0. , 0][0],
    linePb[ linePb[:,1] == 0. , 0][0],
    linePc[ linePc[:,1] == 0. , 0][0],
    linePd[ linePd[:,1] == 0. , 0][0]
])
PinderxToe = np.round(PinderxToe, decimals=4)
ModifiedxToe = np.array([
    lineMa[ lineMa[:,1] == 0. , 0][0],
    lineMb[ lineMb[:,1] == 0. , 0][0],
    lineMc[ lineMc[:,1] == 0. , 0][0],
    lineMd[ lineMd[:,1] == 0. , 0][0]
])
ModifiedxToe = np.round(ModifiedxToe, decimals=4)

tabla = {
    "Num_nodes": num_nodes,
    "Original": OriginalxToe,
    "Pinder": PinderxToe,
    "Modified": ModifiedxToe
}

df = pandas.DataFrame(tabla)
print("   XToe Position \n _________________________")
print(df)

# =============================================================================
# Stationary Psi, C, vel, Original, Pinder, Modified
# =============================================================================
from scipy.interpolate import griddata

x = np.linspace(0, 2, 41)
y = np.linspace(0, 1, 21)
x,y = np.meshgrid(x,y)

fig, axes = plt.subplots(3, 3, sharex="col", sharey="row", figsize=(15,9), constrained_layout=True)

ax0 = axes[0,0]
ax1 = axes[1,0]
ax2 = axes[2,0]
ax3 = axes[0,1]
ax4 = axes[1,1]
ax5 = axes[2,1]
ax6 = axes[0,2]
ax7 = axes[1,2]
ax8 = axes[2,2]

ax0.set_aspect("equal", "box")
ax1.set_aspect("equal", "box")
ax2.set_aspect("equal", "box")
ax3.set_aspect("equal", "box")
ax4.set_aspect("equal", "box")
ax5.set_aspect("equal", "box")
ax6.set_aspect("equal", "box")
ax7.set_aspect("equal", "box")
ax8.set_aspect("equal", "box")

ax0.tricontourf(pc[:,0], pc[:,1], Uc[:Nc,-1], cmap=mapa_de_color, levels=levelsP)
ax1.tricontourf(pc[:,0], pc[:,1], Uc[Nc:,-1], cmap=mapa_de_color, levels=levelsC)
vx = Dypsic @ Uc[:Nc,3]
vy = -Dxpsic @ Uc[:Nc,3]
velx = griddata(pc, vx, (x,y))
vely = griddata(pc, vy, (x,y))
norm_vel = np.sqrt(velx**2 + vely**2)
ax2.streamplot(x, y, velx, vely, color=norm_vel, cmap="plasma", density=0.7)

ax3.tricontourf(Ppc[:,0], Ppc[:,1], PUc[:PNc,-1], cmap=mapa_de_color, levels=levelsP)
ax4.tricontourf(Ppc[:,0], Ppc[:,1], PUc[PNc:,-1], cmap=mapa_de_color, levels=levelsC)
vx = PDypsic @ PUc[:PNc,3]
vy = -PDxpsic @ PUc[:PNc,3]
velx = griddata(Ppc, vx, (x,y))
vely = griddata(Ppc, vy, (x,y))
norm_vel = np.sqrt(velx**2 + vely**2)
ax5.streamplot(x, y, velx, vely, color=norm_vel, cmap="plasma", density=0.7)

ax6.tricontourf(Mpc[:,0], Mpc[:,1], MUc[:MNc,-1], cmap=mapa_de_color, levels=levelsP)
ax7.tricontourf(Mpc[:,0], Mpc[:,1], MUc[MNc:,-1], cmap=mapa_de_color, levels=levelsC)
vx = MDypsic @ MUc[:MNc,3]
vy = -MDxpsic @ MUc[:MNc,3]
velx = griddata(Mpc, vx, (x,y))
vely = griddata(Mpc, vy, (x,y))
norm_vel = np.sqrt(velx**2 + vely**2)
ax8.streamplot(x, y, velx, vely, color=norm_vel, cmap="plasma", density=0.7)

ax0.set_title("$\Psi$ Original")
ax1.set_title("$C$ Original")
ax2.set_title("Velocity Original")

ax3.set_title("$\Psi$ Pinder")
ax4.set_title("$C$ Pinder")
ax5.set_title("Velocity Pinder")

ax6.set_title("$\Psi$ Modified")
ax7.set_title("$C$ Modified")
ax8.set_title("Velocity Modified")

fig.suptitle("Solution at $t=0.21$ for different values $a$, $b$, using $N=%d$" %Nc, fontsize=20)

if save_figures:
    plt.savefig("figuras/Henry/stationary_versions.pdf")
