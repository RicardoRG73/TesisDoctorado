# -*- coding: utf-8 -*-
# =============================================================================
# Importing libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt


plt.style.use([
    "seaborn-v0_8-darkgrid",
    "seaborn-v0_8-colorblind",
    "seaborn-v0_8-talk"
])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.6
color_map = "plasma"

from GFDM import create_system_K_F

# =============================================================================
# Mesh
# =============================================================================
nodes_x = 4
nodes_y = 4

coords_x = np.linspace(0,2,nodes_x)
coords_y = np.linspace(0,1,nodes_y)
coords_x,coords_y = np.meshgrid(coords_x,coords_y)

# p: coords
p = np.vstack((
    coords_x.T.flatten(),
    coords_y.T.flatten()
)).T

tri = np.array([
    [0, nodes_y, nodes_y+1],
    [0, nodes_y+1, 1]
])

tritemp = tri.copy()
for i in range(nodes_y-2):
    tri = np.vstack((
        tri,
        tritemp+i+1
    ))
tritemp = tri.copy()
for i in range(nodes_x-2):
    tri = np.vstack((tri, tritemp+(i+1)*nodes_y))

# Ploting triangulation
plt.figure()
plt.triplot(p[:,0], p[:,1], tri, lw=1)
plt.scatter(coords_x,coords_y)
for i in range(p.shape[0]):
    plt.text(p[i,0],p[i,1],s=str(i))
plt.axis("equal")

# =============================================================================
# Boundaries indexing
# L: Left, R: Right, T: Top, B: Bottom.
# =============================================================================
N = nodes_x * nodes_y   # N: total nodes

cornerLB = 0
cornerLT = nodes_y - 1
cornerRB = N-nodes_y
cornerRT = N - 1

index_corners = np.array([
    cornerLB,
    cornerLT, 
    cornerRB,
    cornerRT
])

indexB = np.arange(nodes_y, N-nodes_y, nodes_y)

indexT = np.arange(2*nodes_y-1, N-nodes_y, nodes_y)

indexL = np.arange(1, nodes_y-1)
indexL = np.hstack((cornerLB, indexL, cornerLT))

indexR = np.arange(N-nodes_y+1, N-1)
indexR = np.hstack((cornerRB, indexR, cornerRT))

Boundaries = np.hstack((index_corners,indexB,indexT,indexL,indexR))

interior = np.arange(0,N)
interior = np.setdiff1d(interior, Boundaries)

# Ploting Boundaries by color
plt.scatter(p[index_corners,0], p[index_corners,1], label="Corners")
plt.scatter(p[indexB,0], p[indexB,1], label="Bottom")
plt.scatter(p[indexT,0], p[indexT,1], label="Top")
plt.scatter(p[indexL,0], p[indexL,1], label="Left")
plt.scatter(p[indexR,0], p[indexR,1], label="Right")
plt.scatter(p[interior,0], p[interior,1], label="Interior")
plt.legend(loc="center")
plt.axis("equal")
plt.title("Boundaries by color")

# =============================================================================
# Problem parameters
# =============================================================================
a = 0.2637
b = 0.1
k = lambda p: 1         # difusividad
f = lambda p: 0         # fuente

# =============================================================================
# Boundary Conditions (BC)
# =============================================================================
# C: Concentration
CL = lambda p: 0
CR = lambda p: 1
CT = lambda p: 0
CB = lambda p: 0

# P: Stream-function (Psi)
PL = lambda p: 0
PR = lambda p: 0
PT = lambda p: 1
PB = lambda p: 0

# Ploting BC
fig = plt.figure()

ax1 = plt.subplot(121, projection="3d")

ztemp = np.array([CL(p[i,:]) for i in indexL])
ax1.scatter3D(p[indexL,0], p[indexL,1], ztemp, depthshade=False, label="$C=0$")

ztemp = np.array([CR(p[i,:]) for i in indexR])
ax1.scatter3D(p[indexR,0], p[indexR,1], ztemp, depthshade=False, label="$C=1$")

ztemp = np.array([CT(p[i,:]) for i in indexT])
ax1.scatter3D(p[indexT,0], p[indexT,1], ztemp, depthshade=False, label="$C_n=0$")

ztemp = np.array([CB(p[i,:]) for i in indexB])
ax1.scatter3D(p[indexB,0], p[indexB,1], ztemp, depthshade=False, label="$C_n=0$")

ax1.legend()

ax1.set_title("C boundary conditions")
ax1.view_init(elev=30, azim=-90)
ax1.set_ylabel("y")
ax1.set_xlabel("x")
ax1.set_zlabel("z")

ax2 = plt.subplot(122, projection="3d")

ztemp = np.array([PL(p[i,:]) for i in indexL])
ax2.scatter3D(p[indexL,0], p[indexL,1], ztemp, depthshade=False, label="$\Psi_n=0$")

ztemp = np.array([PR(p[i,:]) for i in indexR])
ax2.scatter3D(p[indexR,0], p[indexR,1], ztemp, depthshade=False, label="$\Psi_n=0$")

ztemp = np.array([PT(p[i,:]) for i in indexT])
ax2.scatter3D(p[indexT,0], p[indexT,1], ztemp, depthshade=False, label="$\Psi=1$")

ztemp = np.array([PB(p[i,:]) for i in indexB])
ax2.scatter3D(p[indexB,0], p[indexB,1], ztemp, depthshade=False, label="$\Psi=0$")

ax2.legend()
ax2.set_title("$\Psi$ boundary conditions")
ax2.view_init(elev=30, azim=-90)
ax2.set_ylabel("y")
ax2.set_xlabel("x")
ax2.set_zlabel("z")

# =============================================================================
# Boundary and material dictionaries
# =============================================================================
materials = {}
materials["0"] = [k, interior]

# C: Concentration
CDirich = {}
CDirich["L"] = [indexL, CL]
CDirich["R"] = [indexR, CR]
CNeumann = {}
CNeumann["T"] = [k, indexT, CT]
CNeumann["B"] = [k, indexB, CB]

L2 = np.array([0,0,0,2,0,2])
K,F = create_system_K_F(p, tri, L2, f, materials, CNeumann, CDirich)

from scipy import sparse as sp
U = sp.linalg.spsolve(K,F)
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_trisurf(p[:,0],p[:,1],U)
ax.view_init(elev=30, azim=-90)
plt.show()