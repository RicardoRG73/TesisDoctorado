# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:46:10 2024
@author: ricardo
"""
#%%
# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-talk"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.6
color_map = "plasma"

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

from graficas import nodos_por_color
from GFDM import create_system_K_F

#%%
# =============================================================================
# Geometry
# =============================================================================

geometry = cfg.Geometry()

# points
geometry.point([0,0])      # 0
geometry.point([4,0])      # 1
geometry.point([4,1])      # 2
geometry.point([3,1])      # 3
geometry.point([1,1])      # 4
geometry.point([0,1])      # 5

# lines
left = 10
right = 11
top_a = 12
top_b = 13
top_c = 14
bottom = 15

geometry.line([0,1], marker=bottom)    # 0
geometry.line([1,2], marker=right)     # 1
geometry.line([2,3], marker=top_c)       # 2
geometry.line([3,4], marker=top_b)      # 3
geometry.line([4,5], marker=top_a)      # 4
geometry.line([5,0], marker=left)      # 5


# surfaces
mat0 = 0
geometry.surface([0,1,2,3,4,5], marker=mat0)


# plotting
cfv.figure()
cfv.title('Geometry')
cfv.draw_geometry(geometry, draw_axis=True)

#%%
# =============================================================================
# Mesh
# =============================================================================
mesh = cfm.GmshMesh(
    geometry,
    el_size_factor=0.06
)

coords, edof, dofs, bdofs, elementmarkers = mesh.create()
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

# plotting
cfv.figure()
cfv.title('Malla $N=%d' %coords.shape[0] +'$')
cfv.draw_mesh(
    coords=coords,
    edof=edof,
    dofs_per_node=mesh.dofs_per_node,
    el_type=mesh.el_type,
    filled=True
)

#%%
# =============================================================================
# Nodes identification by color
# bl: left
# br: right
# bb: bottom
# bt: top
# =============================================================================
bl = np.asarray(bdofs[left]) - 1
bl = np.setdiff1d(bl, [0,5])
br = np.asarray(bdofs[right]) - 1
br = np.setdiff1d(br, [1,2])
bb = np.asarray(bdofs[bottom]) - 1
bb = np.setdiff1d(bb, [0,1])
bta = np.asarray(bdofs[top_a]) - 1
bta = np.setdiff1d(bta, [4,5])
btb = np.asarray(bdofs[top_b]) - 1
btc = np.asarray(bdofs[top_c]) - 1
btc = np.setdiff1d(btc, [2,3])

esquinas = np.array([0,1,2,5])

boundaries_tuple = (bl, br, bb, bta, btb, btc, esquinas)
Boundaries = np.hstack(boundaries_tuple)

N = coords.shape[0]
interior = np.setdiff1d(np.arange(N) , np.hstack(boundaries_tuple))
etiquetas = (
    "Left",
    "Right",
    "Bottom",
    "Top-a",
    "Top-b",
    "Top-c",
    "Esquinas"
)

plt.figure()
nodos_por_color(
    boundaries=boundaries_tuple,
    p=coords,
    labels=etiquetas,
    interior=interior,
    label_interior="Nodos interior",
    alpha=1,
    nums=True,
    legend=True,
    loc="center"
)
plt.axis('equal')

#%%
# =============================================================================
# Problem parameters
# =============================================================================
Ra = 400
k = lambda p: 1
f = lambda p: 0

L2 = np.array([0,0,0,2,0,2])
Lx = np.array([0,1,0,0,0,0])
Ly = np.array([0,0,1,0,0,0])

#%%
# =============================================================================
# Boundary conditions
# P: Stream-function (Psi)
# C: Concentration
# =============================================================================
P = lambda p: 0

CN = lambda p: 0
Ct = lambda p: 1
Cb = lambda p: 0

#%%
# =============================================================================
# P-Operators discretization using GFDM
# =============================================================================
materials = {}
materials["0"] = [k, interior]

PDirich = {}
PDirich["All"] = [Boundaries, P]

PNeu = {}

D2P, F2P = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L2,
    source=f,
    materials=materials,
    dirichlet_boundaries=PDirich,
    neumann_boundaries=PNeu
)
D2P = D2P.toarray()

DxP, FxP = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Lx,
    source=f,
    materials=materials,
    dirichlet_boundaries=PDirich,
    neumann_boundaries=PNeu
)
DxP.toarray()

DyP, FyP = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Ly,
    source=f,
    materials=materials,
    dirichlet_boundaries=PDirich,
    neumann_boundaries=PNeu
)
DyP.toarray()

#%%
# =============================================================================
# C-Operators discretization using GFDM
# =============================================================================
CDirich = {}
CDirich["Top-b"] = [btb, Ct]
CDirich["Bottom"] = [np.hstack((0,1,bb)), Cb]

CNeu = {}
CNeu["Left"] = [k, np.hstack((5,bl)), CN]
CNeu["Right"] = [k, np.hstack((2,br)), CN]
CNeu["Top-a"] = [k, bta, CN]
CNeu["Top-c"] = [k, btc, CN]

D2C, F2C = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L2,
    source=f,
    materials=materials,
    dirichlet_boundaries=CDirich,
    neumann_boundaries=CNeu
)
D2C = D2C.toarray()

DxC, FxC = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Lx,
    source=f,
    materials=materials,
    dirichlet_boundaries=CDirich,
    neumann_boundaries=CNeu
)
DxC = DxC.toarray()

DyC, FyC = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Ly,
    source=f,
    materials=materials,
    dirichlet_boundaries=CDirich,
    neumann_boundaries=CNeu
)
DyC = DyC.toarray()

#%%
# =============================================================================
# Problem coupling
# =============================================================================
zeros_mat = np.zeros((N,N))
zeros_vec = np.zeros(N)

DxCP = DxC.copy()
DxCP[Boundaries,:] = 0
FxCP = FxC.copy()
FxCP[Boundaries] = 0

DyPC = DyP.copy()
DyPC[Boundaries,:] = 0
FyPC = FyP.copy()
FyPC[Boundaries] = 0

DxPC = DxP.copy()
DxPC[Boundaries,:] = 0
FxPC = FxP.copy()
FxP[Boundaries] = 0

# Linear
Linear_mat = np.vstack((
    np.hstack((D2P, -Ra*DxCP)),
    np.hstack((zeros_mat, D2C))
))

Linear_vec = - np.hstack((
    F2P - Ra*FxCP,
    F2C
))

ULinear = np.linalg.solve(Linear_mat, Linear_vec)

fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
contourf1 = ax1.tricontourf(coords[:,0], coords[:,1], ULinear[:N], cmap=color_map)
ax1.axis("equal")
ax1.set_title("$\Psi$")
fig.colorbar(contourf1)

contourf2 = ax2.tricontourf(coords[:,0], coords[:,1], ULinear[N:], cmap=color_map)
ax2.axis("equal")
ax2.set_title("$C$")
fig.colorbar(contourf2)

fig.suptitle("Linear-Stationary solution")

# Non-Linear
def nonLinear_mat(U):
    term1 = (DyPC @ U[:N]) * (DxC @ U[N:])
    term2 = (DxPC @ U[:N]) * (DyC @ U[N:])
    vec = np.hstack((
        zeros_vec,
        - term1 + term2
    ))
    return vec

nonLinear_vec = - np.hstack((
    zeros_vec,
    (FyPC * FxC) - (FxPC * FyC)
))

def rhs(t,U):
    vec = Linear_mat @ U + Linear_vec
    vec += nonLinear_mat(U) + nonLinear_vec
    return vec

#%%
# =============================================================================
# Solving IVP with RKF45
# =============================================================================
tfinal = 1.239
tspan = [0, tfinal]

P0 = zeros_vec.copy()
C0 = zeros_vec.copy()

C0[btb] = 1

U0 = np.hstack((P0,C0))

sol = solve_ivp(rhs, tspan, U0)

U = sol.y
times = sol.t

# %%
# =============================================================================
# Plotting solution
# =============================================================================

levelsP = 20
levelsC = 20

fig, axes = plt.subplots(3, 2, sharex="col", sharey="row")

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

ax1.tricontourf(coords[:,0], coords[:,1], U[:N,0], cmap=color_map, levels=levelsP)
ax1.set_title("$\Psi$ at $t=%1.3f" %sol.t[0] + "$")

# ax2 = plt.subplot(322)
ax2.tricontourf(coords[:,0], coords[:,1], U[N:,0], cmap=color_map, levels=levelsC)
ax2.set_title("$C$ at $t=%1.3f" %sol.t[0] + "$")

t_index = sol.t.shape[0]//8
# ax3 = plt.subplot(323, sharex=True)
ax3.tricontourf(coords[:,0], coords[:,1], U[:N,t_index], cmap=color_map, levels=levelsP)
ax3.set_title("$\Psi$ at $t=%1.3f" %sol.t[t_index] + "$")

# ax4 = plt.subplot(324)
ax4.tricontourf(coords[:,0], coords[:,1], U[N:,t_index], cmap=color_map, levels=levelsC)
ax4.set_title("$C$ at $t=%1.3f" %sol.t[t_index] + "$")

# ax5 = plt.subplot(325)
ax5.tricontourf(coords[:,0], coords[:,1], U[:N,-1], cmap=color_map, levels=levelsP)
ax5.set_title("$\Psi$ at $t=%1.3f" %sol.t[-1] + "$")

# ax6 = plt.subplot(326)
ax6.tricontourf(coords[:,0], coords[:,1], U[N:,-1], cmap=color_map, levels=levelsC)
ax6.set_title("$C$ at $t=%1.3f" %sol.t[-1] + "$")

fig.suptitle("Solution with $N=%d" %coords.shape[0] +"$")

plt.show()