# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:19:29 2024

@author: rick_
"""
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

# =============================================================================
# Geometry
# th: Angle (Theta)
# HL: Proportion H/L
# =============================================================================
height = 0.3
HL = 0.25
length = height / HL

Degrees = np.pi / 180
th = -30 * Degrees

length_x = length * np.cos(th)
length_y = length * np.sin(th)

geometry = cfg.Geometry()

# points
geometry.point([0,0])      # 0
geometry.point([length_x, length_y])      # 1
geometry.point([length_x, length_y + height])      # 2
geometry.point([0, height])      # 3

# lines
left = 10
right = 11
top = 12
bottom = 13

geometry.line([0,1], marker=bottom)     # 0
geometry.line([1,2], marker=right)      # 1
geometry.line([2,3], marker=top)        # 2
geometry.line([3,0], marker=left)       # 3


# surfaces
mat0 = 0
geometry.surface([0,1,2,3], marker=mat0)


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
# =============================================================================
index_left = np.asarray(bdofs[left]) - 1

index_right = np.asarray(bdofs[right]) - 1

index_bottom = np.asarray(bdofs[bottom]) - 1
index_bottom = np.setdiff1d(index_bottom, [0,1])

index_top = np.asarray(bdofs[top]) - 1
index_top = np.setdiff1d(index_top, [2,3])

boundaries_tuple = (index_left, index_right, index_bottom, index_top)
Boundaries = np.hstack(boundaries_tuple)

Nnodes = coords.shape[0]
interior = np.setdiff1d(np.arange(Nnodes) , np.hstack(boundaries_tuple))
etiquetas = (
    "Left",
    "Right",
    "Bottom",
    "Top"
)

plt.figure()
nodos_por_color(
    boundaries=boundaries_tuple,
    p=coords,
    labels=etiquetas,
    interior=interior,
    label_interior="Interior",
    alpha=1,
    nums=True,
    legend=True,
    loc="best"
)
plt.axis('equal')

#%%
# =============================================================================
# Problem parameters
# Ra: Rayleigh number
# Le: Lewis number
# N: Buoyancy Ratio
# =============================================================================
Ra = 100
Le = 0.8
N = 2
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
# T: Temperature
# =============================================================================
P = lambda p: 0

CN = lambda p: 0
CL = lambda p: 1
CR = lambda p: 0

TN = lambda p: 0
TL = lambda p: 1
TR = lambda p: 0

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
CDirich["Left"] = [index_left, CL]
CDirich["Right"] = [index_right, CR]

CNeu = {}
CNeu["Top"] = [k, index_top, CN]
CNeu["Bottom"] = [k, index_bottom, CN]

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
# T-Operators discretization using GFDM
# =============================================================================
TDirich = {}
TDirich["Left"] = [index_left, TL]
TDirich["Right"] = [index_right, TR]

TNeu = {}
TNeu["Top"] = [k, index_top, TN]
TNeu["Bottom"] = [k, index_bottom, TN]

D2T, F2T = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L2,
    source=f,
    materials=materials,
    dirichlet_boundaries=TDirich,
    neumann_boundaries=TNeu
)
D2T = D2T.toarray()

DxT, FxT = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Lx,
    source=f,
    materials=materials,
    dirichlet_boundaries=TDirich,
    neumann_boundaries=TNeu
)
DxT = DxT.toarray()

DyT, FyT = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Ly,
    source=f,
    materials=materials,
    dirichlet_boundaries=TDirich,
    neumann_boundaries=TNeu
)
DyT = DyT.toarray()

#%%
# =============================================================================
# Problem coupling
# =============================================================================
zeros_mat = np.zeros((Nnodes,Nnodes))
zeros_vec = np.zeros(Nnodes)

DxC_noBound = DxC.copy()
DxC_noBound[Boundaries,:] = 0
FxC_noBound = FxC.copy()
FxC_noBound[Boundaries] = 0

DxT_noBound = DxT.copy()
DxT_noBound[Boundaries,:] = 0
FxT_noBound = FxT.copy()
FxT_noBound[Boundaries] = 0

DyP_noBound = DyP.copy()
DyP_noBound[Boundaries,:] = 0
FyP_noBound = FyP.copy()
FyP_noBound[Boundaries] = 0

DxP_noBound = DxP.copy()
DxP_noBound[Boundaries,:] = 0
FxP_noBound = FxP.copy()
FxP_noBound[Boundaries] = 0

# Linear
Linear_mat = np.vstack((
    np.hstack((D2P, Ra*DxT_noBound, (Ra*N)*DxC_noBound)),
    np.hstack((zeros_mat, D2T, zeros_mat)),
    np.hstack((zeros_mat, zeros_mat, D2C))
))

Linear_vec = - np.hstack((
    F2P + Ra*FxT_noBound + (Ra*N)*FxC_noBound,
    F2T,
    F2C
))

ULinear = np.linalg.solve(Linear_mat, Linear_vec)

fig, axes = plt.subplots(2,2, sharex="col", sharey="row")

ax0 = axes[0,0]
ax1 = axes[0,1]
ax2 = axes[1,0]
ax3 = axes[1,1]

ax0.set_aspect("equal", "box")
ax1.set_aspect("equal", "box")
ax2.set_aspect("equal", "box")
ax3.set_aspect("equal", "box")

# ax0.scatter(coords[:,0], coords[:,1])
meshplot = ax0.triplot(coords[:,0], coords[:,1], faces)
ax0.set_title("Mesh with $%d$ nodes" %Nnodes)

contourf1 = ax1.tricontourf(coords[:,0], coords[:,1], ULinear[:Nnodes], cmap=color_map)
ax1.set_title("$\Psi$")
# fig.colorbar(contourf1)

contourf2 = ax2.tricontourf(coords[:,0], coords[:,1], ULinear[Nnodes:2*Nnodes], cmap=color_map)
ax2.set_title("$T$")
# fig.colorbar(contourf2)

contourf3 = ax3.tricontourf(coords[:,0], coords[:,1], ULinear[2*Nnodes:], cmap=color_map)
ax3.set_title("$C$")
# fig.colorbar(contourf3)

fig.suptitle("Linear-Stationary solution")

# Non-Linear
def nonLinear_mat(U):
    Tterm1 = (DyP_noBound @ U[:Nnodes]) * (DxT @ U[Nnodes:2*Nnodes])
    Tterm2 = (DxP_noBound @ U[:Nnodes]) * (DyT @ U[Nnodes:2*Nnodes])
    
    Cterm1 = (DyP_noBound @ U[:Nnodes]) * (DxC @ U[2*Nnodes:])
    Cterm2 = (DxP_noBound @ U[:Nnodes]) * (DyC @ U[2*Nnodes:])
    vec = np.hstack((
        zeros_vec,
        - Tterm1 + Tterm2,
        Le * (- Cterm1 + Cterm2)
    ))
    return vec

nonLinear_vec = - np.hstack((
    zeros_vec,
    (FyP_noBound * FxT) - (FxP_noBound * FyT),
    Le * ( (FyP_noBound * FxC) - (FxP_noBound * FyC) )
))

def rhs(t,U):
    vec = Linear_mat @ U + Linear_vec
    vec += nonLinear_mat(U) + nonLinear_vec
    return vec

#%%
# =============================================================================
# Solving IVP with RKF45
# =============================================================================
tfinal = 1.2
tspan = [0, tfinal]

P0 = zeros_vec.copy()
T0 = zeros_vec.copy()
C0 = zeros_vec.copy()

T0[index_left] = 1
C0[index_left] = 1

U0 = np.hstack((P0,T0,C0))

sol = solve_ivp(rhs, tspan, U0)

U = sol.y
times = sol.t

# %%
# =============================================================================
# Plotting solution
# =============================================================================

levelsP = 20
levelsT = 20
levelsC = 20

fig, axes = plt.subplots(3, 3, sharex="col", sharey="row")

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

ax1.tricontourf(coords[:,0], coords[:,1], U[:Nnodes,0], cmap=color_map, levels=levelsP)
ax1.set_title("$\Psi$ at $t=%1.3f" %sol.t[0] + "$")

ax2.tricontourf(coords[:,0], coords[:,1], U[Nnodes:2*Nnodes,0], cmap=color_map, levels=levelsT)
ax2.set_title("$T$ at $t=%1.3f" %sol.t[0] + "$")

ax7.tricontourf(coords[:,0], coords[:,1], U[2*Nnodes:,0], cmap=color_map, levels=levelsC)
ax7.set_title("$C$ at $t=%1.3f" %sol.t[0] + "$")

t_index = sol.t.shape[0]//8
ax3.tricontourf(coords[:,0], coords[:,1], U[:Nnodes,t_index], cmap=color_map, levels=levelsP)
ax3.set_title("$\Psi$ at $t=%1.3f" %sol.t[t_index] + "$")

ax4.tricontourf(coords[:,0], coords[:,1], U[Nnodes:2*Nnodes,t_index], cmap=color_map, levels=levelsT)
ax4.set_title("$T$ at $t=%1.3f" %sol.t[t_index] + "$")

ax8.tricontourf(coords[:,0], coords[:,1], U[2*Nnodes:,t_index], cmap=color_map, levels=levelsC)
ax8.set_title("$C$ at $t=%1.3f" %sol.t[t_index] + "$")

ax5.tricontourf(coords[:,0], coords[:,1], U[:Nnodes,-1], cmap=color_map, levels=levelsP)
ax5.set_title("$\Psi$ at $t=%1.3f" %sol.t[-1] + "$")

ax6.tricontourf(coords[:,0], coords[:,1], U[Nnodes:2*Nnodes,-1], cmap=color_map, levels=levelsT)
ax6.set_title("$T$ at $t=%1.3f" %sol.t[-1] + "$")

ax9.tricontourf(coords[:,0], coords[:,1], U[2*Nnodes:,-1], cmap=color_map, levels=levelsC)
ax9.set_title("$C$ at $t=%1.3f" %sol.t[-1] + "$")

fig.suptitle("Solution with $N=%d" %coords.shape[0] +"$")

plt.show()