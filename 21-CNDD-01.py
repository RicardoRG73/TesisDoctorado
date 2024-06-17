# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:19:29 2024

@author: ricardo
"""
save_figures = False
save_solution = False
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
th = 30 * Degrees

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
# | el_size_factor |   N   |
# |    0.02        |   976 |
# |    0.013       |  2341 |
# |    0.01        |  3751 |
# =============================================================================
mesh = cfm.GmshMesh(
    geometry,
    el_size_factor=0.03
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
if save_figures:
    plt.savefig("figuras/CNDD/mallaN=%d.pdf" %coords.shape[0])

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

plt.figure(figsize=(8,7))
nodos_por_color(
    boundaries=boundaries_tuple,
    p=coords,
    labels=etiquetas,
    interior=interior,
    label_interior="Interior",
    alpha=1,
    nums=False,
    legend=True,
    loc="best",
    s=10,
)
plt.axis('equal')
if save_figures:
    plt.savefig("figuras/CNDD/nodosN=%d.pdf" %coords.shape[0])

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

tfinal = 0.13

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
# D2P = D2P.toarray()

DxP, FxP = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Lx,
    source=f,
    materials=materials,
    dirichlet_boundaries=PDirich,
    neumann_boundaries=PNeu
)
# DxP.toarray()

DyP, FyP = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Ly,
    source=f,
    materials=materials,
    dirichlet_boundaries=PDirich,
    neumann_boundaries=PNeu
)
# DyP.toarray()

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
# D2C = D2C.toarray()

DxC, FxC = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Lx,
    source=f,
    materials=materials,
    dirichlet_boundaries=CDirich,
    neumann_boundaries=CNeu
)
# DxC = DxC.toarray()

DyC, FyC = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Ly,
    source=f,
    materials=materials,
    dirichlet_boundaries=CDirich,
    neumann_boundaries=CNeu
)
# DyC = DyC.toarray()

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
# D2T = D2T.toarray()

DxT, FxT = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Lx,
    source=f,
    materials=materials,
    dirichlet_boundaries=TDirich,
    neumann_boundaries=TNeu
)
# DxT = DxT.toarray()

DyT, FyT = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Ly,
    source=f,
    materials=materials,
    dirichlet_boundaries=TDirich,
    neumann_boundaries=TNeu
)
# DyT = DyT.toarray()

#%%
# =============================================================================
# Problem coupling
# =============================================================================
import scipy.sparse as sp

zeros_mat = sp.lil_matrix(np.zeros((Nnodes,Nnodes)))
zeros_vec = np.zeros(Nnodes)

DxC_noBound = DxC.copy()
DxC_noBound = sp.lil_matrix(DxC_noBound)
DxC_noBound[Boundaries,:] = 0
FxC_noBound = FxC.copy()
FxC_noBound[Boundaries] = 0

DxT_noBound = DxT.copy()
DxT_noBound = sp.lil_matrix(DxT_noBound)
DxT_noBound[Boundaries,:] = 0
FxT_noBound = FxT.copy()
FxT_noBound[Boundaries] = 0

DyP_noBound = DyP.copy()
DyP_noBound = sp.lil_matrix(DyP_noBound)
DyP_noBound[Boundaries,:] = 0
FyP_noBound = FyP.copy()
FyP_noBound[Boundaries] = 0

DxP_noBound = DxP.copy()
DxP_noBound = sp.lil_matrix(DxP_noBound)
DxP_noBound[Boundaries,:] = 0
FxP_noBound = FxP.copy()
FxP_noBound[Boundaries] = 0

# Linear
A = sp.vstack((
    sp.hstack((D2P, Ra*DxT_noBound, (Ra*N)*DxC_noBound)),
    sp.hstack((zeros_mat, D2T, zeros_mat)),
    sp.hstack((zeros_mat, zeros_mat, D2C))
))

# Non-Linear
def B(U):
    Tterm1 = (
        DyP_noBound @ U[:Nnodes] - FyP_noBound
    ) * (
        DxT @ U[Nnodes:2*Nnodes] - FxT
    )
    Tterm2 = (
        DxP_noBound @ U[:Nnodes] - FxP_noBound
    ) * (
        DyT @ U[Nnodes:2*Nnodes] - FyT
    )
    
    Cterm1 = (
        DyP_noBound @ U[:Nnodes] - FyP_noBound
    ) * (
        DxC @ U[2*Nnodes:] - FxC
    )
    Cterm2 = (
        DxP_noBound @ U[:Nnodes] - FxP_noBound
    ) * (
        DyC @ U[2*Nnodes:] - FyC
    )
    
    vec = np.hstack((
        - F2P - Ra*FxT_noBound - (Ra*N)*FxC_noBound,
        - F2T - Tterm1 + Tterm2,
        - F2C + Le * (- Cterm1 + Cterm2)
    ))
    return vec

def rhs(t,U):
    return A @ U + B(U)

#%%
# =============================================================================
# Condition numbers
# =============================================================================
N = coords.shape[0]
print("N = ", N)
print("\n=============================================================")
print("Condition Number")
print("---------------------------------------------------------------")
print("   DxP",
      "DyP",
      "D2P",
      "DxT",
      "DyT",
      "D2T",
      "DxC",
      "DyC",
      "D2C",
      sep="   ||    "
)
print("%1.2e || %1.2e || %1.2e || %1.2e || %1.2e || %1.2e || %1.2e || %1.2e || %1.2e " %(
    np.linalg.cond(DxP.toarray()),
    np.linalg.cond(DyP.toarray()),
    np.linalg.cond(D2P.toarray()),
    np.linalg.cond(DxT.toarray()),
    np.linalg.cond(DyT.toarray()),
    np.linalg.cond(D2T.toarray()),
    np.linalg.cond(DxC.toarray()),
    np.linalg.cond(DyC.toarray()),
    np.linalg.cond(D2C.toarray()),
))
print("==============================================================\n")


#%%
# =============================================================================
# Solving IVP with RKF45
# =============================================================================
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

if save_solution:
    import pickle
    path = "figuras/CNDD/solN%d.pkl" %coords.shape[0]
    pickle.dump([U, times, coords, DxP, DyP], open(path, "wb"))

# %%
# =============================================================================
# Plotting solution
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

lines = ax1.tricontourf(coords[:,0], coords[:,1], U[:Nnodes,0], cmap=color_map, levels=levelsP)
ax1.set_title("$\Psi$ at $t=%1.3f" %sol.t[0] + "$")
fig.colorbar(lines)

lines = ax2.tricontourf(coords[:,0], coords[:,1], U[Nnodes:2*Nnodes,0], cmap=color_map, levels=levelsT)
ax2.set_title("$T$ at $t=%1.3f" %sol.t[0] + "$")
fig.colorbar(lines)

lines = ax7.tricontourf(coords[:,0], coords[:,1], U[2*Nnodes:,0], cmap=color_map, levels=levelsC)
ax7.set_title("$C$ at $t=%1.3f" %sol.t[0] + "$")
fig.colorbar(lines)

t_index = sol.t.shape[0]//8
lines = ax3.tricontourf(coords[:,0], coords[:,1], U[:Nnodes,t_index], cmap=color_map, levels=levelsP)
# ax3.streamplot(coords[:,0], coords[:,1], DyP@U[:Nnodes,t_index], -DxP@U[:Nnodes,t_index])
ax3.set_title("$\Psi$ at $t=%1.3f" %sol.t[t_index] + "$")
fig.colorbar(lines)

lines = ax4.tricontourf(coords[:,0], coords[:,1], U[Nnodes:2*Nnodes,t_index], cmap=color_map, levels=levelsT)
ax4.set_title("$T$ at $t=%1.3f" %sol.t[t_index] + "$")
fig.colorbar(lines)

lines = ax8.tricontourf(coords[:,0], coords[:,1], U[2*Nnodes:,t_index], cmap=color_map, levels=levelsC)
ax8.set_title("$C$ at $t=%1.3f" %sol.t[t_index] + "$")
fig.colorbar(lines)

lines = ax5.tricontourf(coords[:,0], coords[:,1], U[:Nnodes,-1], cmap=color_map, levels=levelsP)
ax5.set_title("$\Psi$ at $t=%1.3f" %sol.t[-1] + "$")
fig.colorbar(lines)

lines = ax6.tricontourf(coords[:,0], coords[:,1], U[Nnodes:2*Nnodes,-1], cmap=color_map, levels=levelsT)
ax6.set_title("$T$ at $t=%1.3f" %sol.t[-1] + "$")
fig.colorbar(lines)

lines = ax9.tricontourf(coords[:,0], coords[:,1], U[2*Nnodes:,-1], cmap=color_map, levels=levelsC)
ax9.set_title("$C$ at $t=%1.3f" %sol.t[-1] + "$")
fig.colorbar(lines)

fig.suptitle("Solution with $N=%d$" %coords.shape[0], fontsize=20)

plt.show()