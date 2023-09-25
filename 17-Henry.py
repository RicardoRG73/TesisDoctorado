#%%

"""
Librerías necesarias
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-v0_8','paper.mplstyle'])
plt.rcParams['text.usetex'] = False
mapa_de_color = "plasma"

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv


"""
Geometría
"""

geometria = cfg.Geometry()

# puntos
geometria.point([0,0])      # 0
geometria.point([2,0])     # 1
geometria.point([2,1])     # 2
geometria.point([0,1])    # 3

# líneas
left = 10
right = 11
top = 12
bottom = 13

geometria.line([0,1], marker=bottom)       # 0
geometria.line([1,2], marker=right)       # 1
geometria.line([2,3], marker=top)       # 2
geometria.line([3,0], marker=left)    # 3

# superficies
mat0 = 0
geometria.surface([0,1,2,3], marker=mat0)

# gráfica de la geometría
cfv.figure(fig_size=(8,5))
cfv.title('Geometría', fontdict={"fontsize": 32})
cfv.draw_geometry(geometria, font_size=16, draw_axis=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


"""
Malla
"""

mesh = cfm.GmshMesh(geometria)

mesh.el_type = 2                            # type of element: 2 = triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.1

coords, edof, dofs, bdofs, elementmarkers = mesh.create()   # create the geometry
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

# gráfica de la malla
cfv.figure(fig_size=(8,5))
cfv.title('Malla', fontdict={"fontsize": 32})
cfv.draw_mesh(
    coords=coords,
    edof=edof,
    dofs_per_node=mesh.dofs_per_node,
    el_type=mesh.el_type,
    filled=True
)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


"""
Identificación de los nodos de frontera
bl: left
br: right
bb: bottom
bt: top
"""

bl = np.asarray(bdofs[left]) - 1
bl = np.setdiff1d(bl, [0,3])
br = np.asarray(bdofs[right]) - 1
br = np.setdiff1d(br, [1,2])
bb = np.asarray(bdofs[bottom]) - 1
bt = np.asarray(bdofs[top]) - 1

fronteras = (bl, br, bb, bt)
Boundaries = np.hstack(fronteras)
interiores = np.setdiff1d(np.arange(coords.shape[0]) , np.hstack(fronteras))
etiquetas = (
    "Frontera Izquierda",
    "Frontera Derecha",
    "Frontera Inferior",
    "Frontera Superior"
)

from graficas import nodos_por_color
plt.figure(figsize=(16,8))
nodos_por_color(
    boundaries=fronteras,
    p=coords,
    labels=etiquetas,
    interior=interiores,
    label_interior="Nodos Interiores",
    alpha=1,
    nums=True
)
plt.axis('equal')


"""
Parámetros del problema
"""
a = 0.2637
b = 0.1
k = lambda p: 1
f = lambda p:  0

"""
Matrices D para la función de flujo \Psi
"""
# Condicinoes de frontera
Psil = lambda p: 0      # Neumann
Psir = lambda p: 0      # Neumann
Psib = lambda p: 0      # Dirichlet
Psit = lambda p: 1      # Dirichlet

materials = {}
materials["0"] = [k, interiores]

neumann_boundaries = {}
neumann_boundaries["left"] = [k, bl, Psil]
neumann_boundaries["right"] = [k, br, Psir]

dirichlet_boundaries = {}
dirichlet_boundaries["bottom"] = [bb, Psib]
dirichlet_boundaries["top"] = [bt, Psit]

# Vectores de coeficientes L
Lx = np.array([0,1,0,0,0,0])
Ly = np.array([0,0,1,0,0,0])
L2 = np.array([0,0,0,2,0,2])

# Ensamble de las matrices D
from GFDM import create_system_K_F
Dx_Psi, Fx_Psi, _ = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Lx,
    source=f,
    materials=materials,
    neumann_boundaries=neumann_boundaries,
    dirichlet_boundaries=dirichlet_boundaries,
    interfaces={}
)
Dy_Psi, Fy_Psi, _ = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Ly,
    source=f,
    materials=materials,
    neumann_boundaries=neumann_boundaries,
    dirichlet_boundaries=dirichlet_boundaries,
    interfaces={}
)
D2_Psi, F2_Psi, _ = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L2,
    source=f,
    materials=materials,
    neumann_boundaries=neumann_boundaries,
    dirichlet_boundaries=dirichlet_boundaries,
    interfaces={}
)


"""
Matrices D para la concentración C
"""
# bl = np.asarray(bdofs[left]) - 1
# br = np.asarray(bdofs[right]) - 1
# bb = np.asarray(bdofs[bottom]) - 1
# bb = np.setdiff1d(bb, [0,1])
# bt = np.asarray(bdofs[top]) - 1
# bt = np.setdiff1d(bt, [2,3])

# Condicinoes de frontera
Cl = lambda p: 0      # Dirichlet
Cr = lambda p: 1      # Dirichlet
Cb = lambda p: 0      # Neumann
Ct = lambda p: 0      # Neumann

materials = {}
materials["0"] = [k, interiores]

dirichlet_boundaries = {}
dirichlet_boundaries["left"] = [bl, Cl]
dirichlet_boundaries["right"] = [br, Cr]

neumann_boundaries = {}
neumann_boundaries["bottom"] = [k, bb, Cb]
neumann_boundaries["top"] = [k, bt, Ct]

# Ensamble de las matrices D
Dx_C, Fx_C, _ = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Lx,
    source=f,
    materials=materials,
    neumann_boundaries=neumann_boundaries,
    dirichlet_boundaries=dirichlet_boundaries,
    interfaces={}
)
Dy_C, Fy_C, _ = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Ly,
    source=f,
    materials=materials,
    neumann_boundaries=neumann_boundaries,
    dirichlet_boundaries=dirichlet_boundaries,
    interfaces={}
)
D2_C, F2_C, _ = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L2,
    source=f,
    materials=materials,
    neumann_boundaries=neumann_boundaries,
    dirichlet_boundaries=dirichlet_boundaries,
    interfaces={}
)


# Conversión de sparse matrix a array
Dx_Psi = Dx_Psi.toarray()
Fx_Psi = Fx_Psi.toarray()[:,0]
Dy_Psi = Dy_Psi.toarray()
Fy_Psi = Fy_Psi.toarray()[:,0]
D2_Psi = D2_Psi.toarray()
F2_Psi = F2_Psi.toarray()[:,0]
Dx_C = Dx_C.toarray()
Fx_C = Fx_C.toarray()[:,0]
Dy_C = Dy_C.toarray()
Fy_C = Fy_C.toarray()[:,0]
D2_C = D2_C.toarray()
F2_C = F2_C.toarray()[:,0]


""" 
Ensamble del problema de valor inicial
du/dt = fun(u,t)
"""
# Modificaciones para no afectar las fronteras
Dx_C_Psi = Dx_C.copy()
Dx_C_Psi[Boundaries,:] = 0

Fx_C_Psi = Fx_C.copy()
Fx_C_Psi[Boundaries] = 0

Dy_Psi_C = Dy_Psi.copy()
Dy_Psi_C[Boundaries,:] = 0

Fy_Psi_C = Fy_Psi.copy()
Fy_Psi_C[Boundaries] = 0

Dx_Psi_C = Dx_Psi
Dx_Psi_C[Boundaries,:] = 0

Fx_Psi_C = Fx_Psi.copy()
Fx_Psi_C[Boundaries] = 0

# Parte lineal del sistema: matriz A
N = coords.shape[0]
A = np.vstack((
    np.hstack((D2_Psi, -1/a*Dx_C_Psi)),
    np.hstack((np.zeros((N,N)), D2_C))
))

# Parte no lineal del sistema: vector B
def B(U):
    U1 = U[0:N]
    U2 = U[N:2*N]
    term1 = (Dy_Psi_C @ U1) * (Dx_C @ U2)
    term2 = (Dx_Psi_C @ U1) * (Dy_C @ U2)
    B = np.hstack((
        np.zeros(N),
        -1/b * (term1 - term2)
    ))
    return B

# valores conocidos del sistema: F
F = np.hstack((
    -F2_Psi + 1/a * Fx_C_Psi,
    -F2_C-1/b*( Fy_Psi_C * Fx_C - Fx_Psi_C * Fy_C )
))

fun = lambda t,U: A@U + B(U) + F

Psi0 = np.zeros(N)
Psi0[bl] = Psil(coords[bl])
Psi0[br] = Psir(coords[br])
Psi0[bb] = Psib(coords[bb])
Psi0[bt] = Psit(coords[bt])

C0 = np.zeros(N)
C0[bl] = Cl(coords[bl])
C0[br] = Cr(coords[br])
C0[bb] = Cb(coords[bb])
C0[bt] = Ct(coords[bt])

U0 = np.hstack((Psi0, C0))

from scipy.integrate import solve_ivp
sol = solve_ivp(fun=fun, t_span=[0,1.27], y0=U0)

U = sol.y
Psi = U[0:N,:]
C = U[N:2*N,:]

#%%
index = -1
fig = plt.figure(layout='constrained', figsize=(16,5))
subfigs = fig.subfigures(1,2, wspace=0.07)
ax0 = subfigs[0].subplots(1,1)
plot0 = ax0.tricontourf(
    coords[:,0],
    coords[:,1],
    Psi[:,index],
    levels=20,
    cmap=mapa_de_color,
)
# ax0.tricontour(
#     coords[:,0],
#     coords[:,1],
#     Psi[:,index],
#     levels=20,
#     color="k"
# )
# ax0.quiver(
#     coords[:,0],
#     coords[:,1],
#     Dy_Psi@Psi[:,index],
#     -Dx_Psi@Psi[:,index],
#     color="k"
# )
ax0.axis("equal")
ax0.set_xlabel('$x$')
ax0.set_ylabel('$y$')
subfigs[0].suptitle("$\Psi$")

ax1 = subfigs[1].subplots(1,1)
plot1 = ax1.tricontourf(
    coords[:,0],
    coords[:,1],
    C[:,index],
    levels=20,
    cmap=mapa_de_color,
)
ax1.axis("equal")
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
subfigs[1].suptitle("$C$")

fig.colorbar(plot0)
fig.colorbar(plot1)
fig.suptitle("$t=$"+str(np.round(sol.t[index],4)))
# %%

plt.style.use("paper3dplot.mplstyle")
fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection="3d")
ax.plot_trisurf(
    coords[:,0],
    coords[:,1],
    C[:,-1],
    cmap=mapa_de_color,
    linewidth=1,
    edgecolor='k',
    antialiased=False
)
ax.view_init(azim=120, elev=30)
plt.show()
# %%
