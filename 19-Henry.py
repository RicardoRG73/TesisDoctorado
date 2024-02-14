#%%
"""
Librerías necesarias
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-talk"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.6
mapa_de_color = "plasma"

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

from graficas import nodos_por_color
from GFDM import create_system_K_F

"""
Geometría
"""

geometria = cfg.Geometry()

# puntos
geometria.point([0,0])      # 0
geometria.point([2,0])      # 1
geometria.point([2,1])      # 2
geometria.point([0,1])      # 3

# líneas
left = 10
right = 11
top = 12
bottom = 13

geometria.line([0,1], marker=bottom)    # 0
geometria.line([1,2], marker=right)     # 1
geometria.line([2,3], marker=top)       # 2
geometria.line([3,0], marker=left)      # 3

# superficies
mat0 = 0
geometria.surface([0,1,2,3], marker=mat0)

# gráfica de la geometría
cfv.figure(fig_size=(8,5))
cfv.title('Geometría')
cfv.draw_geometry(geometria, font_size=16, draw_axis=True)


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
cfv.title('Malla $N=%d' %coords.shape[0] +'$')
cfv.draw_mesh(
    coords=coords,
    edof=edof,
    dofs_per_node=mesh.dofs_per_node,
    el_type=mesh.el_type,
    filled=True
)


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
bb = np.setdiff1d(bb, [0,1])
bt = np.asarray(bdofs[top]) - 1
bt = np.setdiff1d(bt, [2,3])
esquinas = np.array([0,1,2,3])

fronteras = (bl, br, bb, bt, esquinas)
Boundaries = np.hstack(fronteras)
interiores = np.setdiff1d(np.arange(coords.shape[0]) , np.hstack(fronteras))
etiquetas = (
    "Frontera Izquierda",
    "Frontera Derecha",
    "Frontera Inferior",
    "Frontera Superior",
    "Esquinas"
)

plt.figure(figsize=(16,8))
nodos_por_color(
    boundaries=fronteras,
    p=coords,
    labels=etiquetas,
    interior=interiores,
    label_interior="Nodos Interiores",
    alpha=1,
    nums=True,
    legend=False
)
plt.axis('equal')

#%%
"""
Parámetros del problema
"""
a = 0.2637
b = 0.1
k = lambda p: 1         # difusividad
f = lambda p: 0         # fuente

"""
Matrices D para la función de flujo \Psi
"""
# Definición de las condiciones de frontera y nodos interiores
Psit = lambda p: 1
Psib = lambda p: 0
Psil = lambda p: 0
Psir = lambda p: 0

materiales = {}
materiales["0"] = [k, interiores]

fronteas_dirichlet = {}
fronteas_dirichlet["top"] = [bt, Psit]
fronteas_dirichlet["bottom"] = [bb, Psib]
fronteas_dirichlet["esquinas_top"] = [[2,3], Psit]
fronteas_dirichlet["esquinas_bottom"] = [[0,1], Psib]

fronteras_neumann = {}
fronteras_neumann["left"] = [k, bl, Psil]
fronteras_neumann["right"] = [k, br, Psir]

# Ensamble de las matrices y vectores (D's y F's)
L2 = np.array([0,0,0,2,0,2])
D2psi, F2psi = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L2,
    source=f,
    materials=materiales,
    dirichlet_boundaries=fronteas_dirichlet,
    neumann_boundaries=fronteras_neumann
)
D2psi = D2psi.toarray()

Lx = np.array([0,1,0,0,0,0])
Dxpsi, Fxpsi = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Lx,
    source=f,
    materials=materiales,
    dirichlet_boundaries=fronteas_dirichlet,
    neumann_boundaries=fronteras_neumann
)
Dxpsi = Dxpsi.toarray()

Ly = np.array([0,0,1,0,0,0])
Dypsi, Fypsi = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Ly,
    source=f,
    materials=materiales,
    dirichlet_boundaries=fronteas_dirichlet,
    neumann_boundaries=fronteras_neumann
)
Dypsi = Dypsi.toarray()


"""
Matrices D para la concentración C
"""
# Definición de las condiciones de frontera
Cl = lambda p: 0
Cr = lambda p: 1
Ct = lambda p: 0
Cb = lambda p: 0


fronteas_dirichlet = {}
fronteas_dirichlet["left"] = [bl, Cl]
fronteas_dirichlet["right"] = [br, Cr]
fronteas_dirichlet["esquinas_left"] = [[0,3], Cl]
fronteas_dirichlet["esquinas_right"] = [[1,2], Cr]

fronteras_neumann = {}
fronteras_neumann["top"] = [k, bt, Ct]
fronteras_neumann["bottom"] = [k, bb, Cb]

# Ensamble de las matrices y vectores (D's y F's)
D2c, F2c = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L2,
    source=f,
    materials=materiales,
    dirichlet_boundaries=fronteas_dirichlet,
    neumann_boundaries=fronteras_neumann
)
D2c = D2c.toarray()

Dxc, Fxc = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Lx,
    source=f,
    materials=materiales,
    dirichlet_boundaries=fronteas_dirichlet,
    neumann_boundaries=fronteras_neumann
)
Dxc = Dxc.toarray()

Dyc, Fyc = create_system_K_F(
    p=coords,
    triangles=faces,
    L=Ly,
    source=f,
    materials=materiales,
    dirichlet_boundaries=fronteas_dirichlet,
    neumann_boundaries=fronteras_neumann
)
Dyc = Dyc.toarray()


"""
Ensamble del IVP
"""
#  modificaciones para no afectar las condiciones de frontera
Dxcpsi = Dxc.copy()
Fxcpsi = Fxc.copy()

Dxcpsi[Boundaries,:] = 0
Fxcpsi[Boundaries] = 0

Dypsic = Dypsi.copy()
Fypsic = Fypsi.copy()

Dypsic[Boundaries,:] = 0
Fypsic[Boundaries] = 0

Dxpsic = Dxpsi.copy()
Fxpsic = Fxpsi.copy()

Dxpsic[Boundaries,:] = 0
Fxpsic[Boundaries] = 0

# print("\n=============")
# print("Condition Number")
# print("--------------")
# print("||   Dxc    ||   Dyc   ||   Dxpsi   ||   Dypsi  ||")
# print("|| %1.2e || %1.2e || %1.2e || %1.2e ||" %(
#         np.linalg.cond(Dxc),np.linalg.cond(Dyc),np.linalg.cond(Dxpsi),np.linalg.cond(Dypsi)
#     )
# )
# print("=============")
# print("Max value in diag")
# print("--------------")
# print("||   Dxc    ||   Dyc   ||   Dxpsi   ||   Dypsi  ||")
# print("|| %1.2f || %1.2f || %1.2f || %1.2f ||" %(
#         np.max(np.diag(Dxc)),np.max(np.diag(Dyc)),np.max(np.diag(Dxpsi)),np.max(np.diag(Dypsi))
#     )
# )
# print("=============")
# print("Min value in diag")
# print("--------------")
# print("||   Dxc    ||   Dyc   ||   Dxpsi   ||   Dypsi  ||")
# print("|| %1.2f || %1.2f || %1.2f || %1.2f ||" %(
#         np.min(np.diag(Dxc)),np.min(np.diag(Dyc)),np.min(np.diag(Dxpsi)),np.min(np.diag(Dypsi))
#     )
# )
# print("=============\n")

# Parte lineal del sistema (matriz A)
N = coords.shape[0]
A = np.vstack((
    np.hstack((
        D2psi, -1/a * Dxcpsi
    )),
    np.hstack((
        np.zeros((N,N)), D2c
    ))
))

# Valores conocidos lineales (vector Fl)
Fl = np.hstack((
    - F2psi  +  1/a * Fxcpsi,
    - F2c
))

# Parte no lineal (Vector B)
def B(U):
    term1 = (Dypsic@U[:N]) * (Dxc@U[N:])
    term2 = (Dxpsic@U[:N]) * (Dyc@U[N:])
    vec2 = -1/b * (term1 - term2) 
    vec1 = np.zeros(N)
    vec = np.hstack((vec1, vec2))
    return vec

# Valores conocidos no lineales (vector Fn)
Fn = np.hstack((
    np.zeros(N),
    - 1/b * (Fypsic*Fxc - Fxpsic*Fyc)
))

# Acoplamiento del lado derecho en la función anónima fun

fun = lambda t,U: A@U + Fl + B(U) + Fn

# Condiciones iniciales
C0 = np.zeros(N)
Psi0 = np.zeros(N)
for i in bl:
    C0[i] = Cl(coords[i,:])
    Psi0[i] = Psil(coords[i,:])
for i in br:
    C0[i] = Cr(coords[i,:])
    Psi0[i] = Psir(coords[i,:])
for i in bt:
    C0[i] = Ct(coords[i,:])
    Psi0[i] = Psit(coords[i,:])
for i in bb:
    C0[i] = Cb(coords[i,:])
    Psi0[i] = Psib(coords[i,:])
i = 0
C0[i] = Cl(coords[i,:])
Psi0[i] = Psib(coords[i,:])
i = 1
C0[i] = Cr(coords[i,:])
Psi0[i] = Psib(coords[i,:])
i = 2
C0[i] = Cr(coords[i,:])
Psi0[i] = Psit(coords[i,:])
i = 3
C0[i] = Cl(coords[i,:])
Psi0[i] = Psit(coords[i,:])

fig = plt.figure()
ax = plt.subplot(1, 2, 1, projection="3d")
ax.plot_trisurf(coords[:,0],coords[:,1],Psi0, cmap=mapa_de_color, edgecolor="k")
ax.set_title("$\Psi_0$")
ax2 = plt.subplot(1, 2, 2, projection="3d")
ax2.plot_trisurf(coords[:,0],coords[:,1],C0, cmap=mapa_de_color, edgecolor="k")
ax2.set_title("$C_0$")

U0 = np.hstack((Psi0, C0))

# Solución del IVP
tspan = [0,0.21]             # intervalo de solución
sol = solve_ivp(fun, tspan, U0, method="RK45")

U = sol.y

# Gráfica de la solución en el tiempo final
plt.style.use("paper3dplot.mplstyle")
fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection="3d")
ax.plot_trisurf(
    coords[:,0],
    coords[:,1],
    U[:N,-1],
    cmap=mapa_de_color,
    linewidth=1,
    antialiased=False
)
ax.view_init(azim=-120, elev=50)
plt.title("$\Psi$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection="3d")
ax.plot_trisurf(
    coords[:,0],
    coords[:,1],
    U[N:,-1],
    cmap=mapa_de_color,
    linewidth=1,
    antialiased=False
)
ax.view_init(azim=-120, elev=50)
plt.title("$C$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")


plt.style.use(["default", "seaborn-v0_8", "seaborn-v0_8-talk"])

plt.figure()
plt.tricontourf(
    coords[:,0],
    coords[:,1],
    U[:N,-1],
    levels=20,
    cmap=mapa_de_color
)
plt.axis("equal")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title("$\Psi$")
plt.colorbar()

plt.figure()
plt.tricontourf(
    coords[:,0],
    coords[:,1],
    U[N:,-1],
    levels=20,
    cmap=mapa_de_color
)
plt.axis("equal")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title("$C$")
plt.colorbar()

# %%

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

ax1.tricontourf(coords[:,0], coords[:,1], U[:N,0], cmap=mapa_de_color, levels=levelsP)
ax1.set_title("$\Psi$ at $t=%1.3f" %sol.t[0] + "$")

# ax2 = plt.subplot(322)
ax2.tricontourf(coords[:,0], coords[:,1], U[N:,0], cmap=mapa_de_color, levels=levelsC)
ax2.set_title("$C$ at $t=%1.3f" %sol.t[0] + "$")


t_index = sol.t.shape[0]//4
# ax3 = plt.subplot(323, sharex=True)
ax3.tricontourf(coords[:,0], coords[:,1], U[:N,t_index], cmap=mapa_de_color, levels=levelsP)
ax3.set_title("$\Psi$ at $t=%1.3f" %sol.t[t_index] + "$")

# ax4 = plt.subplot(324)
ax4.tricontourf(coords[:,0], coords[:,1], U[N:,t_index], cmap=mapa_de_color, levels=levelsC)
ax4.set_title("$C$ at $t=%1.3f" %sol.t[t_index] + "$")

# ax5 = plt.subplot(325)
ax5.tricontourf(coords[:,0], coords[:,1], U[:N,-1], cmap=mapa_de_color, levels=levelsP)
ax5.set_title("$\Psi$ at $t=%1.3f" %sol.t[-1] + "$")

# ax6 = plt.subplot(326)
ax6.tricontourf(coords[:,0], coords[:,1], U[N:,-1], cmap=mapa_de_color, levels=levelsC)
ax6.set_title("$C$ at $t=%1.3f" %sol.t[-1] + "$")

fig.suptitle("Solution with $N=%d" %coords.shape[0] +"$")

plt.show()