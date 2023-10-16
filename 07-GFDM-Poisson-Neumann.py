#%%
""" Librerías necesarias """
import numpy as np
from scipy import sparse as sp

# calfem-python
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

import matplotlib.pyplot as plt
plt.style.use(['seaborn-v0_8','paper.mplstyle'])
plt.rcParams['text.usetex'] = False
mapa_de_color = "plasma"

""" Objeto geometría """
geometria = cfg.Geometry()

# puntos
geometria.point([0,0])      # 0
geometria.point([55,0])     # 1
geometria.point([75,0])     # 2
geometria.point([100,0])    # 3
geometria.point([130,0])    # 4
geometria.point([70,30])    # 5
geometria.point([60,30])    # 6
geometria.point([50,25])    # 7

# líneas
Dirich_left = 10
Dirich_right = 11
Neumann_top = 12
Neumann_bottom = 13

geometria.line([0,1], marker=Neumann_bottom)       # 0
geometria.line([1,2], marker=Neumann_bottom)       # 1
geometria.line([2,3], marker=Neumann_bottom)       # 2
geometria.line([3,4], marker=Dirich_right)    # 3
geometria.line([4,5], marker=Neumann_top)       # 4
geometria.line([5,6], marker=Neumann_top)       # 5
geometria.line([6,7], marker=Neumann_top)       # 6
geometria.line([7,0], marker=Dirich_left)    # 7

# superficies
mat0 = 100
geometria.surface([0,1,2,3,4,5,6,7], marker=mat0)

# gráfica de la geometría
cfv.figure(fig_size=(16,5))
cfv.title('Geometría', fontdict={"fontsize": 32})
cfv.draw_geometry(geometria, font_size=16, draw_axis=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

""" Creación del objeto malla usando el objeto geometría """
mesh = cfm.GmshMesh(geometria)

mesh.el_type = 2                            # type of element: 2 = triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 2

coords, edof, dofs, bdofs, elementmarkers = mesh.create()   # create the geometry
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

# gráfica de la malla
cfv.figure(fig_size=(16,5))
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


""" Identificación de las diferentes fronteras """
BDirl = np.asarray(bdofs[Dirich_left]) - 1
BDirr = np.asarray(bdofs[Dirich_right]) - 1
BNeub = np.asarray(bdofs[Neumann_bottom]) - 1
BNeub = np.setdiff1d(BNeub, [0,3])
BNeut = np.asarray(bdofs[Neumann_top]) - 1
BNeut = np.setdiff1d(BNeut, [4,7])

fronteras = (BDirl, BDirr, BNeub, BNeut)
interiores = np.setdiff1d(np.arange(coords.shape[0]) , np.hstack(fronteras))
etiquetas = (
    "Dirichlet Izquierda",
    "Dirichlet Derecha",
    "Dirichlet Inferior",
    "Dirichlet Superior"
)

from graficas import nodos_por_color
plt.figure(figsize=(30,8))
nodos_por_color(
    boundaries=fronteras,
    p=coords,
    labels=etiquetas,
    interior=interiores,
    label_interior="Nodos Interiores",
    alpha=1
)
plt.axis('equal')

""" Parámetros del problema """
L = np.array([0,0,0,2,0,2])
k0 = lambda p: 1
f = lambda p:  0 #-0.001
ul = lambda p: 1 + 0.25 * np.sin(np.pi * p[1]/25)
ur = lambda p: 0
ub = lambda p: 0
ut = lambda p: 0

materials = {}
materials["0"] = [k0, interiores]

dirichlet_boundaries = {}
dirichlet_boundaries["left"] = [BDirl, ul]
dirichlet_boundaries["right"] = [BDirr, ur]

neumann_boundaries = {}
neumann_boundaries["bottom"] = [k0, BNeub, ub]
neumann_boundaries["top"] = [k0, BNeut, ut]



from GFDM import create_system_K_F
K, F = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L,
    source=f,
    materials=materials,
    neumann_boundaries=neumann_boundaries,
    dirichlet_boundaries=dirichlet_boundaries,
    interfaces={}
)

from scipy.sparse.linalg import spsolve
U = spsolve(K,F)

fig = plt.figure(figsize=(16,8))
plt.tricontourf(
    coords[:,0],
    coords[:,1],
    U,
    levels=20,
    cmap=mapa_de_color,
)
plt.colorbar()
plt.axis('equal')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title("Solución (Contorno)")

plt.style.use("paper3dplot.mplstyle")
fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection="3d")
ax.plot_trisurf(
    coords[:,0],
    coords[:,1],
    U,
    cmap=mapa_de_color,
    linewidth=1,
    antialiased=False
)
ax.view_init(azim=-60, elev=50)

plt.title("Solución (3D)")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$u(x,y)$")

plt.show()
# %%
