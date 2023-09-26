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
geometria.point([3,0])     # 1
geometria.point([3,1])     # 2
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

from graficas import nodos_por_color
plt.figure(figsize=(16,8))
nodos_por_color(
    boundaries=fronteras,
    p=coords,
    labels=etiquetas,
    interior=interiores,
    label_interior="Nodos Interiores",
    alpha=0.5,
    nums=True,
    loc='center'
)
plt.axis('equal')


plt.show()
# %%
