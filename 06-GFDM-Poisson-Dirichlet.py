""" Librerías necesarias """
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-v0_8','paper2.mplstyle'])
mapa_de_color = "plasma"

# calfem-python
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

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
Neuman0 = 10
Dirichlet0 = 11
Neuman1 = 12
Dirichlet1 = 13

geometria.line([0,1], marker=Neuman0)       # 0
geometria.line([1,2], marker=Neuman0)       # 1
geometria.line([2,3], marker=Neuman0)       # 2
geometria.line([3,4], marker=Dirichlet1)    # 3
geometria.line([4,5], marker=Neuman1)       # 4
geometria.line([5,6], marker=Neuman1)       # 5
geometria.line([6,7], marker=Neuman1)       # 6
geometria.line([7,0], marker=Dirichlet0)    # 7

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
BDir0 = np.asarray(bdofs[Dirichlet0]) - 1
BDir1 = np.asarray(bdofs[Dirichlet1]) - 1
BNeu0 = np.asarray(bdofs[Neuman0]) - 1
BNeu0 = np.setdiff1d(BNeu0, [0,3])
BNeu1 = np.asarray(bdofs[Neuman1]) - 1
BNeu1 = np.setdiff1d(BNeu1, [4,7])

plt.figure(figsize=(15,5))
fronteras = (BDir0, BDir1, BNeu0, BNeu1)
interiores = np.setdiff1d(np.arange(coords.shape[0]) , np.hstack(fronteras))
etiquetas = (
    "Dirichlet Izquierda",
    "Dirichlet Derecha",
    "Dirichlet Inferior",
    "Dirichlet Superior"
)

from graficas import nodos_por_color
nodos_por_color(
    boundaries=fronteras,
    p=coords,
    labels=etiquetas,
    interior=interiores,
    label_interior="Nodos Interiores"
    )


plt.show()