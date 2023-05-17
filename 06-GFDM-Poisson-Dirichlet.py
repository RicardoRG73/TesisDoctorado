""" Librerías necesarias """
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-v0_8','paper.mplstyle'])
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

geometria.line([0,1], marker=Neuman0)
geometria.line([1,2], marker=Neuman0)
geometria.line([2,3], marker=Neuman0)
geometria.line([3,4], marker=Dirichlet1)
geometria.line([4,5], marker=Neuman1)
geometria.line([5,6], marker=Neuman1)
geometria.line([6,7], marker=Neuman1)
geometria.line([7,0], marker=Dirichlet1)

# gráfica de la geometría
cfv.figure(fig_size=(16,5))
cfv.title('Geometría')
cfv.draw_geometry(geometria, font_size=26)

""" Creating mesh from geometry object """
# mesh = cfm.GmshMesh(geometria)

# mesh.el_type = 2                            # type of element: 2 = triangle
# mesh.dofs_per_node = 1
# mesh.el_size_factor = 1

# coords, edof, dofs, bdofs, elementmarkers = mesh.create()   # create the geometry
# verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
#     coords,
#     edof,
#     mesh.dofs_per_node,
#     mesh.el_type
# )

# # mesh plot
# cfv.figure(fig_size=(8,4))
# cfv.title('Malla')
# cfv.draw_mesh(
#     coords=coords,
#     edof=edof,
#     dofs_per_node=mesh.dofs_per_node,
#     el_type=mesh.el_type,
#     filled=True
# )

plt.show()