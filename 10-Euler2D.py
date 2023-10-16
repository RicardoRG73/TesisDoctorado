#%%

"""
Librerías necesarias
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-v0_8','seaborn-v0_8-talk'])
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.shadow'] = True
plt.rcParams['font.family'] = "serif"
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
cfv.title('Geometría')
cfv.draw_geometry(geometria, draw_axis=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


"""
Malla
"""

mesh = cfm.GmshMesh(geometria)

mesh.el_type = 2                            # type of element: 2 = triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.2

coords, edof, dofs, bdofs, elementmarkers = mesh.create()   # create the geometry
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

# gráfica de la malla
cfv.figure(fig_size=(8,5))
cfv.title('Malla')
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
interiores = np.setdiff1d(np.arange(coords.shape[0]) , Boundaries)
etiquetas = (
    "Frontera Izquierda",
    "Frontera Derecha",
    "Frontera Inferior",
    "Frontera Superior",
    "Esquinas"
)

from graficas import nodos_por_color
plt.figure()
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

# %%
"""
Parámetros del problema
"""
k = lambda p: 1
f = lambda p:  0
L = np.array([0,0,0,2,0,2])

# Condicinoes de frontera
ul = lambda p: 1 # 1 + 0.2*np.sin(np.pi*p[1])
ur = lambda p: 0
ub = lambda p: 0
ut = lambda p: 0

# Ensamble de las condiciones en los diferentes diccionarios
materials = {}
materials["0"] = [k, interiores]

neumann_boundaries = {}
neumann_boundaries["bottom"] = [k, bb, ub]
neumann_boundaries["top"] = [k, bt, ut]

dirichlet_boundaries = {}
dirichlet_boundaries["left"] = [bl, ul]
dirichlet_boundaries["right"] = [br, ur]
dirichlet_boundaries["esquinas"] = [esquinas, lambda p: 1-p[0]/3]

# Ensamble y solución del sistema
from GFDM import create_system_K_F
D2, F2 = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L,
    source=f,
    materials=materials,
    neumann_boundaries=neumann_boundaries,
    dirichlet_boundaries=dirichlet_boundaries,
    interfaces={}
)
F2 = F2.toarray()[:,0]

dt = 0.0001
T = 1.6
pasos = int(np.round(T/dt,0))
t = np.linspace(0,T,pasos)

U = np.zeros((F2.shape[0], pasos))
U0 = np.zeros(F2.shape[0])
for i in bl:
    U0[i] = ul(coords[i])
for i in br:
    U0[i] = ur(coords[i])
U0[bb] = 0 # 1 - coords[bb,0]/3
U0[bt] = 0 # 1 - coords[bt,0]/3
U0[esquinas] = 1 - coords[esquinas,0]/3

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_trisurf(coords[:,0], coords[:,1], U0, cmap=mapa_de_color)
plt.title("Condición inicial $U_0$")
plt.xlabel("$x$")
plt.ylabel("$y$")

U[:,0] = U0
I = np.eye(D2.shape[0])
A = I + dt*D2
for i in range(pasos-1):
    U[:,i+1] = A@U[:,i] - dt*F2


#%%
index = -1
fig = plt.figure(layout='constrained', figsize=(16,5))
subfigs = fig.subfigures(1,2, wspace=0)
ax0 = subfigs[0].subplots(1,1)
plot0 = ax0.tricontourf(
    coords[:,0],
    coords[:,1],
    U[:,index],
    levels=20,
    cmap=mapa_de_color
)
ax0.axis("equal")
ax0.set_xlabel('$x$')
ax0.set_ylabel('$y$')
subfigs[0].suptitle("Contourf")

ax1 = subfigs[1].add_subplot(111, projection="3d")
plot1 = ax1.plot_trisurf(
    coords[:,0],
    coords[:,1],
    U[:,index],
    triangles=faces,
    cmap=mapa_de_color
)
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
subfigs[0].suptitle("3D")


fig.colorbar(plot0)
fig.colorbar(plot1)
fig.suptitle("Solución $U$ en $t=$"+str(np.round(t[index],4)))
# %%
guarda_figuras = False
indices = [0,50,200,1000,5000,15999]
if guarda_figuras:
    for i in indices:
        fig = plt.figure(layout='constrained', figsize=(16,5))
        subfigs = fig.subfigures(1,2, wspace=0)
        ax0 = subfigs[0].subplots(1,1)
        plt.style.context("paper3dplot.mplstyle")
        plot0 = ax0.tricontourf(
            coords[:,0],
            coords[:,1],
            U[:,i],
            levels=20,
            cmap=mapa_de_color
        )
        ax0.axis("equal")
        ax0.set_xlabel('$x$')
        ax0.set_ylabel('$y$')
        subfigs[0].suptitle("Contourf")

        ax1 = subfigs[1].add_subplot(111, projection="3d")
        plt.style.context("paper.mplstyle")
        plot1 = ax1.plot_trisurf(
            coords[:,0],
            coords[:,1],
            U[:,i],
            triangles=faces,
            cmap=mapa_de_color
        )
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$y$')
        subfigs[1].suptitle("3D")


        fig.colorbar(plot0)
        fig.colorbar(plot1)
        fig.suptitle("Solución $U$ en $t=$ %1.3f" %t[i])
        print("guardando figura con índice i=",i)
        fig.savefig("figuras/10-Euler2D-t-%1.3f.png" %t[i])

# %%
make_video = False
video_duracion = 30
fps_limit = 144
salta_graph = int(
    np.ceil(
        pasos / video_duracion / fps_limit
    )
)
if make_video:
    frames = []
    import moviepy.editor as mpy
    from moviepy.video.io.bindings import mplfig_to_npimage
    fig = plt.figure(layout='constrained', figsize=(16,5))
    subfigs = fig.subfigures(1,2, wspace=0)
    ax0 = subfigs[0].subplots(1,1)
    
    ax0.axis("equal")
    ax0.set_xlabel('$x$')
    ax0.set_ylabel('$y$')
    subfigs[0].suptitle("Contourf")

    ax1 = subfigs[1].add_subplot(111, projection="3d")
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    subfigs[1].suptitle("3D")

    for i in np.arange(0,U.shape[1],step=salta_graph):
        ax0.clear()
        ax1.clear()

        plot0 = ax0.tricontourf(
            coords[:,0],
            coords[:,1],
            U[:,i],
            levels=20,
            cmap=mapa_de_color
        )

        plot1 = ax1.plot_trisurf(
            coords[:,0],
            coords[:,1],
            U[:,i],
            triangles=faces,
            cmap=mapa_de_color
        )

        # fig.colorbar(plot0)
        # fig.colorbar(plot1)
        fig.suptitle("Solución $U$ en $t=$"+str(np.round(t[i],5)))
        
        frames.append(mplfig_to_npimage(fig))
        print("frame = %d        t = %1.4f" %(i, t[i]))

    video_fps = len(frames) / video_duracion
    animation = mpy.VideoClip(lambda t: frames[int(t*video_fps)], duration=video_duracion)
    # duration es la duración del video
    # t irá desde 0 hasta el valor dado en duration
    # los índices de frames[i] deberán de ser enteros i=int(variable)
    # los índices irán desde 0 hasta el tamaño de frames
    animation.write_videofile('figuras/euler2D.mp4', fps=video_fps, verbose=False, logger=None)
    # se puede seleccionar fps según la cantidad de frames entre duration
    # fps = frames.shape[0] / duration
    # puede ser este valor o uno inferior

#%%
plt.show()