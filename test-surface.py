import numpy as np
import matplotlib.pyplot as plt

start = -2*np.pi
finish = 2*np.pi
N = 101

x = np.linspace(start,finish,N)
y = x.copy()

x, y = np.meshgrid(x,y)

z = 10*np.sin(np.sqrt(x**2 + y**2))

plt.style.use(['seaborn-v0_8','paper3dplot.mplstyle'])
mapa_de_color = "plasma"

fig = plt.figure(figsize=(7,7))
ax = plt.axes(projection="3d")
ax.plot_trisurf(
    x.flatten(),
    y.flatten(),
    z.flatten(),
    cmap=mapa_de_color,
    linewidth=2,
    antialiased=False
)
# ax.view_init(azim=-60, elev=20)

plt.title("Superficie")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$u(x,y)$")

fig = plt.figure(figsize=(7,6))
plt.tricontourf(
    x.flatten(),
    y.flatten(),
    z.flatten(),
    levels=10,
    cmap=mapa_de_color
)
plt.colorbar()
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Contourf")


plt.show()