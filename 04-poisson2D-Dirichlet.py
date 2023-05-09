#%%
"""
Solución de la ecuación de Poisson en 2D
\nabla^2 u = f
en el dominio [0,2]x[0,1]

con condiciones de frontera
u(0,y) = 0
u(2,y) = 2
u(x,0) = x
u(x,1) = x

y fuente
f(x,y) = -5
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-v0_8','paper.mplstyle'])

""" Discretización del dominio """
n = 13                  # número de nodos en la dirección x
m = 11                  # número de nodos en la dirección y
N = n*m                 # número total de nodos
a0 = 0                  # inicio x, dominio
b0 = 2                  # fin x, dominio
a1 = 0                  # inicio y, dominio
b1 = 1                  # fin y, dominio
X = np.linspace(a0,b0,n)
Y = np.linspace(a1,b1,m)
h = X[1] - X[0]
k = Y[1] - Y[0]
x, y = np.meshgrid(X,Y)

p = np.vstack((x.T.flatten(), y.T.flatten())).T

""" Parámetros del problema """
f = lambda p: -5        # fuente
u_izq = lambda p: 0     # dirichlet izquierda
u_der = lambda p: 2     # dirichlet derecha
u_inf = lambda p: p[0]  # dirichlet inferior
u_sup = lambda p: p[0]  # dirichlet superior


""" Ensamble del sistema KU=F """
K = np.zeros((N,N))
F = np.zeros(N)

b_izq = np.arange(0,m)
b_der = np.arange(N-m,N)
b_inf = np.arange(m,N-m,m)
b_sup = np.arange(2*m-1,N-m,m)

B = np.hstack((b_izq,b_der,b_inf,b_sup))

interiores = np.setdiff1d(np.arange(N), B)

# Gráfia de los nodos
fig = plt.figure(figsize=(8*(b0-a0),8*(b1-a1)))

plt.scatter(p[interiores,0], p[interiores,1], label="Nodos Interiores")

plt.scatter(p[b_izq,0], p[b_izq,1], label="Frontera Izquierda")
plt.scatter(p[b_der,0], p[b_der,1], label="Frontera Derecha")
plt.scatter(p[b_inf,0], p[b_inf,1], label="Frontera Inferior")
plt.scatter(p[b_sup,0], p[b_sup,1], label="Frontera Superior")

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend(loc="center")

# llenado de la matriz K
for i in interiores:
    K[i,i] = - 2/h**2 - 2/k**2
    K[i,i-1] = 1/k**2
    K[i,i+1] = 1/k**2
    K[i,i+m] = 1/h**2
    K[i,i-m] = 1/h**2

# llenado del vector F
for i in interiores:
    F[i] = f(p[i])

# Valores que pasan al lado derecho
for i in b_izq:
    F -= K[:,i] * u_izq(p[i])
for i in b_der:
    F -= K[:,i] * u_der(p[i])
for i in b_sup:
    F -= K[:,i] * u_sup(p[i])
for i in b_inf:
    F -= K[:,i] * u_inf(p[i])

# valores en las fronteras
K[:,B] = 0
K[B,B] = 1

for i in b_izq:
    F[i] = u_izq(p[i])
for i in b_der:
    F[i] = u_der(p[i])
for i in b_sup:
    F[i] = u_sup(p[i])
for i in b_inf:
    F[i] = u_inf(p[i])


""" Solución del sistema KU=F """
U = np.linalg.solve(K,F)

plt.style.use(['seaborn-v0_8','paper3dplot.mplstyle'])
mapa_de_color = "plasma"

fig = plt.figure(figsize=(7,7))
ax = plt.axes(projection="3d")
ax.plot_trisurf(
    p[:,0],
    p[:,1],
    U,
    cmap=mapa_de_color,
    linewidth=1,
    antialiased=False
)
ax.view_init(azim=-110, elev=15)

plt.title("Solución (3D)")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$u(x,y)$")

fig = plt.figure(figsize=(7,4))
plt.tricontourf(
    p[:,0],
    p[:,1],
    U,
    levels=10,
    cmap=mapa_de_color
)
plt.colorbar()
plt.axis('equal')
plt.ylim([a1,b1])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title("Solución (Contorno)")


plt.show()
# %%
