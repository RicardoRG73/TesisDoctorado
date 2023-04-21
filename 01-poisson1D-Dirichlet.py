"""
Solución de la ecuación de Poisson
\nabla^2 u = f
utilizando diferencias finitas clásicas (FDM)
En el dominio [0,1]
con condiciones de frontera
u(0) = \alpha = 0
u(1) = \beta = 0
y con fuente constante
f(x) = 1
"""
import numpy as np
import matplotlib.pyplot as plt


""" Discretización del dominio """
N = 11                  # número de nodos
x = np.linspace(0,1,N)
h = x[1] - x[0]

""" Parámetros del problema """
f = lambda x: 0*x + 1
alpha = 0
beta = 0

""" Ensamble del sistema KU = F """
K = np.zeros((N,N))
Ktemp = np.diag(2*np.ones(N-2)) \
    + np.diag(-1*np.ones(N-3), k=1) \
    + np.diag(-1*np.ones(N-3), k=-1)
K[1:-1, 1:-1] = Ktemp
K[0,0] = -1
K[-1,-1] = -1
print(K)
K = K / h**2

F = f(x)
F[1] += alpha / h**2
F[-2] += beta / h**2
F[0] = alpha / h**2
F[-1] = beta / h**2
print(F)

""" Solución del sistema KU = F """
U = np.linalg.solve(K,F)

""" Gáficas """
print(plt.style.available)
plt.style.use(['seaborn-v0_8','paper.mplstyle'])
plt.rcParams['text.usetex'] = True

fig1 = plt.figure(figsize=(8,4))
plt.plot(x,0*x, marker='o')
plt.text(x=x[N//2], y=0.1, s=r"$x_i$")
plt.axis("equal")
plt.title("Nodos")


fig2 = plt.figure(figsize=(8,4))
plt.plot(x,U, marker='o')
plt.axis("equal")
plt.title("Solución")

plt.show()