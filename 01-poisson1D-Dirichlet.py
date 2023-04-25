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
Ktemp = np.diag(-2*np.ones(N-2)) \
    + np.diag(1*np.ones(N-3), k=1) \
    + np.diag(1*np.ones(N-3), k=-1)
K[1:-1, 1:-1] = Ktemp
K[0,0] = 1
K[-1,-1] = 1
print("\n\n\nK =\n", K)
K = K / h**2

F = f(x)
F[0] = alpha / h**2
F[1] -= alpha / h**2
F[-2] -= beta / h**2
F[-1] = beta / h**2
print("F =\n",F)

""" Solución del sistema KU = F """
U = np.linalg.solve(K,F)

""" Gáficas """
plt.style.use(['seaborn-v0_8','paper.mplstyle'])

fig = plt.figure(figsize=(7,3))
plt.plot(x,U, marker='o', label="Numérica")
plt.axis("equal")
plt.title("Solución")

# exacta
plt.plot(x, 0.5*(x**2 - x), label=r"Exacta $\frac{1}{2}(x^2-x)$")
plt.legend()

plt.show()