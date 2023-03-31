"""
Diferencias Finitas 1D
Ecuación de Poisson
`\nabla u = f(x)`
Condiciones de Dirichlet en ambos extremos
`u_0 = \alpha` y `u_{N+1} = \beta`
"""

""" Librerias necesarias """
import numpy as np
import matplotlib.pyplot as plt

""" Definición de las condiciones de frontera y fuente """
alpha = 0
beta = 0
fuente = lambda x: 1

""" Discretización del dominio """
# Dominio: [0,1]
N = 11                      # número de nodos en el dominio
x = np.linspace(0,1,N)
h = x[1] - x[0]
fig0 = plt.figure(figsize=(6,3))
plt.plot(x,x*0, marker="o")
plt.title("Dominio Discretizado")
plt.xlabel("x")

""" Ensamble del sistema `KU=F` """
K = np.zeros((N,N))
F = np.zeros(N)
for i in range(1,N-1):
    K[i,i] = 2
    K[i,i-1] = -1
    K[i,i+1] = -1

    F[i] = fuente(x[i])

K = K / h ** 2

F -= K[:,0]*alpha
K[:,0] = 0
F -= K[:,-1]*beta
K[:,-1] = 0

K[0,0] = 1
F[0] = alpha

K[-1,-1] = 1
F[-1] = beta

print(K)
print(F)

""" Solución analítica """
x2 = np.linspace(0,1,101)
y2 = 0.5 * (x2 - x2 ** 2)

fig1 = plt.figure(figsize=(6,3))
plt.plot(x2,y2, label="Solución Analítica")

""" Solución numérica """
U = np.linalg.solve(K,F)

plt.scatter(
    x,
    U,
    marker="o",
    s=70,
    label="Solución Numérica",
    color="magenta"
)
plt.axis("equal")
plt.title("Solución")
plt.legend()
plt.xlabel("x")

plt.show()