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
plt.scatter(x,x*0+1)



plt.show()