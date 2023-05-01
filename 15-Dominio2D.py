import numpy as np
import matplotlib.pyplot as plt

""" Discretización del dominio """
N = 11                  # número de nodos por dirección
X = np.linspace(0,1,N)
Y = X.copy()
h = X[1] - X[0]
k = h
x, y = np.meshgrid(X,Y)
x = x.flatten()
y = y.flatten()


""" Estilo de las gráficas """
plt.style.use(['seaborn-v0_8','paper.mplstyle'])

""" Gráfia de los nodos """
fig = plt.figure(figsize=(8,8))

for i in range(X.shape[0]):
    plt.plot(X[i]+Y*0, Y, marker='o')
for i in range(Y.shape[0]):
    plt.plot(X, Y[i]+X*0, marker='o')

plt.xlabel("$x$")
plt.ylabel("$y$")


plt.axis("equal")
plt.title("Nodos")
plt.show()