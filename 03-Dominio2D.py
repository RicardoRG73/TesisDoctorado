import numpy as np
import matplotlib.pyplot as plt

""" Discretización del dominio """
n = 13                  # número de nodos en la dirección x
m = 11                  # número de nodos en la dirección y
a0 = 0                  # inicio x, dominio
b0 = 2                  # fin x, dominio
a1 = 0                  # inicio y, dominio
b1 = 1                  # fin y, dominio
X = np.linspace(a0,b0,n)
Y = np.linspace(a1,b1,m)
h = X[1] - X[0]
k = Y[1] - Y[0]
x, y = np.meshgrid(X,Y)
x = x.flatten()
y = y.flatten()


""" Estilo de las gráficas """
plt.style.use(['seaborn-v0_8','paper.mplstyle'])

""" Gráfia de los nodos """
fig = plt.figure(figsize=(8*(b0-a0),8*(b1-a1)))

for i in range(X.shape[0]):
    plt.plot(X[i]+Y*0, Y, marker='o', color="tab:blue")
for i in range(Y.shape[0]):
    plt.plot(X, Y[i]+X*0, marker='o', color="tab:blue")

plt.xlabel("$x$")
plt.ylabel("$y$")


plt.axis("equal")
plt.title("Nodos")
plt.show()