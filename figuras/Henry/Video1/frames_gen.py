import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
plt.rcParams["text.usetex"] = True
mapa_de_color = "plasma"

# Lectura de los datos de solucion
U,t,p,Dxpsi,Dypsi = pickle.load(open(
    "../solN5969_long_time.pkl",
    "rb"
))
N = 5969

print("\n Solution shape = ", U.shape, "\n")

# creación de los frames
fig = plt.figure(layout='constrained', figsize=(1920/300,1080/300), dpi=300)

subfigs = fig.subfigures(1,2, wspace=0)
ax0 = subfigs[0].subplots(1,1)

ax0.axis("equal")
ax0.set_xlabel('$x$')
ax0.set_ylabel('$y$')
subfigs[0].suptitle(r"$\Psi$")

ax1 = subfigs[1].add_subplot(111)
ax1.axis("equal")
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
subfigs[1].suptitle("C")

for i in np.arange(0, U.shape[1]):
    ax0.clear()
    ax1.clear()

    plot0 = ax0.tricontourf(
        p[:,0],
        p[:,1],
        U[:N,i],
        levels=20,
        cmap=mapa_de_color
    )

    plot1 = ax1.tricontourf(
        p[:,0],
        p[:,1],
        U[N:,i],
        levels=20,
        cmap=mapa_de_color
    )

    fig.suptitle("Solución $U$ en $t=%1.5f$" %t[i])
    
    plt.savefig("t%1.5f" %t[i] + ".jpeg")

    print("frame = %d / %d          t = %1.4f" %(i+1, t.shape[0], t[i]))


print("\nTodos los frames guardados\n")