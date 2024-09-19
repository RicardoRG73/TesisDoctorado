import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
plt.rcParams["text.usetex"] = True
mapa_de_color = "plasma"

# Lectura de los datos de solucion
U,t,p,_,_ = pickle.load(open(
    "../solN976_long_time.pkl",
    "rb"
))

N = p.shape[0]

# creación de los frames
fig = plt.figure(layout='constrained', figsize=(1920/175,1080/175), dpi=175)

subfigs = fig.subfigures(1,3, wspace=0)
ax0 = subfigs[0].subplots(1,1)

ax0.axis("equal")
subfigs[0].suptitle(r"$\Psi$")

ax1 = subfigs[1].add_subplot(111)
ax1.axis("equal")
subfigs[1].suptitle(r"$C$")

ax2 = subfigs[2].add_subplot(111)
ax2.axis("equal")
subfigs[2].suptitle(r"$T$")


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
        U[N:2*N,i],
        levels=20,
        cmap=mapa_de_color
    )

    plot2 = ax2.tricontourf(
        p[:,0],
        p[:,1],
        U[2*N:,i],
        levels=20,
        cmap=mapa_de_color
    )

    fig.suptitle("Solución $U$ en $t=%1.5f$" %t[i])
    
    plt.savefig("t%1.5f" %t[i] + ".jpeg")

    print("frame = %d / %d        t = %1.5f / %1.5f" %(i, t.shape[0], t[i], t[-1]))


print("\nTodos los frames guardados\n")