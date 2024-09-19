import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
plt.rcParams["text.usetex"] = True
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.5
plt.rcParams['legend.shadow'] = True
mapa_de_color = "plasma"

# Lectura de los datos de solucion
Ua, ta, pa, _, _ = pickle.load(open(
    "../solN533_medium_time.pkl",
    "rb"
))
Na = 533

Ub, tb, pb, _, _ = pickle.load(open(
    "../solN1430_medium_time.pkl",
    "rb"
))
Nb = 1430

Uc, tc, pc, _, _ = pickle.load(open(
    "../solN6820_medium_time.pkl",
    "rb"
))
Nc = 6820

for i in range(ta.shape[0]):
    fig = plt.figure()
    ax = plt.axes()
    
    ax.tricontour(pa[:,0], pa[:,1], Ua[Na:,i], levels=[0.5])
    ax.tricontour(pb[:,0], pb[:,1], Ub[Nb:,i], levels=[0.5])
    ax.tricontour(pc[:,0], pc[:,1], Uc[Nc:,i], levels=[0.5])
    
    line_a = ax.collections[0].get_paths()[0].vertices
    line_b = ax.collections[1].get_paths()[0].vertices
    line_c = ax.collections[2].get_paths()[0].vertices

    plt.close()
    
    fig = plt.figure(layout="constrained", figsize=(1920/300,1080/300), dpi=300)
    plt.plot([0,4,4,0,0], [0,0,1,1,0], "k", lw=1)
    plt.plot(line_a[:,0], line_a[:,1], label=r"$N = 533$")
    plt.plot(line_b[:,0], line_b[:,1], "--", label=r"$N = 1430$")
    plt.plot(line_c[:,0], line_c[:,1], ":", label=r"$N = 5453$")
    plt.axis("equal")
    plt.legend(loc="lower right")
    plt.title("Isochlor $C=0.5$,   $t=%1.4f$" %ta[i])
    plt.xlim([0,4])
    plt.ylim([0,1])

    plt.savefig("t%1.4f" %ta[i] + ".jpeg")
    plt.close()

    print("frame = %d / %d          t = %1.4f / %1.4f" %(i+1, ta.shape[0], ta[i], ta[-1]))

print("\nTodos los frames guardados\n")