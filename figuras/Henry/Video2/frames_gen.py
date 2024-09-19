import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
plt.rcParams["text.usetex"] = True
mapa_de_color = "plasma"

# Lectura de los datos de solucion
# original
Uo, to, po, Dxpsio, Dypsio = pickle.load(open(
    "../solN5969_long_time.pkl",
    "rb"
))
No = 5969
# pinder
Up, tp, pp, Dxpsip, Dypsip = pickle.load(open(
    "../PindersolN5969_long_time.pkl",
    "rb"
))
Np = 5969
# modified
Um, tm, pm, Dxpsim, Dypsim = pickle.load(open(
    "../ModifiedsolN5969_long_time.pkl",
    "rb"
))
Nm = 5969

for i in range(tm.shape[0]):
    fig = plt.figure()
    ax = plt.axes()
    
    ax.tricontour(po[:,0], po[:,1], Uo[No:,i], levels=[0.5])
    ax.tricontour(pp[:,0], pp[:,1], Up[Np:,i], levels=[0.5])
    ax.tricontour(pm[:,0], pm[:,1], Um[Nm:,i], levels=[0.5])
    
    line_o = ax.collections[0].get_paths()[0].vertices
    line_p = ax.collections[1].get_paths()[0].vertices
    line_m = ax.collections[2].get_paths()[0].vertices

    plt.close()
    
    fig = plt.figure(layout="constrained", figsize=(1920/300,1080/300), dpi=300)
    plt.plot(line_o[:,0], line_o[:,1], label="Original")
    plt.plot(line_p[:,0], line_p[:,1], "--", label="Pinder")
    plt.plot(line_m[:,0], line_m[:,1], ":", label="Modified")
    plt.axis("equal")
    plt.legend()
    plt.title("Isochlor $C=0.5$")
    plt.xlim([1,2])
    plt.ylim([0,1])

    plt.savefig("t%1.5f" %tm[i] + ".jpeg")
    plt.close()

    print("frame = %d / %d          t = %1.5f" %(i+1, tm.shape[0], tm[i]))

print("\nTodos los frames guardados\n")