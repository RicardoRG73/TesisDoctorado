import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
plt.rcParams["text.usetex"] = True
mapa_de_color = "plasma"

# Lectura de los datos de solucion
# pinder

# N = 274
U274, t274, p274, _, _ = pickle.load(open(
    "../PindersolN274_medium_time.pkl",
    "rb"
))

# N = 998
U998, t998, p998, _, _ = pickle.load(open(
    "../PindersolN998_medium_time.pkl",
    "rb"
))

# N = 2748
U2748, t2748, p2748, _, _ = pickle.load(open(
    "../PindersolN2748_medium_time.pkl",
    "rb"
))

# N = 5969
U5969, t5969, p5969, _, _ = pickle.load(open(
    "../PindersolN5969_medium_time.pkl",
    "rb"
))

# grafica
for i in range(t274.shape[0]):
    fig = plt.figure()
    ax = plt.axes()
    
    ax.tricontour(p274[:,0], p274[:,1], U274[274:,i], levels=[0.5])
    ax.tricontour(p998[:,0], p998[:,1], U998[998:,i], levels=[0.5])
    ax.tricontour(p2748[:,0], p2748[:,1], U2748[2748:,i], levels=[0.5])
    ax.tricontour(p5969[:,0], p5969[:,1], U5969[5969:,i], levels=[0.5])

    line_274 = ax.collections[0].get_paths()[0].vertices
    line_998 = ax.collections[1].get_paths()[0].vertices
    line_2748 = ax.collections[2].get_paths()[0].vertices
    line_5969 = ax.collections[3].get_paths()[0].vertices
    
    fig = plt.figure(layout="constrained", figsize=(1920/300,1080/300), dpi=300)
    plt.plot(line_274[:,0], line_274[:,1], ".", label=r"$N=274$")
    plt.plot(line_998[:,0], line_998[:,1], label=r"$N=998$")
    plt.plot(line_2748[:,0], line_2748[:,1], "--", label=r"$N=2748$")
    plt.plot(line_5969[:,0], line_5969[:,1], ":", label=r"$N=5969$")
    plt.axis("equal")
    plt.legend()
    plt.title("Isochlor $C=0.5, t=%1.4f$" %t274[i])
    plt.xlim([1,2])
    plt.ylim([0,1])
    plt.savefig("t%1.4f" %t274[i] + ".jpeg")
    print("frame = %d / %d       t = %1.4f" %(i, t274.shape[0], t274[i]))

print("\nTodos los frames guardados\n")