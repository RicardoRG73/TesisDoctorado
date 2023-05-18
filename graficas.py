import matplotlib.pyplot as plt
import numpy as np

def nodos_por_color(boundaries, p, labels=[], interior=[], label_interior="", alpha=1):
    plt.scatter(p[interior,0], p[interior,1], label=label_interior, alpha=alpha)
    i = 0
    for b in boundaries:
        plt.scatter(p[b,0], p[b,1], label=labels[i], alpha=alpha)
        i += 1
    plt.legend()
    plt.title("Fronteras por color")