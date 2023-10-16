import matplotlib.pyplot as plt
import numpy as np

def nodos_por_color(
        boundaries,
        p,
        labels=[],
        interior=[],
        label_interior="",
        alpha=1,
        nums=False,
        titulo="Fronteras por color",
        loc='best',
        legend=True
    ):
    plt.scatter(p[interior,0], p[interior,1], label=label_interior, alpha=alpha)
    i = 0
    for b in boundaries:
        plt.scatter(p[b,0], p[b,1], label=labels[i], alpha=alpha)
        i += 1
    if nums:
        for i in range(p.shape[0]):
            plt.text(x=p[i,0], y=p[i,1], s=str(i), fontdict={"fontsize": 10})
    if legend:
        plt.legend(loc=loc)
    plt.title(titulo)