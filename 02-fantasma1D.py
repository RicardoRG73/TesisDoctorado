import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1,11)
y = np.zeros(x.shape)

plt.style.use(['seaborn-v0_8','paper.mplstyle'])

fig = plt.figure(figsize=(7,2.5))
plt.scatter(x,y, s=50)
plt.plot(x,y)
plt.scatter(1.1, 0, s=50, color="tab:green")

plt.text(x=1.1, y=0.01, s=r"$p_g$", color="tab:green", ha="center")

plt.title("Nodo Fantasma")


plt.show()