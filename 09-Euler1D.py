import numpy as np
import matplotlib.pyplot as plt
plt.style.use(["seaborn-v0_8", "paper.mplstyle"])

u_0 = 1
alpha = 2
u_exact = lambda t: u_0 * np.exp(alpha * t)

N = 11
intervalo = [0, 1]
t = np.linspace(
    intervalo[0],
    intervalo[1],
    N
)
dt = np.diff(intervalo)/N

u_ex = u_exact(t)
plt.plot(t, u_ex, label="$u_{ex} ="+str(u_0)+"e^{"+str(alpha)+"t}$")

# método de Euler
u = np.zeros(N)
u[0] = u_0
for i in range(N-1):
    u[i+1] = (1 + alpha * dt) * u[i]

plt.plot(t, u, label="$u_{num}$")

print(
    "\n\n\n Método de Euler \n",
    "---\n",
    "t, u_ex, u_num, u_ex - u_num\n",
    "---",
)

errores = np.vstack((t, u_ex, u, u_ex-u)).T
print(np.round(errores, 2))
print("Norma 2 = ", np.linalg.norm(errores[:,3]))


plt.legend()
plt.title("Método de Euler hacia adelante 1D")
plt.savefig('figuras/09-Euler1D.png')
plt.show()