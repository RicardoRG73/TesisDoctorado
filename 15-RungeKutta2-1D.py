#%%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(["seaborn-v0_8", "paper.mplstyle"])
# plt.rcParams["text.usetex"] = False

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

# Método FE
u_fe = np.zeros(N)
u_fe[0] = u_0
for i in range(N-1):
    u_fe[i+1] = (1 + alpha * dt) * u_fe[i]

plt.plot(t, u_fe, label="FE")

# Método BE
u_be = np.zeros(N)
u_be[0] = u_0
for i in range(N-1):
    u_be[i+1] = 1 / (1 - alpha * dt) * u_be[i]

plt.plot(t, u_be, "o", alpha=0.5, label="BE")

# Método CN
u_cn = np.zeros(N)
u_cn[0] = u_0
for i in range(N-1):
    u_cn[i+1] = (1 + alpha * dt / 2) / (1 - alpha * dt / 2) * u_cn[i]

plt.plot(t, u_cn, label="CN")

# Método RK2
u_rk2 = np.zeros(N)
u_rk2[0] = u_0
for i in range(N-1):
    u_rk2[i+1] = ( 1 + 2*dt + 2*dt**2 ) * u_rk2[i]

plt.plot(t, u_rk2, "o", markersize=10, alpha=0.8, label="RK2")

print(
    "\n\n\n Método Runge-Kutta de orden 2 \n",
    "---\n",
    "t, u_ex, u_num, u_ex - u_num\n",
    "---",
)

errores = np.vstack((t, u_ex, u_rk2, u_ex-u_rk2)).T
for i in range(errores.shape[0]):
    print("%1.1f & %1.2f & %1.2f & %1.3f \\\\" %tuple(errores[i,:]))
print("Norma 2 = %1.4f" %np.linalg.norm(errores[:,3]))

plt.legend()
plt.title("Método Runge-Kuta 2 1D")
plt.savefig('figuras/15-RK21D.png')
plt.show()
# %%
