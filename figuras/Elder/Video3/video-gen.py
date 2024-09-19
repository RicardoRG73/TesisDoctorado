#%%
from manim import *
import pickle
import numpy as np
from scipy.interpolate import LinearNDInterpolator

# Lectura de los datos de solucion
U,t,p,Dx,Dy = pickle.load(open(
    "../solN6820_long_time.pkl ",
    "rb"
))

N = p.shape[0]

vx = Dy @ U[:N,-1]
vy = -Dx @ U[:N,-1]

velx = np.zeros((N,3))
for i in range(vx.shape[0]):
    velx[i] = vx[i] * RIGHT

vely = np.zeros((N,3))
for i in range(vy.shape[0]):
    vely[i] = vy[i] * UP

vel = velx + vely

vel = vel / np.max(np.max(vel))

p[:,0] *= 2
p[:,0] -= 4
p[:,1] *= 2
p[:,1] -= 1

fun = LinearNDInterpolator(p, vel)

maxx = 4
maxy = 1
delta = 0.2
# import matplotlib.pyplot as plt
# xtest = np.arange(-maxx,maxx,delta)
# ytest = np.arange(-maxy,maxy,delta)
# xtest, ytest = np.meshgrid(xtest, ytest)
# xtest = xtest.flatten()
# ytest = ytest.flatten()
# veltest = fun(xtest, ytest)
# u = veltest[:,0]
# v = veltest[:,1]
# plt.quiver(xtest, ytest, u, v)
# plt.axis("equal")
# plt.show()
margen = 0.05
def vel_fun(pos):
    if pos[0]>=-maxx+margen and pos[0]<=maxx-margen and pos[1]>=-maxy+margen and pos[1]<=maxy-margen:
        return fun(*pos[0:2])
    else:
        return np.array([0,0,0])

#%%
config.frame_height = 2
config.frame_width = 8
class Elder3(Scene):
    def construct(self):
        stream_lines = StreamLines(
            vel_fun,
            x_range=[-maxx,maxx,delta],
            y_range=[-maxy,maxy,delta],
            virtual_time=3
        )
        self.add(stream_lines)
        stream_lines.start_animation(
            warm_up=False,
            flow_speed=1,
            time_width=0.4,

        )
        self.wait(stream_lines.virtual_time / stream_lines.flow_speed)
