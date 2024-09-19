#%%
from manim import *
import pickle
import numpy as np
from scipy.interpolate import LinearNDInterpolator

# Lectura de los datos de solucion
U,t,p,Dx,Dy = pickle.load(open(
    "../PinderSolN5969.pkl",
    "rb"
))
N = 5969

vx = Dy @ U[:N,-1]
vy = -Dx @ U[:N,-1]

velx = np.zeros((5969,3))
for i in range(vx.shape[0]):
    velx[i] = vx[i] * RIGHT

vely = np.zeros((5969,3))
for i in range(vy.shape[0]):
    vely[i] = vy[i] * UP

vel = velx + vely
vel = vel / np.max(np.abs(vel)) * 2

p[:,0] *= 2
p[:,0] -= 2
p[:,1] *= 2
p[:,1] -= 1

# import matplotlib.pyplot as plt
# plt.quiver(p[:,0], p[:,1], vel[:,0], vel[:,1])
# plt.show()

fun = LinearNDInterpolator(p, vel)

def vel_fun(pos):
    if pos[0]>=-2 and pos[0]<=2 and pos[1]>=-1 and pos[1]<=1:
        return fun(*pos[0:2])
    else:
        return np.array([0,0,0])

#%%
config.frame_height = 2
config.frame_width = 4
class Henry4(Scene):
    def construct(self):
        delta_space = 0.3
        stream_lines = StreamLines(
            vel_fun,
            x_range=[-2,2,delta_space],
            y_range=[-1,1,delta_space],
            stroke_width=0.8
        )
        self.add(stream_lines)
        stream_lines.start_animation(warm_up=False, flow_speed=1.5, time_width=0.75)
        self.wait(stream_lines.virtual_time / stream_lines.flow_speed)
