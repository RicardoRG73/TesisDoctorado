#%%
from manim import *
import pickle
import numpy as np
from scipy.interpolate import LinearNDInterpolator

# Lectura de los datos de solucion
U,t,p,Dx,Dy = pickle.load(open(
    "../solN976_long_time.pkl ",
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

vel = vel / np.max(np.abs(vel)) * 0.75

height = 0.3
HL = 0.25
length = height / HL
Degrees = np.pi / 180
th = 30 * Degrees

length_x = length * np.cos(th)
length_y = length * np.sin(th)

maxx = length_x
maxy = length_y + height

maxx = maxx / 2
maxy = maxy / 2

p[:,0] -= maxx
p[:,1] -= maxy

fun = LinearNDInterpolator(p, vel)

delta = 0.03

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

margen = 0.001
m = np.tan(th)
def in_domain(pos):
    line_inf = m*pos[0] - height / 2
    line_inf = pos[1] >= line_inf + margen

    line_sup = m*pos[0] + height / 2
    line_sup = pos[1] <= line_sup - margen

    line_left = pos[0] >= -maxx + margen

    line_right = pos[0] <= maxx - margen
    
    return line_inf and line_sup and line_left and line_right


def vel_fun(pos):
    if in_domain(pos):
        return fun(*pos[0:2])
    else:
        return np.array([0,0,0])

#%%
config.frame_height = maxy + 1
config.frame_width = maxx + 1
class CNDD2(Scene):
    def construct(self):
        stream_lines = StreamLines(
            vel_fun,
            x_range=[-maxx,maxx,delta],
            y_range=[-maxy,maxy,delta],
            virtual_time=3,
            stroke_width=0.2
        )
        self.add(stream_lines)
        stream_lines.start_animation(
            warm_up=False,
            flow_speed=1,
            time_width=0.05,

        )
        self.wait(stream_lines.virtual_time / stream_lines.flow_speed)
