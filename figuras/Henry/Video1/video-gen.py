# Generador de video

from manim import *
import pickle

# Lectura de los datos de solucion
U,t,p,Dxpsi,Dypsi = pickle.load(open(
    "../solN5969_long_time.pkl",
    "rb"
))
N = 5969

duracion = 15
N_times = t.shape[0]
dt = duracion / N_times

config.frame_rate = int(N_times / duracion)
print(config.frame_rate)

class Henry1(Scene):
    def construct(self):
        for time in t:
            im = ImageMobject("t%1.5f"  %time + ".jpeg")
            self.add(im)
            self.wait(dt)
            self.remove(im)