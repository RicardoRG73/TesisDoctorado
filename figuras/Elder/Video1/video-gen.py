# Generador de video

from manim import *
import pickle

# Lectura de los datos de solucion
_,t,_,_,_ = pickle.load(open(
    "../solN6820_long_time.pkl",
    "rb"
))
N = 5969

duracion = 45
N_times = t.shape[0]
dt = duracion / N_times

config.frame_rate = 144
print(config.frame_rate)

class Elder1(Scene):
    def construct(self):
        for time in t:
            im = ImageMobject("t%1.5f" %time + ".jpeg")
            self.add(im)
            self.wait(dt)
            self.remove(im)