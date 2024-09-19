# Generador de video

from manim import *
import pickle

# Lectura de los datos de solucion
_,t,_,_,_ = pickle.load(open(
    "../solN976_long_time.pkl",
    "rb"
))

duracion = 60
N_times = t.shape[0]
dt = duracion / N_times

config.frame_rate = 60
print(config.frame_rate)

class CNDD1(Scene):
    def construct(self):
        for time in t:
            im = ImageMobject("t%1.5f" %time + ".jpeg")
            self.add(im)
            self.wait(dt)
            self.remove(im)