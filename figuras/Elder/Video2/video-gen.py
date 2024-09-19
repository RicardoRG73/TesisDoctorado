# Generador de video

from manim import *
import pickle

# Lectura de los datos de solucion
_,t,_,_,_ = pickle.load(open(
    "../solN533_medium_time.pkl",
    "rb"
))

duracion = 40
N_times = t.shape[0]
dt = duracion / N_times

config.frame_rate = 60

class Elder2(Scene):
    def construct(self):
        for time in t:
            im = ImageMobject("t%1.4f" %time + ".jpeg")
            self.add(im)
            self.wait(dt)
            self.remove(im)