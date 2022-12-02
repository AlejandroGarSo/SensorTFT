import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import os
import glob
import pnn
import trayectoryDraw
import cutter
import ui
import utils

dataDir = r"RawData\B"
sepDir = r"SepData"
type = "tray"
cutpoint = 0.3
timer = .5
timer2 = 20
timer3 = 50
modelfile = r"model.h5"

#trayectoryDraw.draw(r"SepData/F/accel/D17/3/a")

w = ui.initUI()
model = pnn.loadModel(modelfile)
ui.manageUI(w, model)

#cutter.getFilesLoose(dataDir, sepDir, cutpoint, timer, "accel")
#min, max = utils.getSizeLimits()
#model = pnn.createModel(max)
