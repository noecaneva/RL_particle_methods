#!/usr/bin/env python
import sys
sys.path.append('..')
from objectivefunction import *

def evaluateModel(p):
    delrOrientation = p["Parameters"][0]
    delrAttraction = p["Parameters"][1]
    emptycorecofac = p["Parameters"][2]
    initialcircle = p["Parameters"][3]
    height = p["Parameters"][4]
    retvalue = objectivefunction(delrOrientation, delrAttraction, emptycorecofac, initialcircle, height)
    print(retvalue[0])
    p["F(x)"] = retvalue[0]
