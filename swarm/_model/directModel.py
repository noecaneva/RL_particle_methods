#!/usr/bin/env python
import sys
sys.path.append('..')
from objectivefunction import *

def evaluateModel(p):
    rRepulsion = p["Parameters"][0]
    delrOrientation = p["Parameters"][1]
    delrAttraction = p["Parameters"][2]
    psi = p["Parameters"][3]
    retvalue = objectivefunction(rRepulsion, delrOrientation, delrAttraction,psi)
    p["F(x)"] = retvalue[1]
