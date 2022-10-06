#!/usr/bin/env python
import sys
sys.path.append('..')
from objectivefunction import *

def evaluateModel(p):
    rRepulsion = p["Parameters"][0]
    delrOrientation = p["Parameters"][1]
    delrAttraction = p["Parameters"][2]
    psi = p["Parameters"][3]
    f = p["Parameters"][4]
    retvalue = objectivefunction(rRepulsion, delrOrientation, delrAttraction, psi, f)
    print(retvalue[0])
    p["F(x)"] = retvalue[0]
