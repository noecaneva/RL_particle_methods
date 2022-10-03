#!/usr/bin/env python
import sys
sys.path.append('..')
import objectivefunction

def evaluateModel(p):
    rRepulsion = p["Parameters"][0]
    delrOrientation = p["Parameters"][1]
    delrAttraction = p["Parameters"][2]
    psi = p["Parameters"][3]
    retvalue = objectivefunction(rRepulsion, delrOrientation, delrAttraction,psi)[1]
    p["F(x)"] = retvalue