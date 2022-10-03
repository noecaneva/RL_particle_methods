#!/usr/bin/env python3
import korali

import sys
sys.path.append('./_model')
from directModel import *

k = korali.Engine()
e = korali.Experiment()

e["Problem"]["Objective Function"] = evaluateModel

e["Problem"]["Type"] = "Optimization"

e["Variables"][0]["Name"] = "rRepulsion"
e["Variables"][0]["Lower Bound"] = 0.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Variables"][1]["Name"] = "delrOrientation"
e["Variables"][1]["Lower Bound"] = 0.0
e["Variables"][1]["Upper Bound"] = +10.0

e["Variables"][2]["Name"] = "delrAttraction"
e["Variables"][2]["Lower Bound"] = 0.0
e["Variables"][2]["Upper Bound"] = +10.0

e["Variables"][3]["Name"] = "psi"
e["Variables"][3]["Lower Bound"] = -1.
e["Variables"][3]["Upper Bound"] = +1.

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 32
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-7
e["Solver"]["Termination Criteria"]["Max Generations"] = 100

k.run(e)