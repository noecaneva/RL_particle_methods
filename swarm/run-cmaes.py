#!/usr/bin/env python3
import korali

import sys
sys.path.append('./_model')
from directModel import evaluateModel

k = korali.Engine()
e = korali.Experiment()

# Configuring Problem
e["Random Seed"] = 0xBEEF
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evaluateModel

e["Variables"][0]["Name"] = "rRepulsion"
e["Variables"][0]["Lower Bound"] = 1e-14
e["Variables"][0]["Upper Bound"] = +2.0

e["Variables"][1]["Name"] = "delrOrientation"
e["Variables"][1]["Lower Bound"] = 1e-14
e["Variables"][1]["Upper Bound"] = +2.0

e["Variables"][2]["Name"] = "delrAttraction"
e["Variables"][2]["Lower Bound"] = 1e-14
e["Variables"][2]["Upper Bound"] = +2.0

e["Variables"][3]["Name"] = "psi"
e["Variables"][3]["Lower Bound"] = -1.0
e["Variables"][3]["Upper Bound"] = +1.0

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 32
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-7
e["Solver"]["Termination Criteria"]["Max Generations"] = 100

# Configuring results path
e["File Output"]["Enabled"] = True
e["File Output"]["Path"] = '_korali_result_cmaes'
e["File Output"]["Frequency"] = 1

k.run(e)
