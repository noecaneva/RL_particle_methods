#!/usr/bin/env python3
import korali
import argparse

import sys
sys.path.append('./_model')
from objectivefunction import objectivefunction2d

parser = argparse.ArgumentParser()

parser.add_argument('--run', help='Run tag', type=int, default=0, required=False)
args = vars(parser.parse_args())

run = args["run"]

k = korali.Engine()
e = korali.Experiment()
   
# Configuring Problem
e["Random Seed"] = 0xBEEF
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = objectivefunction2d

e["Variables"][0]["Name"] = "rRepulsion"
e["Variables"][0]["Lower Bound"] = 0.0
e["Variables"][0]["Upper Bound"] = +2.0

e["Variables"][1]["Name"] = "delrOrientation"
e["Variables"][1]["Lower Bound"] = 0.
e["Variables"][1]["Upper Bound"] = +20

e["Variables"][2]["Name"] = "delrAttraction"
e["Variables"][2]["Lower Bound"] = 0.
e["Variables"][2]["Upper Bound"] = +20.0

e["Variables"][3]["Name"] = "alpha"
e["Variables"][3]["Lower Bound"] = 0.
e["Variables"][3]["Upper Bound"] = 6.28

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 32
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-12
e["Solver"]["Termination Criteria"]["Max Generations"] = 100

# Configuring results path
e["File Output"]["Enabled"] = True
e["File Output"]["Use Multiple Files"] = True
e["File Output"]["Path"] = f'_result_cmaes_{run}'
e["File Output"]["Frequency"] = 1

e["Console Output"]["Verbosity"] = "Detailed"

k["Conduit"]["Type"] = "Concurrent"
k["Conduit"]["Concurrent Jobs"] = 8

k.run(e)
