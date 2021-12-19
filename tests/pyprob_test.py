
import bioscrapePPL
import bioscrape
import numpy as np
model = bioscrape.sbmlutil.import_sbml("//mnt//d//northeastern//research//causal_inference/Vivarium//vivarium-notebooks//notebooks//LacOperon_stochastic.xml")

timepoints = np.arange(0, 10, 1.0)
#results_det = py_simulate_model(timepoints, Model = model) #Returns a Pandas DataFrame

#Simulate the Model Stochastically
results_stoch = bioscrapePPL.simulator.py_simulate_model(timepoints, Model = model, stochastic = True)