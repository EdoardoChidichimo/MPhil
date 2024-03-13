import sys
from config import *
import pickle
import numpy as np
from numpy import pi, arange, random, max
from scipy.io import loadmat
import os
os.environ["CC"] = "gcc"
from jitcdde import jitcdde_input, y, t, input
from symengine import sin
from chspy import CubicHermiteSpline
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import random



# Constants
velocity = 1.65 # (Dumas e2012) or 6? (Axonal velocity)
freq_mean = 40.0 # Gamma oscillations 
freq_std_factor = 10 
n = 90 # No. of oscillators

# Define numerical integration parameters
sfreq = 500 # sampling frequency
n_times = 600 # seconds
times = arange(0, n_times, 1. / float(sfreq))

# Define connectivity strength and lag matrices
dti = loadmat("connectomes.mat")['cuban']
dti = (dti - np.min(dti)) / (np.max(dti) - np.min(dti)) 
w_d = np.array(loadmat("distance.mat")['distance'])
τ = sfreq * w_d / velocity # lag matrix

# Parameters of Interest
cintra_values = np.linspace(0., 1., 50)
value_range = [0.0, 0.5, 1.0]
results = {}

# Set up figure plot
n_conditions = len(value_range)**2 # 9
grid_size = int(np.ceil(np.sqrt(n_conditions))) # 3 x 3
fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
fig.subplots_adjust(hspace=0.4, wspace=0.4)




def kuramotos():
    for i in range(n):
        yield ω[i] + 0.1 * (sum(
            A[j, i] * sin(y(j, t - τ[i, j]) - y(i))
            for j in range(n)
        ) + phase_noise * input(i))



for freq_std in value_range:

    pulsations = (np.random.randn(1, n) * freq_std_factor * freq_std + freq_mean) * (2 * pi) / (sfreq)
    ω = pulsations.flatten()

    for phase_noise in value_range:

        results[(freq_std, phase_noise)] = {}
        cintra_R_values = {}

        for cintra in cintra_values:
            
            # Intra-brain connectivity matrix
            A = np.array(dti * cintra)

            input_data = np.random.normal(size=(len(times), n))
            input_spline = CubicHermiteSpline.from_data(times, input_data)

            # Numerical integration
            DDE = jitcdde_input(f_sym=kuramotos,
                                    n=n,
                                    input=input_spline,
                                    verbose=True)
            DDE.compile_C(simplify=False, do_cse=False, chunk_size=1)
            DDE.set_integration_parameters(rtol=0.0001, atol=0.0000001)
            DDE.constant_past(random.uniform(0, 2 * pi, n), time=0.0)
            DDE.integrate_blindly(max(τ), 1)

            # Retrieve phase angles
            output = []
            for time in times:
                output.append([*DDE.integrate(time) % (2*pi)])
            phases = np.array(output)

            # Calculate order parameter
            R = np.abs(np.mean(np.exp(1j * phases), axis=1))
            cintra_R_values[cintra] = R

        results[(freq_std, phase_noise)] = cintra_R_values


# WRITE THESE RESULTS TO A FILE
with open('debug_order_param_results.pkl', 'wb') as file: 
    pickle.dump(results, file)