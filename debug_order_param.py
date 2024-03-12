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



dti = loadmat("connectomes.mat")['cuban']

nsfreq = 100.0
freq_mean = 4.0 # theta oscillations 
freq_std_factor = 1
n = 90

cintra_values = np.linspace(0.0, 0.2, 50)
value_range = [0.0, 0.5, 1.0]

w_d = np.array(loadmat("distance.mat")['distance'])
times = arange(0, n_times, 1. / float(sfreq))
velocity = 1.65 # (Dumas e2012) or 6?

results = {}

def kuramotos():
        for i in range(n):
            yield ω[i] + 0.1 * (sum(
                A[j, i] * sin(y(j, t - τ[i, j]) - y(i))
                for j in range(n)
            ) + phase_noise * input(i))

# Set up figure
n_conditions = len(value_range)**2 #9
grid_size = int(np.ceil(np.sqrt(n_conditions))) #3 x 3
fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
fig.subplots_adjust(hspace=0.4, wspace=0.4)


for i, freq_std in value_range:

    pulsations = (np.random.randn(1, n) * freq_std_factor * freq_std + freq_mean) * (2 * pi) / (sfreq)
    ω = pulsations.flatten()

    for j, phase_noise in value_range:

        results[(freq_std, phase_noise)] = {}
        cintra_R_values = {}

        for cintra in cintra_values:

            # Intra-brain coupling
            w_c = np.array(dti * cintra)

            A = w_c
            τ = sfreq * w_d / velocity

            input_data = np.random.normal(size=(len(times), n))
            input_spline = CubicHermiteSpline.from_data(times, input_data)


            DDE = jitcdde_input(f_sym=kuramotos,
                                    n=n,
                                    input=input_spline,
                                    verbose=True)

            DDE.compile_C(simplify=False, do_cse=False, chunk_size=1)
            DDE.set_integration_parameters(rtol=0.0001, atol=0.0000001)
            DDE.constant_past(random.uniform(0, 2 * pi, n), time=0.0)
            DDE.integrate_blindly(max(τ), 1)
            # DDE.t: 9.150162697654958

            output = []
            for time in (DDE.t + times):
                output.append([*DDE.integrate(time) % (2*pi)])
            
            phases = np.array(output)

            # Calculate ORDER PARAMETER
            R = np.abs(np.mean(np.exp(1j * phases), axis=1)) 
            cintra_R_values[cintra] = R

        results[(freq_std, phase_noise)] = cintra_R_values

        ax = axes[i, j]

        color_map = plt.get_cmap('viridis')
        colors = color_map(np.linspace(0, 1, len(cintra_values)))

        for k, cintra in enumerate(sorted(cintra_R_values.keys())):
            ax.plot(cintra_R_values[cintra], label=f'cintra = {cintra}', color=colors[k], linewidth=0.1)

        ax.set_title(f'freq_std: {freq_std}, phase_noise: {phase_noise}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Order Parameter (R)')


plt.suptitle('R Over Time for Different Conditions', fontsize=16)
figures_dir = "debug_order_param_results"
os.makedirs(figures_dir, exist_ok=True)
filename = "all_plots.png"
filepath = os.path.join(figures_dir, filename)
plt.savefig(filepath, dpi=500)  # Adjust DPI for higher resolution
plt.close(fig)


# WRITE THESE RESULTS TO A FILE
with open('debug_order_param_results.pk1', 'wb') as file: 
    pickle.dump(results, file)
