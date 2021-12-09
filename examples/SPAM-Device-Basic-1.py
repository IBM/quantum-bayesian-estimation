# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os

import winsound
import time
from quantum_bayesian_estimation.spam_1q_model import SPAM1QModel
from quantum_bayesian_estimation.spam_routines import *

# This file gives an example for using the SPAM1QModel class. It can be run on any IBM Quantum device
# accessible via the cloud with Pulse permissions, and will estimate the five SPAM parameters as
# defined below, for each qubit with a high accuracy (that can be controlled), plot the results
# and save them to a file that can be loaded later.

# Parameters that can be modified, if necessary
# ----------------------------------------------------------------
shots = 16 * 1024  # The number of shots each schedule is measured
load_job = ''  # Set to an integer to load a previous job by index, or to a string with a job id
n_repeats = 1  # The effective shots of each gate will be n_repeats * shots.
b_save_results = True  # Whether to save the results to a file that can later be loaded.
b_save_figures = True  # Whether to save the figures to a (.png) file.
fontsize = 14
s_provider = 'ibm-q/open/main'
s_backend = 'ibmq_armonk'  # ibmq_bogota ibmq_lima ibmq_quito ibmq_santiago

n_draws = int(20e6)  # Number of initial Bayesian draws (before applying constraints)
x0amp = .1  # x_0, y_0 amplitude in prior below
gates = SPAM1QModel.BAYESIAN_QPCM_GATES  # Quantum Preparation and Classical Measurements errors
parameters = SPAM1QModel.BAYESIAN_QPCM_PARAMETERS  # These are: x_0, y_0, z_0, pi_z, pi_0
prior_intervals = [[-x0amp, x0amp], [-x0amp, x0amp], [.9, 1.], [.4, .55], [.47, .6]]

s_output_path = os.path.abspath('./output') + '/'
if not os.path.exists(s_output_path):
	os.mkdir(s_output_path)

# Non-configurable parameters from here
# ----------------------------------------------------------------
n_groups = 2  # 2 - two groups disconnected qubits, the only option supported in this file
n_graphs = 1
s_graphs = ['Estimation']
b_sequential = False  # add each an estimation of each qubit separately (sequentially)

# Start the the estimation
plt.ioff()
backend, n_qubits = init_device(s_provider, s_backend)
qubits, _, n_params = process_qubit_groups(s_backend, n_groups, n_qubits, b_sequential)
winsound.Beep(1000, 250)
ts0 = time.time()

job_id = ''
job_dates = []
spam_model = SPAM1QModel(backend, qubits)
spam_model.set_schedule_params(gates, n_repeats = n_repeats)
spam_cube = spam_model.prepare_Bayesian(gates, parameters, prior_intervals, n_draws)
i_graph = 0
data_mean = np.zeros((n_graphs, n_qubits, n_params))
data_vars = np.zeros((n_graphs, n_qubits, n_params))
schedules = []
if load_job is None or load_job == '':
	schedules = spam_model.build_schedules()
job_result = SPAM1QModel.run_schedules(backend, schedules, load_job, shots)
job_dates.append(job_result.date)
job_id = job_result.job_id
spam_model.qubit_counts = spam_model.build_qubit_counts(job_result)

results = []
qubit_results = [None] * n_qubits
for i_group, group in enumerate(qubits):
	group_results = []
	for i_qubit, qubit in enumerate(group):
		print(f'Estimating qubit {qubit} in group {i_group}. Gate counts:')
		gate_counts = spam_model.qubit_counts[i_group][i_qubit]
		print(gate_counts)
		spam_results = spam_model.estimate_Bayesian(spam_model.qubit_counts[i_group][i_qubit],
													parameters, spam_cube)
		group_results.append(spam_results)
		print('')
		process_data(data_mean, data_vars, i_graph, qubit, spam_results)
		mean_dict: Dict = spam_results['mean_dict']
		qubit_results[qubit] = mean_dict
	results.append(group_results)

if b_save_results:
	spam_model.save_results(s_output_path + 'results', job_result, parameters, results, prior_intervals)
s_title = s_backend + ', ' + job_id
plot_device_data(n_graphs, s_graphs, n_qubits, data_mean, data_vars, 'SPAM data, ' + s_title,
				 b_save_figures, s_output_path)

ts1 = time.time()
print(f"Execution time: {ts1 - ts0}s")
winsound.Beep(2000, 250)

plot_device_pairs(n_graphs, s_graphs, n_qubits, data_mean, data_vars, job_dates, 'Gates', s_title,
				  b_save_figures, fontsize, s_output_path)

plt.show()
tmp = 2  # Put a breakpoint on this line if running in debug mode
