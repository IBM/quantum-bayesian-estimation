# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import Dict, List, Union, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from qiskit import pulse, IBMQ


# Helper routines for using the SPAM1QModel class. See usage examples in other files.


_n_params = 12


def get_qubit_groups(s_backend: str, n_qubits: int):
	if n_qubits == 1:
		qubit_group_1 = [0]
		qubit_group_2 = []
	elif n_qubits == 5:
		if s_backend == 'ibmq_manila' or s_backend == 'ibmq_bogota' or s_backend == 'ibmq_santiago':
			# linear chain devices
			qubit_group_1 = [0, 2, 4]
			qubit_group_2 = [1, 3]
		elif s_backend == 'ibmq_quito' or s_backend == 'ibmq_belem' or s_backend == 'ibmq_lima':
			# T-shape devices
			qubit_group_1 = [0, 2, 3]
			qubit_group_2 = [1, 4]
		else:  # dunno
			raise Exception(f"The connectivity of the 5Q device {s_backend} should be explicitly"
							" coded in the lines above.")
	elif n_qubits == 7:
		qubit_group_1 = [0, 2, 3, 4, 6]
		qubit_group_2 = [1, 5]
	elif n_qubits == 27:
		qubit_group_1 = [0, 2, 4, 5, 6, 9, 10, 11, 13, 15, 16, 17, 20, 21, 22, 24, 26]
		qubit_group_2 = [1, 3, 7, 8, 12, 14, 18, 19, 23, 25]
	elif n_qubits == 65:
		qubit_group_1 = [0, 2, 4, 6, 8, 13, 15, 17, 19, 21, 23, 27, 29, 31, 33, 35, 37,
						 41, 43, 45, 47, 49, 51, 55, 57, 59, 61, 63]
		qubit_group_2 = [1, 3, 5, 7, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 25, 26, 28, 30, 32, 34, 36, 38, 39,
						 40, 42, 44, 46, 48, 50, 52, 53, 54, 56, 58, 60, 62, 64]
	elif n_qubits == 127:
		qubit_group_1 = list(range(n_qubits))
		qubit_group_2 = []
	else:
		raise -1
	return qubit_group_1, qubit_group_2


def init_device(s_provider: str, s_backend: str):
	print('Connecting to IBMQ account.\n')
	IBMQ.load_account()
	p_words = s_provider.split('/')
	provider = IBMQ.get_provider(hub = p_words[0], group = p_words[1], project = p_words[2])
	backend = provider.get_backend(s_backend)
	n_qubits = backend.configuration().n_qubits
	return backend, n_qubits


def process_qubit_groups(s_backend: str, n_groups: int, n_qubits: int, b_sequential: bool):
	qubit_group_1, qubit_group_2 = get_qubit_groups(s_backend, n_qubits)
	if n_qubits != len(qubit_group_1) + len(qubit_group_2):
		raise -1  # bug above
	if n_groups == 1:
		qubits = [qubit_group_1 + qubit_group_2]
	elif n_groups == 2:
		qubits = [qubit_group_1, qubit_group_2]
	elif n_groups == 3:
		qubits = [qubit_group_1, qubit_group_2, qubit_group_1 + qubit_group_2]
	else:
		qubits = []
	if b_sequential:
		if n_groups != 3:
			raise -1  # Cannot add sequential unless there are 3 groups, due to current implementation
		n_groups += n_qubits
		for qubit in range(n_qubits):
			qubits.append([qubit])
	if n_groups == 1 or n_groups == 2:
		n_graphs = 1
	elif n_groups == 3:
		n_graphs = 2
	else:
		n_graphs = 3
	return qubits, n_graphs, _n_params


def process_data(data_mean: np.ndarray, data_vars: np.ndarray, i_graph: int, qubit: int, spam_results: Dict):
	mean_dict: Dict = spam_results['mean_dict']
	data_mean[i_graph, qubit, 0:3] = np.asarray([
		mean_dict.get('x_0', 0),
		mean_dict.get('y_0', 0),
		mean_dict.get('z_0', 0)])
	pi_0 = mean_dict['pi_0']
	pi_z = mean_dict['pi_z']
	data_mean[i_graph, qubit, 6:9] = np.asarray([pi_z, pi_0, spam_results['Var_P']])
	vars_dict = spam_results['vars_dict']
	Vpi_0 = vars_dict['pi_0']
	Vpi_z = vars_dict['pi_z']
	data_vars[i_graph, qubit, 0:3] = np.asarray([
		vars_dict.get('x_0', np.nan),
		vars_dict.get('y_0', np.nan),
		vars_dict.get('z_0', np.nan)])
	data_vars[i_graph, qubit, 6:9] = np.asarray([Vpi_z, Vpi_0, 0.])

	rho = np.linalg.norm(data_mean[i_graph, qubit, 0:2], 2)
	r = np.linalg.norm(data_mean[i_graph, qubit, 0:3], 2)
	data_mean[i_graph, qubit, 3] = rho
	if rho:
		data_vars[i_graph, qubit, 3] = ((data_mean[i_graph, qubit, 0] / rho) ** 2) * data_vars[i_graph, qubit, 0] + \
									   ((data_mean[i_graph, qubit, 1] / rho) ** 2) * data_vars[i_graph, qubit, 1]
	data_mean[i_graph, qubit, 4] = r
	if rho:
		data_mean[i_graph, qubit, 5] = np.arctan2(data_mean[i_graph, qubit, 1], data_mean[i_graph, qubit, 0])
	# Variance is not calculated for the above two.

	data_mean[i_graph, qubit, 9] = 1 - (pi_0 + pi_z)
	data_vars[i_graph, qubit, 9] = Vpi_0 + Vpi_z
	# The variance of p(1|0) here neglects the covariance of the two parameters.
	data_mean[i_graph, qubit, 10] = mean_dict.get('epsilon', 0.)
	data_mean[i_graph, qubit, 11] = mean_dict.get('theta', 0.)
	data_vars[i_graph, qubit, 10] = vars_dict.get('epsilon', 0.)
	data_vars[i_graph, qubit, 11] = vars_dict.get('theta', 0.)


s_subs = ['x_0', 'y_0', 'z_0', '\\rho_0', 'r_0', '\\theta_0', '\\pi_z', '\\pi_0', 'Var_P', 'p(1|0)^*', '\\epsilon', '\\theta']
s_figs = ['x_0', 'y_0', 'z_0', 'rho_0', 'r_0', 'theta_0', 'pi_z', 'pi_0', 'Var_P', 'p_1_0', 'epsilon', 'theta']
markers = ['o', 's', '*', '>', '<', '^', 'v', 'd', '.']


def plot_device_data(n_graphs, s_graphs, n_qubits, data_mean, data_vars, s_title, b_save_figures, s_output_path):
	fig, axs = plt.subplots(3, 4, figsize = (19, 10))
	plt.rcParams.update({'font.size': 12})
	i_data = 0
	for i in (0, 1, 2):
		for j in (0, 1, 2, 3):
			for i_graph in range(n_graphs):
				ax = axs[i, j]
				s_label = s_graphs[i_graph]
				im = ax.errorbar(range(n_qubits), data_mean[i_graph, :, i_data],
								 np.sqrt(data_vars[i_graph, :, i_data]),
								 label = s_label, fmt = 'o', capsize = 4)
				s_ylabel = f"${s_subs[i_data]}$"
				ax.set_ylabel(s_ylabel, fontsize = 12)
				if i == 2:
					ax.set_xlabel('qubit', fontsize = 12)
				if i_data == 0:
					ax.set_title(s_title, fontsize = 12)
			i_data += 1
	if n_graphs > 1:
		plt.legend()
	if b_save_figures:
		plt.savefig(s_output_path + s_title)


def plot_device_pairs(n_graphs, s_graphs, n_qubits, data_mean, data_vars, job_dates, s_title_prefix,
					  s_title, b_save_figures, fontsize, s_output_path, fig_data = None):
	if fig_data is None:
		fig_data = [[3, 2], [0, 1], [6, 7], [10, 11]]
	for i_fig in range(len(fig_data)):
		fig, axs = plt.subplots(2, 1, figsize = (13, 9))
		plt.rcParams.update({'font.size': fontsize})
		for i in (0, 1):
			i_data = fig_data[i_fig][i]
			for i_graph in range(n_graphs):
				ax = axs[i]
				s_label = s_graphs[i_graph]
				if job_dates is not None:
					s_label += f', $t$ = {round((job_dates[i_graph] - job_dates[0]).seconds / 60, 1)}m'
				im = ax.errorbar(range(n_qubits), data_mean[i_graph, :, i_data],
								 np.sqrt(data_vars[i_graph, :, i_data]),
								 label = s_label, fmt = markers[i_graph], capsize = 4)
				s_ylabel = f"${s_subs[i_data]}$"
				ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
				ax.set_ylabel(s_ylabel, fontsize = fontsize)
				if i == 1:
					ax.set_xlabel('qubit', fontsize = fontsize)
				if i == 0:
					ax.set_title(s_title, fontsize = fontsize)
				elif i_fig == 0 and i == 0:
					ax.set_ylim([0, .1])
				elif i_fig == 1 and i == 1:
					ax.set_ylim([-.1, .1])
				elif i_fig == 3 and i == 1:
					pass
					# ax.set_ylim([-.006, .006])
			if n_graphs > 1 and i == 1:
				plt.legend(fontsize = 13)
		if b_save_figures:
			plt.savefig(s_output_path + f"{s_title_prefix} {i_fig}, " + s_title)


def pingpong_get_x90(backend, qubit: int, amp_factor = 1.) -> pulse.Schedule:
	"""Get X90p pulse schedule from backend. U2 schedule is used to get half-pi pulse.

	Args:
		qubit: target qubit
		amp_factor: amplitude prefactor, use 1. for x90p, use -1. for x90m.

	Returns:
		The calibrated schedule.
	"""
	u2_sched = backend.defaults().instruction_schedule_map.get('u2', qubit, 0, 0)
	y90p = u2_sched.filter(instruction_types = [pulse.Play]).instructions[0][1].pulse
	cal_pulse = pulse.Drag(y90p.duration, -1j * amp_factor * y90p.amp, y90p.sigma, y90p.beta,
						   'x90p' if amp_factor > 0. else 'x90m')
	return pulse.Schedule(pulse.Play(cal_pulse, backend.configuration().drive(qubit)))


def pingpong_get_amp_schedules(amplitude_scales, n_pingpong_repetitions, qubit_groups, backend):
	instmap = backend.defaults(refresh = True).instruction_schedule_map
	exp_scheds = []
	for amp_scale in amplitude_scales:
		for group in qubit_groups:
			for nrep in range(n_pingpong_repetitions):
				with pulse.build(backend, default_alignment='sequential') as exp_sched:
					with pulse.align_left():
						for qind in group:
							# rescaling pulse amp
							# sx_pulse = instmap.get('sx', (qind,)).instructions[0][1].pulse
							# new_params = sx_pulse.parameters.copy()
							# new_params['amp'] *= amp_scale
							# sx_pulse_scaled = type(sx_pulse)(**new_params, name="sx_scaled")
							# pulse.play(sx_pulse_scaled, pulse.drive_channel(qind))
							# for _ in range(2*nrep):
							# 	pulse.play(sx_pulse_scaled, pulse.drive_channel(qind))
							x90p_sched = pingpong_get_x90(backend, qind, amp_scale)
							pulse.call(x90p_sched)
							for _ in range(2*nrep):
								pulse.call(x90p_sched)
					# pulse.measure(group)
					pulse.measure_all()
				exp_scheds.append(exp_sched)
	return exp_scheds


def pingpong_get_expv(counts, qind, shots):
	outcomes = []
	for count in counts:
		total_counts = 0
		for key, val in count.items():
			if key[::-1][qind] == '1':
				total_counts += val
		outcomes.append(2 * (0.5 - total_counts / shots))

	return np.asarray(outcomes, dtype = float)