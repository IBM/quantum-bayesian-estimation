# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import pickle
from typing import Dict, List, Union, Optional, Tuple, Iterable
from qiskit import pulse
from qiskit.pulse import Schedule
from qiskit.tools import job_monitor
from qiskit.compiler import assemble
from qiskit.providers.ibmq import IBMQBackend
from qiskit.result import marginal_counts, Result


class Bayesian1QModel:
	"""This is a base class with partial implementation of methods for Bayesian parameter estimation
	of single qubit models with closed-form analytic formulae.

	This class implements the execution of schedules, the processing and marginalization of the results,
	the saving and loading of results, and some helper functions.

	Derived classes should implement model-dependent methods for building the schedules, performing the
	Bayesian estimation, and high-level wrapper functions that expose a convenient interface. There
	is no binding interface for implementing these parts, but the derived class SPAM1QModel gives a
	concrete example, using the code written here for all tasks that are independent of the
	model details.
	"""

	DEFAULT_SCHEDULE_PREFIX = 'BAYESIAN1Q'
	"""Default prefix string used in schedule names sent to device, unless user specifies a different one."""

	def __init__(self, backend: IBMQBackend, qubit_groups: Union[List[List[int]], List[int]]):
		"""Constructor defining the backend and qubits the model will work on.

		Args:
			backend: Backend to execute on.
			qubit_groups: Groups of qubits on which the estimation gates are run in parallel.
				This argument can be either a list of qubit numbers (all which are scheduled
				in parallel), or a list of lists of qubit number (all qubits in each group are
				scheduled in parallel).
		"""
		self.job_result = None
		self._gates = []
		self.qubit_counts = []
		self.qubit_estimates = []
		self._backend = backend
		if len(qubit_groups) > 0 and type(qubit_groups[0]) != list:
			qubit_groups = [qubit_groups]
			self._b_flat_results = True
		else:
			self._b_flat_results = False
		self._qubit_groups: List[List[int]] = qubit_groups
		self.schedule_prefix = ''
		self._n_repeats = 1

	def set_base_schedule_params(self, gates: List[str], n_repeats = 1,
								 schedule_prefix = ''):
		"""Set internally stored schedule params - on which a following execution of methods
		build_schedules() and build_qubit_counts() depend.

		Args:
			gates: The names of the calibration gates to generate, must be one of self.SUPPORTED_GATES.
			n_repeats: A number of repetitions of each gate to add. This allows to increase the
				effective shots number of the gates, and will be handled automatically by the
				function build_qubit_counts().
			schedule_prefix: A string to be prepended to the name of every Schedule generated for
			 	the estimation. This allows sending multiple estimation schedules in one job and
			 	later separating them automatically.
				If left as an empty string, the member DEFAULT_SCHEDULE_PREFIX is used.
		"""
		self._gates = gates
		self._n_repeats = n_repeats
		if schedule_prefix == '':
			schedule_prefix = self.DEFAULT_SCHEDULE_PREFIX
		self.schedule_prefix = schedule_prefix

	@staticmethod
	def run_schedules(backend, schedules: List[Schedule] = None, load_job: Union[str, int] = '',
					  shots = 8192, meas_level = 2, meas_return = 'single', **kwargs) -> Result:
		"""Run schedules on the backend.

		Args:
			backend: The device backend to run on.
			schedules: The list of schedules to run. Can be None for a job which is loaded.
			load_job: If a string, assumed to be a valid job_id that will be loaded on the backend.
				If it is a nonnegative integer, used as an index into the recent jobs of the
				backed. Otherwise, the job is executed.
			shots: Passed to the qiskit assemble() call.
			meas_level: Passed to the qiskit assemble() call.
			meas_return: Passed to the qiskit assemble() call.
			kwargs: Passed to the qiskit assemble() call.
		Returns:
			The backend's Result object.
		"""
		s_backend = backend.name()
		b_job_id = False
		if isinstance(load_job, str) and len(load_job):
			b_job_id = True
			print(f"Loading job {load_job} from backend {s_backend}.")
			job = backend.retrieve_job(load_job)
		elif isinstance(load_job, int) and load_job >= 0:
			print(f'Loading job #{load_job} from backend {s_backend}.')
			job = backend.jobs()[load_job]
		else:
			print(f"Sending job to backend {s_backend}.")
			qobj = assemble(schedules, backend, meas_level = meas_level, meas_return = meas_return, shots = shots,
							*kwargs)
			job = backend.run(qobj)
			job_monitor(job)
		if not b_job_id:
			print(f"Retrieved results, job id: {job.job_id()}")
		s_err = job.error_message()
		if s_err:
			print(s_err)
			job_result = None
		else:
			job_result = job.result(timeout = 24 * 3600.)
		return job_result

	def build_qubit_counts(self, job_result: Result, b_force_nested_list = False)\
			-> Union[List[Dict], List[List[Dict]]]:
		"""Generate single-qubit counts dictionaries.

		The device readout results are marginalized, and the generated counts correspond to the
		internally stored gates with their repetitions, as generated by the build_schedules() method.

		Args:
			job_result: The backend's Result object.
			b_force_nested_list: If True, the return value is necessarily a nested list of lists.

		Returns:
			A list with a dictionary for each qubit if the constructor's qubits argument was a list
			of qubits, or a list of lists with a dictionary for each qubit, if the argument
			b_force_nested_list is true, or if the constructor's qubits argument was a list of qubit
			groups. Each qubit dictionary contains marginalized single qubit counts for every gate,
			with the key being a tuple with the gate name and its index into the gates list (this
			key guarantees uniqueness in the case of multiple executions of the same gate).
		"""

		my_schedules = {}
		prefix_len = len(self.schedule_prefix)
		for i_sched, sched_result in enumerate(job_result.results):
			name = sched_result.header.name
			if len(name) >= prefix_len and name[0 : prefix_len] == self.schedule_prefix:
				my_schedules[name] = i_sched

		qubit_counts = []
		for i_group, qubit_group in enumerate(self._qubit_groups):
			group_counts = []
			for i_gate, s_gate in enumerate(self._gates):
				for i_repeat in range(self._n_repeats):
					s_name = self._get_schedule_name(i_group, s_gate, i_gate, i_repeat)
					i_sched = my_schedules[s_name]
					counts1: dict = job_result.get_counts(i_sched)
					for i_qubit, qubit in enumerate(qubit_group):
						if i_gate == 0 and i_repeat == 0:
							gate_counts = dict()
							group_counts.append(gate_counts)
						else:
							gate_counts = group_counts[i_qubit]
						marg_counts = marginal_counts(counts1, [qubit])
						cc = marg_counts.get('0', 0)
						shots = cc + marg_counts.get('1', 0)
						if i_repeat == 0:
							gate_counts[(s_gate, i_gate)] = (cc, shots - cc)
						else:
							prev_cc = gate_counts[(s_gate, i_gate)]
							gate_counts[(s_gate, i_gate)] = (prev_cc[0] + cc, prev_cc[1] + shots - cc)
			qubit_counts.append(group_counts)
		if self._b_flat_results and not b_force_nested_list:
			qubit_counts = qubit_counts[0]
		return qubit_counts

	def save_results(self, s_file_prefix: str, job_result: Result, parameters: Dict, results: List[List[Dict]],
					prior_intervals: Optional[List[List[float]]] = None, metadata = None, protocol = 0):
		s_backend = self._backend.name()
		job_id = job_result.job_id
		result = {'backend_name': s_backend, 'job_id': job_id, 'result_date': job_result.date,
				  'qubit_groups': self._qubit_groups, 'parameters': parameters,
				  'prior_intervals': prior_intervals, 'results': results}
		if metadata is not None:
			if type(metadata) is dict:
				result.update(metadata)
			else:
				raise Exception("The metadata argument must be a dictionary, or None.")
		if protocol == 0:
			s_f = '.txt'
		else:
			s_f = '.pkl'
		with open(s_file_prefix + ', ' + s_backend + ', ' + job_id + s_f, 'wb') as file:
			pickle.dump(result, file, protocol)

	@staticmethod
	def load_results(s_file: str):
		with open(s_file, 'rb') as file:
			return pickle.load(file)

	def _get_schedule_name(self, i_group, s_gate, i_gate, i_repeat):
		return f"{self.schedule_prefix}_{i_group}_{s_gate}_{i_gate}_{i_repeat}"

	def _get_x90(self, qubit: int, amp_factor = 1.) -> pulse.Schedule:
		"""Get X90p pulse schedule from backend. U2 schedule is used to get half-pi pulse.

		Args:
			qubit: target qubit
			amp_factor: amplitude prefactor, use 1. for x90p, use -1. for x90m.

		Returns:
			The calibrated schedule.
		"""
		u2_sched = self._backend.defaults().instruction_schedule_map.get('u2', qubit, 0, 0)
		y90p = u2_sched.filter(instruction_types = [pulse.Play]).instructions[0][1].pulse
		cal_pulse = pulse.Drag(y90p.duration, -1j * amp_factor * y90p.amp, y90p.sigma, y90p.beta,
							   'x90p' if amp_factor > 0. else 'x90m')
		return pulse.Schedule(pulse.Play(cal_pulse, self._backend.configuration().drive(qubit)))

	def _get_y90(self, qubit: int, amp_factor = 1.) -> pulse.Schedule:
		"""Get Y90p pulse schedule from backend. U2 schedule is used to get half-pi pulse.

		Args:
			qubit: target qubit:
			amp_factor: amplitude prefactor, use 1. for y90p, use -1. for y90m.

		Returns:
			The calibrated schedule.
		"""
		u2_sched = self._backend.defaults().instruction_schedule_map.get('u2', qubit, 0, 0)
		y90p = u2_sched.filter(instruction_types=[pulse.Play]).instructions[0][1].pulse
		cal_pulse = pulse.Drag(y90p.duration, amp_factor * y90p.amp, y90p.sigma, y90p.beta,
							   'y90p' if amp_factor > 0. else 'y90m')
		return pulse.Schedule(pulse.Play(cal_pulse, self._backend.configuration().drive(qubit)))
