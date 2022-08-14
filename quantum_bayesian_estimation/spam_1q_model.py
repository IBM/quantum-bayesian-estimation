# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from .mc_cube import *
from .bayesian_1q_model import *


class SPAM1QModel(Bayesian1QModel):
	"""This class implements the generation of schedules for estimation of single-qubit SPAM and some
	gate error parameters. Reusable source code from the base class is used for the their execution
	of the schedules, the marginalization of the results, and the estimation algorithm.
	Two estimation methods are supported - using a Gaussian approximation (for POVM parameters only),
	or a Bayesian estimation method.

	There are three primary ways to use this class after its instantiation (where the constructor
		sets initial parameters).
	Usage 1: Invoke run_preconfigured_estimation(), which constructs, executes (on the device),
		and estimates parameters according to preconfigured settings, with the method's arguments
		allowing some flexibility.
	Usage 2: Using individual methods as follows:
		set_schedule_params() - Stores parameters controlling the building of schedules and the
			corresponding building of the qubit counts.
		build_schedules() - Generates and returns schedules for the estimation. Does not store any
			data, but uses class stored data.
		run_schedules() - A static method for running schedules or loading a job from the backend,
			and return the Result object.
		build_qubit_counts() - Uses a job Result object from a backend to build and return qubit
		 	counts in a data structure usable for the estimation. Does not store any data, but uses
		 	class stored data.
		estimate_one_qubit() - A static method performing estimation based on the qubit counts,
			and returning a dictionary with results.
	Usage 3: For estimation of many qubits, it is advantageous (for the execution time) to avoid
		but rather further break its usage by directly using the two methods prepare_Bayesian() and
			estimate_Bayesian(). See the documentation of these functions for details.
	"""

	_I_PI_X = 0; _I_PI_Y = 1; _I_PI_Z = 2; _I_PI_0 = 3

	_I_X_0 = 4; _I_Y_0 = 5; _I_Z_0 = 6  # Note that _I_Z_0 > _I_X_0, _I_Y_0 must be respected for the code below

	_I_EPSILON = 7; _I_THETA = 8

	SUPPORTED_GATES = ['id', 'x', 'x90p', 'x90m', 'y90p', 'y90m', 'x90p^2', 'x90p^4n', 'x90p^(4n+1)']
	"""Names of the calibrated gates supported for the experiments."""

	SUPPORTED_PARAMETERS = {'pi_x': _I_PI_X, 'pi_y': _I_PI_Y, 'pi_z': _I_PI_Z, 'pi_0': _I_PI_0,
							'x_0': _I_X_0, 'y_0': _I_Y_0, 'z_0': _I_Z_0, 'epsilon': _I_EPSILON,
							'theta': _I_THETA}
	"""Names of the parameters that are supported for estimation."""

	NUM_SUPPORTED_PARAMS = len(SUPPORTED_PARAMETERS)
	"""The number of parameters that are supported for estimation (as listed in
	SUPPORTED_PARAMETERS)."""

	PARAM_BOUNDARIES = {'pi_x': (-1., 1.), 'pi_y': (-1., 1.), 'pi_z': (-1., 1.), 'pi_0': (0., 1.),
						'x_0': (-1., 1.), 'y_0': (-1., 1.), 'z_0': (-1., 1.), 'epsilon': (0., 1.),
						'theta': (-np.pi / 4, np.pi / 4)}
	"""Boundaries of the intervals over which the prior (uniform) distribution of each parameter can
	be defined."""

	GAUSSIAN_FULL_PARAMETERS = ['pi_x', 'pi_y', 'pi_z', 'pi_0']
	"""The four POVM parameters supported for estimation using a Gaussian approximation."""

	GAUSSIAN_FULL_GATES = ['id', 'x', 'y90p', 'x90m']
	"""The four gates used for estimation using a Gaussian approximation of the four POVM parameters."""

	BAYESIAN_CPQM_PARAMETERS = ['pi_x', 'pi_y', 'pi_z', 'pi_0', 'z_0']
	"""The five parameters supported for estimation of Classical Preparation and Quantum Measurement errors."""

	BAYESIAN_CPQM_PRIORS = [[-.1, .1], [-.1, .1], [.4, .55], [.45, .6], [.82, 1.]]
	"""Five default parameter priors for estimation corresponding to the parameters defined in BAYESIAN_CPQM_PARAMETERS.
	 	May not be suitable for all devices, if their manifested errors are too large."""

	BAYESIAN_QPCM_PARAMETERS = ['x_0', 'y_0', 'z_0', 'pi_z', 'pi_0']
	"""The five parameters supported for estimation of Quantum Preparation and Classical Measurement errors."""

	BAYESIAN_QPCM_PRIORS = [[-.1, .1], [-.1, .1], [.82, 1.], [.4, .55], [.45, .6]]
	"""Five default parameter priors for estimation corresponding to the parameters defined in
	BAYESIAN_QPCM_PARAMETERS. May not be suitable for all devices, if their manifested errors
	are too large."""

	BAYESIAN_QPCMG_PARAMETERS = ['x_0', 'y_0', 'z_0', 'pi_z', 'pi_0', 'epsilon', 'theta']
	"""The seven parameters supported for estimation of Quantum Preparation / Classical Measurement
	and Gate errors."""

	BAYESIAN_QPCMG_PRIORS = [[-.1, .1], [-.1, .1], [.82, 1.], [.4, .55], [.45, .6],
							 [.0, .01], [-.01, .01]]
	"""Seven default parameter priors for estimation corresponding to parameters defined in
	BAYESIAN_QPCMG_PARAMETERS. May not be suitable for all devices, if their manifested errors
	are too large."""

	BAYESIAN_CPCMG_PARAMETERS = ['z_0', 'pi_z', 'pi_0', 'epsilon', 'theta']
	"""The five parameters supported for estimation of Classical Preparation / Classical Measurement
	and Gate errors."""

	BAYESIAN_CPCMG_PRIORS = [[.82, 1.], [.4, .55], [.45, .6], [.0, .01], [-.01, .01]]
	"""Seven default parameter priors for estimation corresponding to parameters defined in
	BAYESIAN_CPCMG_PARAMETERS. May not be suitable for all devices, if their manifested errors
	are too large."""

	BAYESIAN_DIRECT_GATES = ['id', 'x', 'x90p', 'x90m', 'y90p', 'y90m']
	"""The six nonconcatenated gates used for estimation using a Bayesian estimation, without
	gate errors."""

	BAYESIAN_QPCM_GATES = ['id', 'x', 'x90p', 'x90m', 'y90p', 'y90m']
	"""The six nonconcatenated gates used for estimation using a Bayesian estimation, without
	gate errors."""

	BAYESIAN_QPCMG_GATES = ['id', 'x90p^2', 'x90p', 'x90m', 'y90p', 'y90m', 'x90p^4n', 'x90p^(4n+1)']
	"""The eight gates used for estimation of seven QPCMG parameters using a Bayesian estimation."""

	DEFAULT_SCHEDULE_PREFIX = 'SPAMPOVM'
	"""Default prefix string used in schedule names sent to device, unless user specifies
	 a different one."""

	DEFAULT_X90P_POWER = 8

	PRE_CONFIGURED_ESTIMATION_METHODS = ['Gaussian_POVM', 'Bayesian_CPQM', 'Bayesian_QPCM',
										 'Bayesian_QPCMG']
	"""Estimation methods preconfigured and supported as the string argument in method
	run_preconfigured_estimation()."""

	def __init__(self, backend: IBMQBackend, qubit_groups: Union[List[List[int]], List[int]]):
		"""Constructor defining the backend and qubits the model will work on.

		Args:
			backend: Backend to execute on.
			qubit_groups: Groups of qubits on which the estimation gates are run in parallel.
				This argument can be either a list of qubit numbers (all which are scheduled
				in parallel), or a list of lists of qubit number (all qubits in each group are
				scheduled in parallel).
		"""
		super().__init__(backend, qubit_groups)
		self._qubit_amp_factors = None
		self.n_x90p_power = self.DEFAULT_X90P_POWER

	def run_preconfigured_estimation(self, s_method = 'Gaussian_POVM', n_repeats = 1, shots = 8192,
									 load_job: Union[str, int] = '', append_schedules = None,
									 prior_intervals: Optional[List[Tuple[float, float]]] = None,
									 n_draws = int(5e7)) -> bool:
		"""Constructs, executes, and estimates parameters using one of the preconfigured methods.

		The field self.qubit_estimates is populated with the result, and it has the same structure
		as the constructor's argument qubit_groups - it is either a list of dictionaries (for every
		qubit), or a list of lists of dictionaries (for every qubit in every group).

		Args:
			s_method: One of the supported strings in the member PRE_CONFIGURED_ESTIMATION_METHODS.
				Can take the following values:
				'Gaussian_POVM' for using the Gaussian approximation for the four POVM parameters
					pi_x, pi_y, pi_z, pi_0, as defined in the member list self.GAUSSIAN_FULL_PARAMETERS,
					using the four gates of self.GAUSSIAN_FULL_GATES.
				'Bayesian_CPQM' for using Bayesian estimation of one classical preparation parameter
				 	z_0 and the four POVM (quantum) measurement parameters pi_x, pi_y, pi_z, pi_0,
				 	as defined in the member list self.BAYESIAN_CPQM_PARAMETERS, using the six gates
				 	of self.BAYESIAN_DIRECT_GATES.
				'Bayesian_QPCM' for using Bayesian estimation of the three (quantum) preparation
					parameters x_0, y_0, z_0, and two (classical) measurement parameters pi_z, pi_0,
					as defined in the member list self.BAYESIAN_QPCM_PARAMETERS, using the six gates
					of self.BAYESIAN_DIRECT_GATES.
				'Bayesian_QPCMG' for using Bayesian estimation of the three (quantum) preparation
					parameters x_0, y_0, z_0, two (classical) measurement parameters pi_z, pi_0,
					and two gate error parameters epsilon, theta, as defined in the member list
					self.BAYESIAN_QPCMG_PARAMETERS, using the eight gates of self.BAYESIAN_QPCMG_GATES.

			n_repeats: A number of repetitions of each gate to add. This allows to increase the
				effective shots number of the gates, and is handled automatically.
			shots: The number of shots to execute on the device.
			load_job: If a string, assumed to be a valid job_id that will be loaded on the backend.
				If a nonnegative integer, assumed to be an index into the recent jobs on the backed.
				Otherwise, the job is executed.
			append_schedules: Additional schedules to be appended to those passed to assemble and
				run on the device.
			prior_intervals: A list corresponding to each of the parameters in
				self.BAYESIAN_6_PARAMETERS, with each entry being a tuple of two elements, the lower
				and upper boundaries of the prior distribution to be used in Bayesian estimation of
				the corresponding parameter. If None, self.DEFAULT_6_PRIORS is used.
			n_draws: The number of points to draw in the Bayesian estimation.

		Returns:
			True if the job results have successfully loaded, otherwise False (in which case,
			estimation is skipped).

		Raises:
			Exception: If the method string is not supported, or the parameters are inconsistent.
		"""
		if s_method == 'Gaussian_POVM':
			gates = self.GAUSSIAN_FULL_GATES
			params = self.GAUSSIAN_FULL_PARAMETERS
			if prior_intervals is not None:
				raise Exception(f"Argument prior_intervals must be set to None for {s_method} "
								"estimation.")
		elif s_method == 'Bayesian_CPQM':
			gates = self.BAYESIAN_DIRECT_GATES
			if prior_intervals is None:
				prior_intervals = self.BAYESIAN_CPQM_PRIORS
			params = self.BAYESIAN_CPQM_PARAMETERS
		elif s_method == 'Bayesian_QPCM':
			gates = self.BAYESIAN_DIRECT_GATES
			if prior_intervals is None:
				prior_intervals = self.BAYESIAN_QPCM_PRIORS
			params = self.BAYESIAN_QPCM_PARAMETERS
		elif s_method == 'Bayesian_QPCMG':
			gates = self.BAYESIAN_QPCMG_GATES
			if prior_intervals is None:
				prior_intervals = self.BAYESIAN_QPCMG_PRIORS
			params = self.BAYESIAN_QPCMG_PARAMETERS
		else:
			raise Exception(f"Unknown/unsupported preconfigured estimation method {s_method}.")
		self.set_schedule_params(gates, n_repeats = n_repeats)
		schedules = self.build_schedules()
		if append_schedules is not None:
			schedules += append_schedules
		self.job_result = self.run_schedules(self._backend, schedules, shots, load_job)
		if self.job_result is None:
			return False
		self.qubit_counts = self.build_qubit_counts(self.job_result, True)

		for i_group, qubit_group in enumerate(self._qubit_groups):
			qubit_counts = self.qubit_counts[i_group]
			qubit_estimates = []
			for i_qubit, qubit in enumerate(qubit_group):
				print(f'Estimating qubit {qubit} using {s_method} estimation.')
				qubit_estimates.append(self.estimate_one_qubit(gates, qubit_counts[i_qubit],
															   params, prior_intervals, n_draws))
			self.qubit_estimates.append(qubit_estimates)
		if self._b_flat_results:
			self.qubit_estimates = self.qubit_estimates[0]
		return True

	def set_schedule_params(self, gates: List[str], n_repeats = 1,
							schedule_prefix = '', n_x90p_power = 0,
							qubit_amplitude_factors: Optional[Iterable[float]] = None):
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
			n_x90p_power: The power n to repeat the application of the x90p gate for estimating gate
				error parameters theta and epsilon (applicable for the supported gates 'x90p^4n' and
				'x90p^(4n+1). If left at 0, the member DEFAULT_X90P_POWER is used.
			qubit_amplitude_factors: A multiplicative amplitude factor for each qubit's x90p/m and
				y90p/m gates.
		"""
		if schedule_prefix == '':
			schedule_prefix = self.DEFAULT_SCHEDULE_PREFIX
		self.set_base_schedule_params(gates, n_repeats, schedule_prefix)
		if n_x90p_power == 0:
			n_x90p_power = self.DEFAULT_X90P_POWER
		self.n_x90p_power = n_x90p_power
		self._qubit_amp_factors = qubit_amplitude_factors

	def build_schedules(self, measure: Optional[Schedule] = None) -> List[Schedule]:
		"""Build schedules for the estimation.

		Args:
			measure: The measurement schedule, and if None, the default schedule will be used.
		Returns:
			The list of generated schedules.
		"""
		if measure is not None:
			raise Exception("A custom measure schedule is not currently supported")
		else:
			pass
			# measure = self._backend.defaults().instruction_schedule_map.get("measure",
			# 							range(self._backend.configuration().n_qubits))
		schedules = []
		for i_group, qubit_group in enumerate(self._qubit_groups):
			for i_gate, s_gate in enumerate(self._gates):
				for i_repeat in range(self._n_repeats):
					s_name = self._get_schedule_name(i_group, s_gate, i_gate, i_repeat)
					with pulse.build(backend = self._backend, name = s_name,
									 default_transpiler_settings = {'basis_gates': ['x']}) as sched:
						for qubit in qubit_group:
							if self._qubit_amp_factors is not None:
								qubit_amp = self._qubit_amp_factors[qubit]
							else:
								qubit_amp = 1.
							sched = self._pre_qubit_call(i_group, i_repeat, qubit, sched)
							if s_gate == 'x':
								pulse.x(qubit)
							elif s_gate[0:4] == 'x90p':
								if s_gate == 'x90p':
									n_len = 1
								elif s_gate == 'x90p^2':
									n_len = 2
								elif s_gate == 'x90p^4n':
									n_len = 4 * self.n_x90p_power
								elif s_gate == 'x90p^(4n+1)':
									n_len = 4 * self.n_x90p_power + 1
								else:
									raise Exception(f"Unknown/unsupported instruction {s_gate}.")
								for _ in range(n_len):
									pulse.call(self._get_x90(qubit, qubit_amp))
							elif s_gate == 'x90m':
								pulse.call(self._get_x90(qubit, -qubit_amp))
							elif s_gate == 'y90p':
								pulse.call(self._get_y90(qubit, qubit_amp))
							elif s_gate == 'y90m':
								pulse.call(self._get_y90(qubit, -qubit_amp))
							elif s_gate == 'id':
								pass
							else:
								raise Exception(f"Unknown/unsupported instruction {s_gate}.")
							sched = self._post_qubit_call(i_group, i_repeat, qubit, sched)
							pulse.barrier(qubit)  # Critical for the measurement timing!
						pulse.measure_all()
					# sched += measure << sched.duration
					schedules.append(sched)
		return schedules

	@staticmethod
	def estimate_one_qubit(gates: List[str], gate_counts: Dict, parameters: List[str],
						   prior_intervals: Optional[List[List[float]]] = None,
						   n_draws = int(5e7), n_x90p_power = 0,
						   b_full_covariances = False, b_full_distributions = False) -> Dict:
		"""Estimate the SPAM/gate error parameters of a single qubit, and their variances.
		If the prior_intervals argument is None, then the estimation is based on a Gaussian
		approximation for the experiment results. Otherwise, the estimation is Bayesian.

		Args:
			gates: A list with the gates to be used for the estimation.
				The supported gates are given in the member list self.SUPPORTED_GATES.
				This parameter is only used with Bayesian estimation.
				If the estimation is analytic (using a Gaussian approximation), only the circuits 'id',
				'x', 'x90m', 'y90p' are used. The first two must be passed, and 'y90m', 'x90p' must be
				passed if the 'pi_x' and 'pi_y' parameters (respectively) are requested for estimation.
				For the Gaussian estimation, the gates must be ordered in the same order in which they
				appear in self.GAUSSIAN_FULL_GATES.
			gate_counts: A dict with string keys of the calibration gates that were run on the device.
				Each keyed entry consists of gate_counts is a two-element tuple,
				the first is the marginalized counts of the result 0, and the second of the result 1.
				The total number of shots for all gate counts must be identical.
			parameters: A list with the string names of the parameters that are to be estimated.
				The supported parameters are given in the list self.SUPPORTED_PARAMETERS.
				The order of the parameters in the parameters list determines the order of entries
				in the output arrays.
			prior_intervals: A list corresponding to each of the parameters in argument 'parameters',
				with each entry being a list of two elements, the lower and upper boundaries of the
				prior distribution to be used in Bayesian estimation of the corresponding parameter.
			n_draws: The number of points to draw in the Bayesian estimation.
			n_x90p_power: The power n with which the application of the x90p gate was repeated for
				estimating gate error parameters theta and epsilon (applicable for the supported
				gates 'x90p^4n' and 'x90p^(4n+1). If left at 0, the member DEFAULT_X90P_POWER is used.
			b_full_covariances: If True, the full covariance matrix of the parameters is estimated.
				Otherwise, only the diagonal elements that correspond to the parameter variances
				are calculated. [CURRENTLY UNIMPLEMENTED]
			b_full_distributions: If True and the estimation is Bayesian, the result dictionary stores
				the full distribution information used in the calculation (which may be very large).

		Returns:
			A dictionary with the results and intermediate calculations.
			Among other entries, the dictionary contains a vector of the means estimated for each
			parameter, and a matrix of covariances. If the estimation is Bayesian an MCCube class
			describing the Monte Carlo calculation is returned as well, and further data generated
			during the estimation.
		"""
		if b_full_covariances:
			raise Exception("Full covariances are currently not implemented.")
		if prior_intervals is None:  # Gaussian approximation
			return SPAM1QModel._estimate_POVM_Gaussian(gate_counts, parameters)
		else:
			cube = SPAM1QModel.prepare_Bayesian(gates, parameters,
												prior_intervals, n_draws, n_x90p_power)
			return SPAM1QModel.estimate_Bayesian(gate_counts, parameters,
												 cube, b_full_covariances, b_full_distributions)

	@staticmethod
	def prepare_Bayesian(gates: List[str], parameters: List[str],
						 prior_intervals: Optional[List[List[float]]], n_draws, n_x90p_power = 0)\
			-> MCCube:
		"""Precomputes the Monte Carlo cube used for Bayesian estimation, enforcing constraints.
		With Python being an interpreted language and this computation being a bottle neck of the
		estimation performance (handling float arrays with Gigas of parameter draws), the code below
		is expanded/unfolded and vectorized as much as possible, to reduce the number of repeated
		computations and achieve the maximal efficiency (while using more intermediate memory).

		Args:
			gates: A list with the gates to be used for the estimation.
				The supported gates are given in the member list self.SUPPORTED_GATES.
			parameters: A list with the string names of the parameters that are to be estimated.
				The supported parameters are given in the list self.SUPPORTED_PARAMETERS.
				The order of the parameters in the parameters list determines the order of entries
				in the output arrays.
			prior_intervals: A list corresponding to each of the parameters in argument 'parameters',
				with each entry being a list of two elements, the lower and upper boundaries of the
				prior distribution to be used in Bayesian estimation of the corresponding parameter.
			n_draws: The number of points to draw in the Bayesian estimation.
			n_x90p_power: The power n with which the application of the x90p gate was repeated for
				estimating gate error parameters theta and epsilon (applicable for the supported
				gates 'x90p^4n' and 'x90p^(4n+1). If left at 0, the member DEFAULT_X90P_POWER is used.

		Returns:
			An MCCube class describing the Monte Carlo calculation, that can be used for the estimation.
		"""
		param_indices = SPAM1QModel._init_param_indices(parameters)
		if len(prior_intervals) != len(parameters):
			raise Exception("The argument prior_intervals must match the parameters argument.")

		V = 1.
		for i_param, s_param in enumerate(parameters):
			lb, ub = SPAM1QModel.PARAM_BOUNDARIES[s_param]
			interval = prior_intervals[i_param]
			if interval[1] < interval[0]:
				raise Exception(f"The prior interval for parameter {s_param} is ill-defined.")
				# We raise exception in this case because such an ambiguity should be fixed by the user.
			b_interval_fixed = False
			if interval[1] > ub:
				interval[1] = ub
				b_interval_fixed = True
			if interval[0] < lb:
				interval[0] = lb
				b_interval_fixed = True
			if b_interval_fixed:
				print(f"Prior interval for parameter {s_param} was restricted to parameter bounds.")
			V *= (interval[1] - interval[0])

		print(f"Preparing Monte Carlo cube with {n_draws:,} parameter draws.")
		cube = MCCube(prior_intervals, V)
		cube.draw(n_draws)
		ordered_values = []
		ordered_indices = []
		b_x0_y0 = False
		for i_param, param_index in enumerate(param_indices):
			if param_index != -1:
				ordered_indices.append(param_index)
				ordered_values.append(cube.values[param_index])
			elif i_param != SPAM1QModel._I_Z_0:
				ordered_indices.append(-1)
				ordered_values.append(cube.zeros)
				if i_param == SPAM1QModel._I_X_0 or i_param == SPAM1QModel._I_Y_0:
					b_x0_y0 = True
			else:  # z_0 is the only one that defaults to 1 if not being estimated
				if b_x0_y0:  # Note that this requires SPAM1QModel._I_Z_0 > SPAM1QModel._I_X_0 and Y0 !
					raise Exception("If either x_0 or y_0 parameters are defined for estimation, z_0 must be estimated"
									" as well for the consistency of the Bloch vector.")
				ordered_indices.append(-2)
				ordered_values.append(cube.ones)
		cube.ordered_indices = ordered_indices
		cube.ordered_values = ordered_values
		pis = (SPAM1QModel._I_PI_X, SPAM1QModel._I_PI_Y, SPAM1QModel._I_PI_Z)
		r0s = (SPAM1QModel._I_X_0, SPAM1QModel._I_Y_0, SPAM1QModel._I_Z_0)
		ipi0 = SPAM1QModel._I_PI_0; ipi_z = SPAM1QModel._I_PI_Z

		# VALIDATIONS
		sum_pi_a_2 = (ordered_values[pis[0]] ** 2) + (ordered_values[pis[1]] ** 2) + (ordered_values[pis[2]] ** 2)
		sum_r_0_2 = (ordered_values[r0s[0]] ** 2) + (ordered_values[r0s[1]] ** 2) + (ordered_values[r0s[2]] ** 2)
		conditions = ((ordered_values[ipi0] + ordered_values[ipi_z]) > 1.,
					  (ordered_values[ipi0] + ordered_values[ipi_z]) <= .0,
					  (ordered_values[ipi0] - ordered_values[ipi_z]) >= 1.,
					  (ordered_values[ipi0] + ordered_values[ipi_z]) < .0,
					  sum_r_0_2 > 1.,
					  ordered_values[ipi0] ** 2 < sum_pi_a_2,
					  (1. - ordered_values[ipi0]) ** 2 < sum_pi_a_2)
		invalid_ps = np.full_like(ordered_values[ipi0], False, dtype = bool)
		for condition in conditions:
			invalid_ps = np.logical_or(invalid_ps, condition)
		n_valid = n_draws - np.count_nonzero(invalid_ps)
		if n_valid != n_draws:
			print(f"Using {n_valid:,} valid parameter draws.")
			cube.delete_values(invalid_ps)

		ones_ = cube.ones
		vals = ordered_values
		ix0 = SPAM1QModel._I_X_0; iy0 = SPAM1QModel._I_Y_0; iz0 = SPAM1QModel._I_Z_0
		ipix = SPAM1QModel._I_PI_X; ipiy = SPAM1QModel._I_PI_Y; ipiz = SPAM1QModel._I_PI_Z
		itheta = SPAM1QModel._I_THETA; iepsilon = SPAM1QModel._I_EPSILON
		vals_pi0 = vals[SPAM1QModel._I_PI_0]
		b_gate_errors = ('theta' in parameters) or ('epsilon' in parameters)
		ct = None; st = None; c2t = None; s2t = None; c4nt = None; s4nt = None; c4n1t = None; s4n1t = None
		err = None; err2 = None; err4n = None; err4n1 = None
		if b_gate_errors:
			if n_x90p_power == 0:
				n_x90p_power = SPAM1QModel.DEFAULT_X90P_POWER
			_4n = 4 * n_x90p_power  # the current number of X90p concatenation used for gate parameters estimation.
			err = (ones_ - vals[iepsilon])
			err2 = (ones_ - vals[iepsilon]) ** 2
			err4n = (ones_ - vals[iepsilon]) ** _4n
			err4n1 = (ones_ - vals[iepsilon]) ** (_4n + 1)
			ct = np.cos(vals[itheta])
			st = np.sin(vals[itheta])
			c2t = np.cos(2. * vals[itheta])
			s2t = np.sin(2. * vals[itheta])
			c4nt = np.cos(_4n * vals[itheta])
			s4nt = np.sin(_4n * vals[itheta])
			c4n1t = np.cos((_4n + 1) * vals[itheta])
			s4n1t = np.sin((_4n + 1) * vals[itheta])

		for i_gate, s_gate in enumerate(gates):
			p = None
			if b_gate_errors:
				if s_gate == 'id':
					p = vals_pi0 + vals[ipix] * vals[ix0] + vals[ipiy] * vals[iy0] + vals[ipiz] * vals[iz0]
				elif s_gate[0:3] == 'y90':
					pm = 1. if s_gate[3] == 'p' else -1.
					p = vals_pi0 + \
						err * (-st * vals[ix0] + pm * ct * vals[iz0]) * vals[ipix] +\
						err * vals[ipiy] * vals[iy0] +\
						err * (-pm * ct * vals[ix0] - st * vals[iz0]) * vals[ipiz]
				elif s_gate[0:3] == 'x90':
					if len(s_gate) == 4:
						pm = 1. if s_gate[3] == 'p' else -1.
						p = vals_pi0 + \
							err * vals[ipix] * vals[ix0] +\
							err * (-st * vals[iy0] - pm * ct * vals[iz0]) * vals[ipiy] +\
							err * (pm * ct * vals[iy0] - st * vals[iz0]) * vals[ipiz]
					elif s_gate == 'x90p^2':
						p = vals_pi0 + \
							err2 * vals[ipix] * vals[ix0] +\
							err2 * (-c2t * vals[iy0] + s2t * vals[iz0]) * vals[ipiy] +\
							err2 * (-s2t * vals[iy0] - c2t * vals[iz0]) * vals[ipiz]
					elif s_gate == 'x90p^4n':
						p = vals_pi0 + \
							err4n * vals[ipix] * vals[ix0] + \
							err4n * (c4nt * vals[iy0] - s4nt * vals[iz0]) * vals[ipiy] + \
							err4n * (s4nt * vals[iy0] + c4nt * vals[iz0]) * vals[ipiz]
					elif s_gate == 'x90p^(4n+1)':
						p = vals_pi0 + \
							err4n1 * vals[ipix] * vals[ix0] + \
							err4n1 * (-s4n1t * vals[iy0] - c4n1t * vals[iz0]) * vals[ipiy] + \
							err4n1 * (c4n1t * vals[iy0] - s4n1t * vals[iz0]) * vals[ipiz]
					else:
						raise Exception(f"Unknown/unsupported gate {s_gate} for estimation.")
			else:
				if s_gate == 'id' or s_gate == 'x90p^4n':
					p = vals_pi0 + vals[ipix] * vals[ix0] + vals[ipiy] * vals[iy0] + vals[ipiz] * vals[iz0]
				elif s_gate == 'x90p' or s_gate == 'x90p^(4n+1)':
					p = vals_pi0 + vals[ipix] * vals[ix0] - vals[ipiy] * vals[iz0] + vals[ipiz] * vals[iy0]
				elif s_gate == 'x90m':
					p = vals_pi0 + vals[ipix] * vals[ix0] + vals[ipiy] * vals[iz0] - vals[ipiz] * vals[iy0]
				elif s_gate == 'y90p':
					p = vals_pi0 + vals[ipix] * vals[iz0] + vals[ipiy] * vals[iy0] - vals[ipiz] * vals[ix0]
				elif s_gate == 'y90m':
					p = vals_pi0 - vals[ipix] * vals[iz0] + vals[ipiy] * vals[iy0] + vals[ipiz] * vals[ix0]
				elif s_gate == 'x' or s_gate == 'x90p^2':
					p = vals_pi0 + vals[ipix] * vals[ix0] - vals[ipiy] * vals[iy0] - vals[ipiz] * vals[iz0]
			if p is None:
				raise Exception(f"Unknown/unsupported gate {s_gate} for estimation (note that 'x' gate is not "
								"supported together with gate errors.")
			log_p = p * 0.
			p_pos = p > 0.
			log_p[p_pos] = np.log(p[p_pos])
			p_non_positive = np.logical_not(p_pos)
			if np.count_nonzero(p_non_positive) == 0:
				p_non_positive = None
			p_non_1 = p < 1.
			log_1_p = p * 0.
			log_1_p[p_non_1] = np.log(1. - p[p_non_1])
			p_non_ones = np.logical_not(p_non_1)
			if np.count_nonzero(p_non_ones) == 0:
				p_non_ones = None

			key = (s_gate, i_gate)
			cube.log_p[key] = log_p
			cube.log_1_p[key] = log_1_p
			cube.p_non_positive[key] = p_non_positive
			cube.p_non_ones[key] = p_non_ones
			# cube.probabilities[key] = p  # commented out to reduce memory usage

		return cube

	@staticmethod
	def estimate_Bayesian(gate_counts: Dict, parameters: List[str], cube: MCCube,
						  b_full_covariances = False, b_full_distributions = False) -> Dict:

		n_valid_draws = len(cube.ones)
		log_L = np.zeros(n_valid_draws)

		for key in gate_counts.keys():
			cc = gate_counts.get(key)
			log_p = cube.log_p.get(key, None)
			if log_p is None:
				raise Exception(f"Inconsistent Bayesian preparation cube.")
			log_1_p = cube.log_1_p.get(key, None)
			p_non_positive = cube.p_non_positive.get(key, None)
			p_non_ones = cube.p_non_ones.get(key, None)
			if cc[0] > 0:
				log_L += log_p * cc[0]
				if p_non_positive is not None:
					log_L[p_non_positive] = -np.inf
			if cc[1] > 0:
				log_L += log_1_p * cc[1]
				if p_non_ones is not None:
					log_L[p_non_ones] = -np.inf

		log_L -= np.max(log_L)
		V_inv = 1. / cube.volume
		P = np.exp(log_L) * V_inv
		N = np.nansum(P / V_inv) / n_valid_draws
		P /= N
		Var_P = np.nansum(np.square(P / V_inv - 1.)) / (n_valid_draws * (n_valid_draws - 1))
		print(f"The sample estimate of the variance of the integral of the posterior distribution "
				   f"is\n\tVar[P] = {Var_P:5}.")

		mean, cov = SPAM1QModel._init_output(parameters)
		mean_dict = {}
		vars_dict = {}
		for i_param, s_param in enumerate(parameters):
			m = np.nansum(P * cube.values[i_param] / V_inv) / n_valid_draws  # mean estimator
			mm = np.nansum(P * np.square(cube.values[i_param]) / V_inv) / n_valid_draws  # 2nd-moment estimator
			v = mm - m ** 2  # variance estimator if MC integral were exact
			# vs = np.nansum(np.square(P * cube.values[i_param] / V_inv - m)) / (n_valid_draws * (n_draws - 1))
			# The above is the sample variance estimate of the integral variance
			stddev = v ** .5
			mean_dict[s_param] = m
			vars_dict[s_param] = v
			s_estimate = f"Estimated {s_param}: mean {round(m, 5)}, std dev: {round(stddev, 5)}."
			print(s_estimate)

		if b_full_covariances:
			raise Exception("Full covariances are currently not implemented.")
		# 	Skeleton, needs fixing:
		# 	for i_param1, _ in enumerate(parameters):
		# 		for i_param2 in range(i_param1):
		# 			mm = np.sum(P * np.square(cube.values[i_param1]) / P_MC) / n_draws
		# 			vv = mm
		# 			cov[i_param1, i_param2] = vv
		# 			cov[i_param2, i_param1] = vv

		result = {'mean': mean, 'cov': cov, 'Var_P': Var_P, 'n_valid_draws': n_valid_draws,
				  'mean_dict': mean_dict, 'vars_dict': vars_dict}
		if b_full_distributions:
			result['cube'] = cube
			result['P'] = P
		return result

	@staticmethod
	def _estimate_POVM_Gaussian(gate_counts: Dict, parameters: List[str]) -> Dict:
		"""Estimate the POVM parameters of a single qubit and their variances.
		The estimation is based on a Gaussian approximation for the experiment results.

		Args:
			gate_counts: A dict with up to six string keys of the calibration gates that were run on the device.
				The supported circuits are 'id', 'x', 'x90p', 'x90m', 'y90p', 'y90m', as given in the list
				self.SUPPORTED_GATES. Each keyed entry consists of gate_counts is a two-element tuple,
				the first is the marginalized counts of the result 0, and the second of the result 1.
				The total number of shots for all gate counts must be identical.
				If the estimation is analytic (using a Gaussian approximation), only the circuits 'id', 'x',
				'x90m', 'y90p' are used. The first two must be passed, and 'y90m', 'x90p' must be passed if the
				'pi_x' and 'pi_y' parameters (respectively) are requested for estimation.
				For Bayesian estimation, all passed circuits are used.
			parameters: A list with up to six string names of the parameters that are to be estimated.
				The supported parameters are 'pi_x', 'pi_y', 'pi_z', 'pi_0', 'z_0', 'epsilon', as given in the list
				self.SUPPORTED_PARAMETERS.
				The order of the parameters in the parameters list determines the order of entries in the output arrays.

		Returns:
			A dictionary with the results and intermediate calculations.
			Among other entries, the dictionary contains a vector of the means estimated for each parameter, and a
			matrix of covariances.
		"""
		param_indices = SPAM1QModel._init_param_indices(parameters)
		mean, cov = SPAM1QModel._init_output(parameters)
		iz = param_indices[SPAM1QModel._I_PI_Z]
		i0 = param_indices[SPAM1QModel._I_PI_0]
		if iz == -1 or i0 == -1:
			raise Exception("The pi_z and pi_0 parameters must be defined for the estimation using a Gaussian"
							" approximation.")
		if param_indices[SPAM1QModel._I_Z_0] != -1 or param_indices[SPAM1QModel._I_EPSILON] != -1:
			raise Exception("The z_0 and epsilon parameters can only be estimated using Bayesian estimation.")

		f0s = gate_counts[('id', 0)]
		shots = f0s[0] + f0s[1]
		f0 = f0s[0] / shots
		fzs = gate_counts[('x', 1)]
		fz = fzs[0] / shots
		pi0 = (f0 + fz) / 2.
		piz = (f0 - fz) / 2.
		Vpi0 = (f0 * (1 - f0) + fz * (1 - fz)) / (4 * shots)
		mean[iz] = piz
		mean[i0] = pi0
		cov[iz, iz] = Vpi0
		cov[i0, i0] = Vpi0  # Identical variance
		mean_dict = {}
		epsilon_mu = np.zeros((4, ))
		epsilon_vars = np.zeros((4, ))
		epsilon_mu[2] = pi0 - piz; epsilon_mu[3] = 1. - (pi0 + piz)
		epsilon_vars[2] = fz * (1 - fz) / shots
		epsilon_vars[3] = f0 * (1 - f0) / shots
		mean_dict['pi_0'] = pi0; mean_dict['pi_z'] = piz

		s_shots_exception = "The number of shots must be identical for all circuits."
		if (fzs[0] + fzs[1]) != shots:
			raise Exception(s_shots_exception)
		ix = param_indices[SPAM1QModel._I_PI_X]
		if ix != -1:
			fxs = gate_counts[('y90p', 2)]
			fx = fxs[0] / shots
			pix = fx - pi0
			if (fxs[0] + fxs[1]) != shots:
				raise Exception(s_shots_exception)
			mean[ix] = pix
			cov[ix, ix] = Vpi0 + fx * (1 - fx) / shots
			epsilon_mu[0] = pix
			epsilon_vars[0] = cov[ix, ix]
			mean_dict['pi_x'] = pix
		iy = param_indices[SPAM1QModel._I_PI_Y]
		if iy != -1:
			fys = gate_counts[('x90m', 3)]
			fy = fys[0] / shots
			piy = fy - pi0
			if (fys[0] + fys[1]) != shots:
				raise Exception(s_shots_exception)
			mean[iy] = piy
			cov[iy, iy] = Vpi0 + fy * (1 - fy) / shots
			epsilon_mu[1] = piy
			epsilon_vars[1] = cov[iy, iy]
			mean_dict['pi_y'] = piy

		for i_param, s_param in enumerate(parameters):
			m = mean[i_param]
			stddev = cov[i_param, i_param] ** .5
			s_estimate = f"Estimated {s_param}: mean {round(m,5)} with std dev: {round(stddev,5)}."
			print(s_estimate)

		result = {'mean': mean, 'cov': cov, 'epsilon_mu': epsilon_mu, 'epsilon_vars': epsilon_vars,
				  'mean_dict': mean_dict}
		return result

	@staticmethod
	def _init_output(parameters: List[str]):
		n_params = len(parameters)
		mean = np.zeros((n_params,))
		cov = np.full((n_params, n_params),
					  np.nan)  # Init to nan, will be overwritten if covariances are calculated
		return mean, cov

	@staticmethod
	def _init_param_indices(parameters: List[str]):
		param_indices = [-1] * SPAM1QModel.NUM_SUPPORTED_PARAMS
		for i_param, s_param in enumerate(parameters):
			param_indices[SPAM1QModel.SUPPORTED_PARAMETERS[s_param.lower()]] = i_param
		return param_indices

	def _pre_qubit_call(self, i_group: int, i_repeat: int, qubit: int, sched: Schedule) -> Schedule:
		"""A hook called before adding a specific gate call to a specific qubit.

		Args:
			i_group: Index of the qubit group.
			i_repeat: Index of the repetition time of the same gate.
			qubit: target qubit.
			sched: The Schedule that is being constructed, that can be manipulated in this hook function
				if overriden by a derived class.

		Returns:
			The (possibly modified) schedule.
		"""
		return sched

	def _post_qubit_call(self, i_group: int, i_repeat: int, qubit: int, sched: Schedule) -> Schedule:
		"""A hook called after adding a specific gate call to a specific qubit.

		Args:
			i_group: Index of the qubit group.
			i_repeat: Index of the repetition time of the same gate.
			qubit: target qubit.
			sched: The Schedule that is being constructed, that can be manipulated in this hook function
				if overriden by a derived class.

		Returns:
			The (possibly modified) schedule.
		"""
		return sched
