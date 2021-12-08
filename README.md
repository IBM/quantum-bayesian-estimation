# quantum-bayesian-estimation

## Scope

Python source code for Bayesian estimation of IBM Quantum qubit device parameters using an analytic approach.

The code in this package generates Pulse schedules (that can be executed directly or together with other schedules), for the estimation of single-qubit parameters of state preparation and measurement (SPAM) errors, and gate errors, employing analytic expressions for the supported parameters.  

The theory and modelling supported by this code, together with results from estimation experiments, are described in the following paper:

https://arxiv.org/abs/2108.10686

Experiments using this code can be run on quantum hardware (such as [IBM Quantum](https://quantum-computing.ibm.com/) devices accessible via the cloud) using [Qiskit](https://qiskit.org/), see also the [Qiskit repositories on github](https://github.com/Qiskit).

## Documentation

See the inline documentation:
* [The main class](quantum_bayesian_estimation/spam_1q_model.py) `SPAM1QModel`, and also the other library files.
* [An example](examples/SPAM-Device-Basic-1.py) that can be easily run to get you started. 

## Notes

If you have any questions or issues you are welcome to contact the project maintainer.

For more information see:

* [LICENSE](LICENSE)
* [MAINTAINERS](MAINTAINERS.md)
* [CHANGELOG](CHANGELOG.md)
