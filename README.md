Noisy Quantum Process Tomography (MSci project)
================================


NoisyQPT contains the code library (rather a collection of related scripts) for synthetic data generation and jupyter notebooks for data-science like analysis, as part of my MSci project. It essentially implements multinomial sampling and semi-definite programming to benchmark single qubit quantum process tomography (QPT). This is then used to
optimize the input stage of 1-qubit QPT by searching through input state space via brute-force and intelligently following symmetry based arguments.

Please open up an issue if you have a problem. 
<p align="center">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
            alt="python"></a> &nbsp;
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg"
            alt="MIT license"></a> &nbsp;
</p>

## Some example uses
It's pretty easy to start optimizing for a noisy input basis where the computational basis is still fixed
```python
theta1, phi1 = 0.5*np.pi, np.pi
theta2, phi2 = 0.5*np.pi, np.pi*0.5
qplist = [[theta1, phi1], [theta2, phi2]]
measurements=1000
noise=0.1
# some incantations
from qpt import Sqpt_protocol
from param_optimizer import QPTparaopt
def channel_dependent_sqpt(channel=damp, measurements=measurements) -> Sqpt_protocol:
        return Sqpt_protocol(channel=channel, noise_level=noise, 
        noisy_axis=(True, False, False), measurements=measurements, qubits=1)

proto = channel_dependent_sqpt(damp)
opt = QPTparaopt(qubit_number=1, qparalist=qplist)
coeffs, basis = proto.sqpt(proto.oics(qplist=qplist, c=opt.cgen()))
## then call some fidelity measure e.g diamond norm and compare with the theoretical channel's transformed states
print(f"The accuracy for {measurements} measurements is {1-dn(coeffs, basis, damp)} and noise is {noise}")
```
For a completely unfixed basis, to save time, one can specify an ansatz for the basis.
For a specific basis e.g. `hexagonal` it is sufficient to specify the points on the bloch sphere, and then test QPT performance for a choice of channel e.g. x_rotation.
Specify the noise level and some binomial sample count.

This is done by the following code in `datacoll.py`,
```python
hexagonal = [[0,0], [np.pi, 0], [np.pi*0.5, np.pi], [np.pi*0.5, np.pi*0.5], [np.pi*0.5, 0], [np.pi*0.5, np.pi*1.5]]
channels = {"rotx90":x_rotation}
configs = {"hexs":hexagonal}
noise=0.3 # gaussian noise level in the measurements    
for config_name in configs:
    for channel_name in channels:
        m = int(40000 / len(configs[config_name]))
        x = Config(channel_name, channels[channel_name], noise, 
            (False, False, True),configs[config_name], config_name, 
            np.arange(10,m,100), "cob")        
        x.set_iterations(100) # averaging results over 100 runs
        x.record()
```
The results can be visualized by code in the companion notebook `two_plus_rotation_analysis.ipynb`. 
## Some semantics and general layout/design talk
Code is pretty modular. The Noisy state tomography protocol is specified in `state_tomography.py` and the standard process tomography protocol is specified in `qpt.py` that can use an arbitrary input basis (number of points on the bloch sphere) that can be obtained from `param_optimizer.py` and are mapped to reconstruct a single basis element in the standard protocol. 