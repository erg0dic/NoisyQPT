# channels
import numpy as np
from misc_utilities import damp
from tensor_algebra import *
from state_tomography import *
from qpt import dn

def depol(param, state):
     return np.matmul((1-param)*np.eye(len(state)), state) + 0.5*(param)*np.eye(len(state))   
        
def dep(param, qubits=1):
    return np.multiply(np.array([np.sqrt(1-0.75*param), 0.5*np.sqrt(param), 0.5*np.sqrt(param), 
                                 0.5*np.sqrt(param)])[:, np.newaxis, np.newaxis],np.array(ko(qubits)))    

theta1, phi1 = 0.5*np.pi, np.pi
theta2, phi2 = 0.5*np.pi, np.pi*0.5
qplist = [[theta1, phi1], [theta2, phi2]]

# some incantations
from qpt import Sqpt_protocol
from param_optimizer import QPTparaopt
def channel_dependent_sqpt(channel=damp, measurements=1000) -> Sqpt_protocol:
        return Sqpt_protocol(channel=channel, noise_level=0.1, 
           noisy_axis=(True, False, False), measurements=measurements, qubits=1)

proto = channel_dependent_sqpt(damp)
opt = QPTparaopt(qubit_number=1, qparalist=qplist)


# tests: 

# test ta 1:

np.allclose(np.kron(np.kron(I, I), X), tp([I, I, X]))   

# test ta 2:
trho = np.array([[1,1],[1,1]])
projectors =   np.matmul(np.linalg.eig(ko(1))[1][:,:,:, np.newaxis],
               np.transpose(np.conjugate(np.linalg.eig(ko(1))[1][:,:,:, np.newaxis]), 
               axes=(0,1,3,2)))
states = sensembler(len(projectors)*int(2**1), trho).reshape(projectors.shape)
bin_vecs = np.trace(np.matmul(projectors, states), axis1=2, axis2=3)

eigvals = np.linalg.eig(ko(1))[0]
op_addition = np.multiply(bin_vecs, eigvals)

np.kron(trho, np.eye(2))
projectors.shape
trho2 = np.kron(trho, np.eye(2))
states = sensembler(len(projectors)*int(2**2), trho2)

# test ta 3:

trho = np.sum(0.5*rho_basis(1), axis=0) 
trho2 = np.kron(trho, np.eye(2)) / 2.
trho3 = np.kron(trho2, np.eye(2)) / 2

def hsip(mat1, mat2):
    "Hilbert-Schmidt inner product that generalizes the inner product in higher dimensions of CES"
    return np.trace(np.matmul(np.conjugate(mat1.T), mat2))


[hsip(i[0], i[1]) for i in cartesian_product(ko(2), repeat=2)]  # inner product of bases vectors is ortho

print(np.allclose(state_tomography(trho, 1, cheat=True), trho),
np.allclose(state_tomography(trho2, 2, cheat=True), trho2),
np.allclose(state_tomography(trho3, 3, cheat=True), trho3))

# damping channel (all computational states under action), the final state is not converging, edit: fixed the main code

c, cd = proto.sqpt(proto.ics())
print(tstate_nu(c, cd, rho_basis(1)[0]),
tstate_nu(c, cd, rho_basis(1)[1]),
tstate_nu(c, cd, rho_basis(1)[2]),
tstate_nu(c, cd, rho_basis(1)[3]))



# for depolarising channel the method is convergent
proto = channel_dependent_sqpt(channel=dep(0.8), measurements=10000)
c, cd = proto.sqpt(proto.ics())
print(tstate_nu(c, cd, rho_basis(1)[0]),

tstate_nu(c, cd, rho_basis(1)[1]),

tstate_nu(c, cd, rho_basis(1)[2]),

tstate_nu(c, cd, rho_basis(1)[3])
)

## Test ####

tets = [[np.arcsin(np.sqrt(2/3)), np.pi*0.25], [np.arcsin(np.sqrt(2/3)), np.pi+np.pi*0.25], 
         [np.pi-np.arcsin(np.sqrt(2/3)), np.pi-np.pi*0.25], 
         [np.pi-np.arcsin(np.sqrt(2/3)), 2*np.pi-np.pi*0.25] ] # tetrahedral bloch vecs

 

print(proto.oics(qplist=qplist, c=opt.cgen()))


if __name__ == '__main__':
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
    print(f"The accuracy for {measurements} measurements is {1-dn(coeffs, basis, damp)} and noise is {noise}")

#measurements = np.arange(10,10000, 20)
#c1, c2, c3, c4 = [], [], [], []
#for index, channel in enumerate([damp, dephase, dep(0.2), x_rotation]):
#    for i in measurements:
#        c, cd = sqpt(ics(channel, vec_basis(1), 1, i, noise_level=0.2, noisy_axis=(False, False, False)), 1)
#        if index == 0:
#            c1.append(dn(c, cd, channel, 1))
#        if index == 1:
#            c2.append(dn(c, cd, channel, 1))
#        if index == 2:
#            c3.append(dn(c, cd, channel, 1))
#           
#        if index == 3:
#            c4.append(dn(c, cd, channel, 1))
#plt.figure()
#
#plt.plot(measurements, c1, label='Amplitude dampening', alpha=0.5)
#plt.plot(measurements, c2, label='Dephasing', alpha=0.5)
#plt.plot(measurements, c3, label='Depolarizing', alpha=0.5)
#plt.plot(measurements, c4, label='pi*0.5 rotation', alpha=0.5)
#plt.xlabel('0% noise measurements')
#plt.ylabel('Dnorm')
#plt.legend()
#
#measurements = np.arange(10,3000, 20)
#c1, c2, c3 = [], [], []
#for index, channel in enumerate([damp, dephase, dep(0.2)]):
#    for i in measurements:
#        c, cd = sqpt(ics(channel, vec_basis(1), 1, i, noise_level=0.3, noisy_axis=(True, False, False)), 1)
#        if index == 0:
#            c1.append(dn(c, cd, channel, 1))
#        if index == 1:
#            c2.append(dn(c, cd, channel, 1))
#        if index == 2:
#            c3.append(dn(c, cd, channel, 1))
#            
##            
##plt.figure()
##
#plt.plot(measurements, c1, label='Amplitude dampening', alpha=0.5)
#plt.plot(measurements, c2, label='Dephasing', alpha=0.5)
#plt.plot(measurements, c3, label='Depolarizing', alpha=0.5)
#plt.xlabel('30% x-noise measurements')
#plt.ylabel('Dnorm')
#plt.legend()
#
#measurements = np.arange(10,3000, 20)
#c1, c2, c3 = [], [], []
#for index, channel in enumerate([damp, dephase, dep(0.2)]):
#    for i in measurements:
#        c, cd = sqpt(ics(channel, vec_basis(1), 1, i, noise_level=0.3, noisy_axis=(False, True, False)), 1)
#        if index == 0:
#            c1.append(dn(c, cd, channel, 1))
#        if index == 1:
#            c2.append(dn(c, cd, channel, 1))
#        if index == 2:
#            c3.append(dn(c, cd, channel, 1))
#            
#            
#plt.figure()
#
#plt.plot(measurements, c1, label='Amplitude dampening', alpha=0.5)
#plt.plot(measurements, c2, label='Dephasing', alpha=0.5)
#plt.plot(measurements, c3, label='Depolarizing', alpha=0.5)
#plt.xlabel('30% y-noise measurements')
#plt.ylabel('Dnorm')
#plt.legend()
#
#measurements = np.arange(10,3000, 20)
#c1, c2, c3 = [], [], []
#for index, channel in enumerate([damp, dephase, dep(0.2)]):
#    for i in measurements:
#        c, cd = sqpt(ics(channel, vec_basis(1), 1, i, noise_level=0.3, noisy_axis=(False, False, True)), 1)
#        if index == 0:
#            c1.append(dn(c, cd, channel, 1))
#        if index == 1:
#            c2.append(dn(c, cd, channel, 1))
#        if index == 2:
#            c3.append(dn(c, cd, channel, 1))
#            
#            
#plt.figure()
#
#plt.plot(measurements, c1, label='Amplitude dampening', alpha=0.5)
#plt.plot(measurements, c2, label='Dephasing', alpha=0.5)
#plt.plot(measurements, c3, label='Depolarizing', alpha=0.5)
#plt.xlabel('30% z-noise measurements')
#plt.ylabel('Dnorm')
#plt.legend()
#
#measurements = np.arange(10,3000, 20)
#c1, c2, c3 = [], [], []
#for index, channel in enumerate([damp, dephase, dep(0.2)]):
#    for i in measurements:
#        c, cd = sqpt(ics(channel, vec_basis(1), 1, i, noise_level=0.2, noisy_axis=(True, True, True)), 1)
#        if index == 0:
#            c1.append(dn(c, cd, channel, 1))
#        if index == 1:
#            c2.append(dn(c, cd, channel, 1))
#        if index == 2:
#            c3.append(dn(c, cd, channel, 1))
#            
#            
#plt.figure()
#
#plt.plot(measurements, c1, label='Amplitude dampening', alpha=0.5)
#plt.plot(measurements, c2, label='Dephasing', alpha=0.5)
#plt.plot(measurements, c3, label='Depolarizing', alpha=0.5)
#plt.xlabel('20% all-axis noise measurements')
#plt.ylabel('Dnorm')
#plt.legend()



#terms = 2
#
#for phi2 in np.linspace(0, np.pi*2, 10):
#    for phi in np.linspace(0, np.pi*2, 10):
#        try:
#            
#            qplist = [[0.5*np.pi, phi], [0.5*np.pi, phi2]]
#            cos = coeffs(f, rho_basis(1)[1], random=False, qplist=qplist, tol=1e-8)
#            new_cos = test_offdiag(cos, qplist, rho_basis(1)[1])
#            qubits = 1
#            dim = int(2**(qubits))
#            fin = np.zeros(dim * dim, dtype='complex128').reshape(dim, dim)
#            
#            diags = np.matmul(vec_basis(qubits), np.transpose(vec_basis(qubits), axes=(0, 2, 1)))
#            
#            for i in range(len(qplist)):
#                fin += new_cos[i]*tstate(damp, noisy_rho(rho(qplist[i][0], qplist[i][1]), qubits, 
#                                       measurements=100, noise_level=0.3, noisy_axis=(True, False, False)))
#                #print(rho(qplist[i][0], qplist[i][1])-noisy_rho(rho(qplist[i][0], qplist[i][1]), 
#                               qubits, measurements=1000, noise_level=0.0, noisy_axis=(False, False, False)))
#            
#            for j in range(len(diags)):
#                fin += new_cos[len(qplist)+j]*tstate(damp, noisy_rho(diags[j], 
#                                   qubits, measurements=100, noise_level=0.3, noisy_axis=(True, False, False)))
#                #print(diags[j] - noisy_rho(diags[j], qubits, measurements=1000, 
#                                   noise_level=0.0, noisy_axis=(False, False, False)))  
#            
#            print(np.round(fin, 3))
#            print(phi, phi2)
#        except NotImplementedError as e:
#            print("{} has failed for phi = {}".format(e, phi / (2*np.pi)))