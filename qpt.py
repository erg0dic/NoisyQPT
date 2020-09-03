import numpy as np

from state_tomography import state_tomography, noisy_state_tomography, noisy_rho
from tensor_algebra import *


class Sqpt_protocol(object):
    def __init__(self, channel, input_basis_vectors, 
                qubits, measurements, noise_level=0, 
                noisy_axis=(False, False, False), t=tstate):
        """
        :param channel: quantum channel 
        :type channel: np.ndarray
        :param input_basis_vectors: input states in the computational basis
        :type input_basis_vectors: np.ndarray, flaot
        :param qubits: number of qubits
        :type qubits: int
        :param measurements: number of measurements
        :type measurements: int
        :param noise_level: percentange of noise between 0 and 1
        :type noise_level: float
        :param noisy_axis: select which orthogonal axis needs to be noisy
        :type noisy_axis: (bool, bool, bool)
        :param tstate: transformed state
        :type tstate: function
        """

        self.channel = channel
        self.input_basis_vectors = input_basis_vectors
        self.qubits = qubits
        self.measurements = measurements
        self.noise_level = noise_level
        self.noisy_axis = noisy_axis
        self.t = t 
        

    def ics(self):
        """
        Informationally complete set of input basis vectors transformed by the channel.
        Helper function that generates a input basis for the sqpt function below.

        :return: transformed informationally complete set of states
        :rtype: np.ndarray, float
        """
        
        
        ics=[]
        
        diags = [] # diagonal meas
        for n in self.input_basis_vectors:
            diags.append(noisy_rho(self.t(self.channel, np.matmul(n, np.conjugate(n).T)), 
                            self.qubits, self.measurements, 
                            self.noise_level, self.noisy_axis))
            
        diags = np.array(diags, dtype='complex128')

        # off diag meas using orthogonal inital state set
        for index, n in enumerate(self.input_basis_vectors):
            for m in self.input_basis_vectors:
                if False in (n == m):          
                    plus = (n + m) / np.sqrt(2)
                    minus = (n + 1j*m) / np.sqrt(2)
    #                print(plus, minus)
                    pp = noisy_rho(self.t(self.channel, np.matmul(plus, np.conjugate(plus).T)),
                                self.qubits, self.measurements, 
                                self.noise_level, self.noisy_axis)
    #                print(pp)                
                    mimi = noisy_rho(self.t(self.channel, np.matmul(minus, np.conjugate(minus).T))
                            , self.qubits, self.measurements, self.noise_level, self.noisy_axis)
    #                print(mimi)

                    nm = pp + 1j*mimi -0.5*(1+1j)*(np.sum(diags, axis=0))
    #               print(nm)
                    ics.append(nm)
                    
                else:
                    nn = diags[index]
                    
                    ics.append(nn)
        
        ics = np.array(ics)
        
        shape = ics.shape
        
        ics = ics.reshape(shape[0], shape[1]*shape[1],1)  # turn into vectors of rhos
                    
                
        return ics
            

           
    def sqpt(self, tstate_vec, want_chi=False):
        """
        Standard quantum process tomography. Returns krauss operators by inverting the basis 
        mapping of the transformed channel states, taken as input.

        :param tstate_vec: vectorized density matrix basis taken as input states
        :type tstate_vec: np.ndarray, float
        :param want_chi: for debugging, get Chi, the transformed states' tensor
        :type want_chi: bool
        :return: Reconstructed quantum channel in Krauss operator representation
        :rtype: np.ndarray, float
        """

        b = beta(self.qubits)
    #    canonical form with nested paulis for 1-qubit, good debugging test
    #    b[[2, 4]] = b[[4, 2]]
    #    b[[3, 5]] = b[[5, 3]]
    #    
    #    b[[10,12]] = b[[12,10]]
    #    b[[11,13]] = b[[13,11]]
    #    
        chi_vec = np.matmul(np.linalg.inv(b), tstate_vec.flatten()[:, np.newaxis])
        
        cshape = chi_vec.shape
        sq_mat = int(cshape[0] / np.sqrt(cshape[0]))
        chi = chi_vec.reshape(sq_mat, sq_mat)

        if want_chi == True:
            return chi   
        
        cels = chi.flatten()[:, np.newaxis, np.newaxis]  # chi elements. add extra axis for below operation
        ob = np.array(cartesian_product(ko(self.qubits), repeat=2))  # operator basis, both E_n and E_m
        
        
        Krauss = np.multiply(cels, ob[:, 0])   # weight the Krauss operators with chi elements
        
        Krauss_dag = np.conjugate(ob[:, 1])    # to match the definition of Chi I take the conjugate only
        
        return Krauss, Krauss_dag

        ##################### ignore this bit for now ########################################
        # Edit: Diagonalization does not work and I suspect that has to do with complex value preservation of Chi elements 
        #       Ex: Try checking it yourself by commenting out the last line above and uncommenting all of below.
        
    #    eigvals, eigvecs = np.linalg.eig(chi)  
    #                   #  the latter has orthogonal columns but need not be unitary i.e. orthogonal!
    #    
    ##    return eigvals, eigvecs
    #    
    #    e = np.matmul(eigvecs, np.sqrt(np.diag(eigvals)))  # decompose as rescaled unitary
    #    v = np.matmul(np.sqrt(np.diag(eigvals)), np.linalg.inv(eigvecs))   # and its inverse
    #    
    #
    #    
    #    kbasis = np.array(ko(qubits))
    #    channel = np.sum(np.multiply(e.T[:, :, np.newaxis, np.newaxis], sensembler(len(kbasis),kbasis)), axis=1)
    #    
    #    channel_dag = np.sum(np.multiply(np.conjugate(v)[:, :, np.newaxis, np.newaxis], 
    #                                     sensembler(4,np.array(ko(1)))), axis=1)   # the inverse of the above ops
    #    
    #   
    #    return channel, channel_dag
        #return chi_mat


    def oics(self, qplist, c):
    """
    Informationally overcomplete set of input basis vectors transformed by the channel.
    Physical tests are commented out as print statements. Use them at your own leisure.
    More dedicated test scripts will be available in due time.

    --> NB: qplist can have multiple copies of the same measurement axis
        bloch parameters (theta, phi) or different theta/phi.
    --> Also NB: Computational basis axis is fixed.

    :param qplist: input vector basis configuration  
    :type qplist: List or np.array, float
    :param c: corresponding coefficients for qplist computable via param_optimizer routine
    :type c: List, float
    :return: tomographed, transformed, informationally overcomplete basis 
    :rtype: np.ndarray, float
    """

        oics=[]
        input_basis_vectors = vec_basis(self.qubits)
        diags = np.matmul(vec_basis(self.qubits), np.transpose(
                                                    vec_basis(self.qubits), axes=(0, 2, 1)))
        dim = int(2**(self.qubits))
        k = 0
        for n in self.input_basis_vectors:
            for m in self.input_basis_vectors:
                if False in (n == m):
                    new_cos = c[k]  # precomputed coefficients for off diagonal decomposition
                
                    fin = np.zeros(dim * dim, dtype='complex128').reshape(dim, dim)
                    
                    
                    for i in range(len(qplist)):
                        fin += new_cos[i]*noisy_rho(self.t(self.channel, rho(qplist[i][0], 
                        qplist[i][1])), self.qubits, self.measurements, self.noise_level, 
                                                                        self.noisy_axis)
                        #print(rho(qplist[i][0], qplist[i][1])-
                        #      noisy_rho(rho(qplist[i][0], qplist[i][1]), qubits, 
                        #      measurements=1000, noise_level=0.0, 
                        #      noisy_axis=(False, False, False)))
                    
                    for j in range(len(diags)):
                        fin += new_cos[len(qplist)+j]*noisy_rho(self.t(channel, diags[j]), 
                                                                self.qubits, self.measurements, 
                                                                self.noise_level, self.noisy_axis)
                    #print(diags[j] - noisy_rho(diags[j], qubits, 
                    #       measurements=1000, noise_level=0.0, noisy_axis=(False, False, False)))  
                
                    oics.append(fin)
                    k+=1   
                        #return []
                else:
                    nn = noisy_rho(self.t(self.channel, np.matmul(n, np.conjugate(n).T))
                                , self.qubits, self.measurements, 
                                self.noise_level, self.noisy_axis)   # in the z basis
                        
                    oics.append(nn)
        oics = np.array(oics)

        shape = oics.shape

        oics = oics.reshape(shape[0], shape[1]*shape[1],1)

        return oics

    def oics_qt(self, qplist, c):
        """
        Modified oics: updated for unfixed basis.

        :param qplist: input vector basis configuration;
                       now necessarily all 4 basis vectors need to be specified
        :type qplist: List or np.array, float
        :param c: corresponding coefficients for qplist computable via param_optimizer routine
        :type c: List, float
        :return: tomographed, transformed, informationally over and/or complete basis 
        :rtype: np.ndarray, float
        
        """

        oics=[]
        input_basis = rho_basis(self.qubits)
        dim = int(2**(self.qubits))
        k = 0
        for basis in input_basis:
            new_cos = c[k]  # precomputed coefficients for off diagonal decomposition
                
            fin = np.zeros(dim * dim, dtype='complex128').reshape(dim, dim)
            
            for term in range(len(qplist)):   # for all the terms in the input basis set       
                fin += new_cos[term]*noisy_rho(self.t(self.channel, rho(qplist[term][0], 
                        qplist[term][1])), self.qubits, self.measurements, 
                        self.noise_level, self.noisy_axis)
                        
            oics.append(fin)
            k+=1   

        oics = np.array(oics)
        
        shape = oics.shape
        
        oics = oics.reshape(shape[0], shape[1]*shape[1],1)
        
        return oics

from qutip import kraus_to_super, dnorm, Qobj


# TODO: 1. maybe put the dn metric calculator in tensor_algebra.py
#       2. Definitely call the cvxpy routine directly without the qutip wrapper
#          (might be more trickier and technical than I thought, plus it's already
#           been implemented by the QuTip team so might as well go along with it
#           after I've understood how it works.)            


def dn(estimated_k, estimated_kd, true, qubits, true_super=False):
    """
    Figure of merit for comparing the estimated and true channel for judging the 
    performance of the tomography scheme. Takes both estimated and true channel 
    krauss operators and transforms both into super operators and then computes 
    the diamond norm (Kitaev et. al.). A semidefinite program formulation of dnorm
    by Watrous et. al. (2013) already thankfully exists in the QuTip library along 
    with adequate testing e.g. CPTP checks, special cases like Brutza ensembles etc. The 
    program is solved by specifying the triple in cvxopt in cvxpy, so that should 
    library is also a dependency in addition to qutip. Solving triples in cvxpy is 
    just a matter of learning the syntax but the diamond norm is more trickier to 
    implement. See qutip.metrics http://qutip.org/docs/4.0.2/modules/qutip/metrics.html 
    for more detail.

    Long story short, in the interest of saving time, rather than reinventing the wheel 
    using cvxpy, I have decided to use the QuTip implementation as it seems quite well 
    thought out and technical. I'm thankful to the QuTip team and Christopher Granade 
    for making something highly specialist available. Please check them out using the link 
    above.

    :param estimated_k: learnt left krauss operators of the tomographed channel
    :type estimated_k: np.ndarray, float
    :param estimated_kd: learnt right krauss operators of the tomographed channel
    :type estimated_k: np.ndarray, float
    :param qubits: number of qubits
    :type qubits: int
    :param true: true channel
    :type true: np.ndarray, float
    :param want_chi: for debugging, get Chi, the transformed states' tensor
    :type want_chi: bool
    :return: Reconstructed quantum channel in Krauss operator representation
    :rtype: np.ndarray, float

    """
    kraus_dim = int(2**(2*qubits))
    
    if not true_super:  # if true is not already a super operator
        #ac = kraus_to_super([Qobj(i) for i in true])    # actual channel
        ac = Qobj(np.array(
            [[tstate(true, rho_basis(qubits)[i])] 
            for i in range(kraus_dim)]).flatten().reshape(kraus_dim,kraus_dim), 
            type='super')
    else:
        ac = Qobj(true, type='super')
    
    ec = Qobj(np.array([[tstate_nu(estimated_k, 
            estimated_kd, rho_basis(qubits)[i])] 
            for i in range(kraus_dim)]).flatten().reshape(kraus_dim,kraus_dim), type='super')
    
    return dnorm(ec, ac)







