"""
Created on Tue Jan  7 17:59:43 2020

@author: Irtaza
"""
import numpy as np

# define Pauli matrices in the computational basis

X = np.array([[0,1], [1,0]], dtype='complex128').reshape(2,2)

Y = np.array([[0,-1j], [1j,0]], dtype='complex128').reshape(2,2)

Z = np.array([[1,0], [0,-1]], dtype='complex128').reshape(2,2)

I = np.eye(2, dtype='complex128')

def vec_basis(qubits):
    """
    Generates a complete cartesian orthonormal basis for 
    the n-qubit.  

    :param qubits: number of qubits
    :type qubits: int
    :return: simplest orthonormal set
    :rtype: np.ndarray, float
    """
    dim = int(2 ** qubits)
    return np.eye(dim).reshape(dim, dim , 1)
    
    
    


def rho_basis(qubits):
    """
    Generates a complete orthonormal basis composed of pure 
    states for the n-qubit density matrix.  

    :param qubits: number of qubits
    :type qubits: int
    :return: pure state spectral decomposition
    :rtype: np.ndarray, float
    """
    return np.array([i*np.conjugate(j).transpose() for i 
            in vec_basis(qubits) for j in vec_basis(qubits)])


def tp(list_of_operators):
    """
    Computes the tensor product of a list of linear operators 
    recursively  to extend to higher dimensions. 
    
    Helper function for nesting np.kron(np.kron(...()...))

    :param list_of_operators: list of linear operators
    :type list_of_operators: List[np.ndarray]
    :return: tensor product of linear operators in list_of_operators
    :rtype: np.ndarray, float
    """
    
    if len(list_of_operators) == 2:
        return np.kron(list_of_operators[0], list_of_operators[1])    
    else:
        return np.kron(tp(list_of_operators[:-1]), list_of_operators[-1])
    
    




import itertools

def krauss_basis(qubits):
    """
    Helper function to return the Krauss operator basis formed by the Cartesian 
    product of [I, X, Y, Z] for the n-qubit.

    :param qubits: number of qubits
    :type qubits: int
    :return: Krauss operator
    :rtype: np.ndarray, float
    """
    
    return [i for i in itertools.product([I, X, Y, Z], repeat=qubits)]


def ko(qubits):
    """
    Generates all the Krauss Operators (ko) for the n-qubit.
    :param qubits: number of qubits
    :type qubits: int
    :return: Krauss operator basis
    :rtype: List[np.ndarray], float
    """
    
    ops = []
    if qubits > 1:
        for i in krauss_basis(qubits):
            ops.append(tp(i))
    else:
        for i in krauss_basis(qubits):
            ops.append(i[0])   # remove a pesky parenthesis 
        
    return ops

def cartesian_product(op_list, repeat=2):
    """
    Helper function that computes the cartesian product for a
    list of linear operators.

    :param op_list: list of linear operators
    :type op_list: List[np.ndarray]
    :param repeat: number of elements multiplied for 1 element in 
                   product, defaults to 2 (cartesian)
    :type repeat: int, optional
    :return: Cartesian product
    :rtype: List[np.ndarray], float
    """

    return [i for i in itertools.product(op_list, repeat=repeat)]



def beta(qubits, input_basis=rho_basis):
    """
    Using the helper functions above, it computes the superoperator 
    that defines the mapping/encoding: E_n rho_j E_m .dagger() in QPT 
    as a dim(qubit)^4 x dim(qubit)^4 matrix in the fixed input basis 
    for the n-qubit.
    
    :param qubits: number of qubits
    :type qubits: int
    :param input_basis: density matrix basis defined by function rho_basis()
    :type input_basis: function, outputs List[np.ndarray]
    :return: Superoperator (tensor)
    :rtype: np.ndarray, float

    """
    
    comp_basis = input_basis(qubits)   # work in the computational basis
    vec_dim = len(comp_basis)
    
    perms = cartesian_product(ko(qubits))   # permutation of acting operators
    
    for j in range(len(comp_basis)):
        for i in range(len(perms)):
        # using trick/Roth's lemma: (A x B.dagger()) * vec(X) = AXB  
        # (but basis vector B is herm here)
            if i == 0:    
                row = np.matmul(np.kron(perms[i][0], perms[i][1]), 
                                comp_basis[j].reshape(vec_dim, 1))[:, np.newaxis]
            else:
                row = np.concatenate((row, 
                            np.array(np.matmul(np.kron(perms[i][0], perms[i][1]), 
                            comp_basis[j].reshape(vec_dim, 1)))[:, np.newaxis]) , axis=2)
        
        state_map = np.squeeze(row, axis=1)  # remove extra dimension here
        
        if j == 0:
            beta = state_map
        else:
            beta = np.concatenate((beta, state_map), axis=0)  # stack for all input states
    
    
    return beta
    
    
# will define the state based on what the qubit number and then order in the list 
# is but as a proof of principle
        
def n_m_state(qubit, n_lex, m_lex):
    """
    Helper function to generate an arbitry n-qubit pure state |n><m|.
    
    :param qubits: number of qubits
    :type qubits: int
    :param n_lex: index n
    :type n_lex: int
    :param m_lex: index m
    :type m_lex: int
    :return: pure state |n><m|
    :rtype: np.ndarray, float
    """
    basis = vec_basis(qubit)
    
    return np.matmul(basis[n_lex], np.conjugate(basis[m_lex].T))


def sensembler(number, obj):
    """
    Helper function that generates an ensemble: a specified number of 
    copies of states or  measurement operator parametrized by the bloch 
    vector or single qubit map. 
    
    :param number: number of copies in ensemble
    :type number: int
    :param obj: Physical object to be copied
    :type obj: np.ndarray or List
    :return: ensemble of obj
    :rtype: np.ndarray, float
    
    """
    if obj.shape == (3,):
        return np.array([[1, obj[0], obj[1], obj[2]] 
                        for _ in range(number)]).reshape(number,4,1)
    if obj.shape == (4,4):
        return np.array([obj for _ in range(number)])
    if obj.shape == (2,2):
        return np.array([obj for _ in range(number)])
    else:
        return np.array([obj for _ in range(number)])
    
    
def ensembler(numbers, objs):
    """
    Vectorize sensemble exploiting functionality bulit into numpy. 
    Both numbers and objs are n-dim arrays.
    
    :param numbers: list of numbers to be copied
    :type objs: List[int] 
    :param objs: List of physical objects to be ensembled.
    :type objs: List[np.ndarray]
    :return: ensemble of objs
    :rtype: np.ndarray, float
    
    """
    
    for i, obj in enumerate(objs):
        if i == 0:
            vensemble = sensembler(numbers[i], obj)
        else:
            vensemble = np.concatenate((vensemble, sensembler(numbers[i], obj) ), axis=0)
    return vensemble



def choi_state(qubits):
    """
    Returns a choi state (in density matrix formalism) that can be used 
    ala' Choi-Jamialkowski Thm to check for complete positivity of a channel.

    :param qubits: number of qubits
    :type qubits: int
    :return: choi state
    :rtype: np.ndarray, float

    """
    
    return np.sum(rho_basis(qubits+1), axis=0) / (2 ** (qubits+1))

def positivity_test(channel, qubits):
    """
    Positivity test of a quantum channel. Choi_ matrix, as a function of 
    choi_state, defined below must have a positive spectrum.

    :param qubits: number of qubits
    :type qubits: int
    :raises Exception: if channel is not positive
    :return: choi state
    :rtype: bool

    """
    
    choi_matrix = np.matmul(np.kron(np.eye(2), channel), choi_state(qubits))
    spectrum = np.linalg.eig(choi_matrix)[0]
    
    if False in (np.round(spectrum, 10) >= 0):  # check for >= 0 eig vals
        raise Exception("Map not completely positive as Choi matrix is not positve!")
        
    return True

def unitary(bloch_vec, theta, phase):
    """
    Polar representation of a 1-unitary as a 3D vector, up to a phase isomorphism.

    :param bloch_vec: normalized, axis vector
    :type bloch_vec: np.ndarray, float
    :param theta: angle_1
    :type theta: float
    :param phase: angle_2
    :type phase: float
    :raises NotImplementedError: if Bloch vector is not normalized or 
                                 if theta or phi are outside boundaries
    :return: 1-unitary, a 2 X 2 hermitian matrix 
    :rtype: np.ndarray, float
    """
    if not np.allclose(np.sum(np.array(bloch_vec)*np.array(bloch_vec)), 1):
        raise NotImplementedError("Bloch vector needs to be normalized")
    if theta < 0 or theta > np.pi:
        raise NotImplementedError("Theta outside [0, pi)")
    if phi < 0 or phi > 2*np.pi:
        raise NotImplementedError("Phi outside [0, 2*pi)")
    

    return (np.cos(theta*0.5)*np.eye(2) + 
            -1j*np.sin(theta*0.5)*(bloch_vec[0]*X+bloch_vec[1]*Y+bloch_vec[2]*Z))*np.exp(1j*phase)

def tstate(channel, state):
    """ 
    Transformed state and/or set. 
    rho -> E(rho) applied on an informationally complete element or specified input set.

    :param channel: quantum channel in diagonalized krauss form or unitary form.
    :type channel: List[np.ndarray], float
    :param state: density matrix of a single state (pure or mixed)
    :type state: np.ndarray, float
    :return: transformed state
    :rtype: np.ndarray, float
    """
    
    if len(channel.shape) == 2:  # i.e. unitary
        trho = np.matmul(np.matmul(channel, state), np.conjugate(channel.T))
        
    else:
        
        trho = np.sum(np.matmul(np.matmul(channel, state), np.conjugate(np.transpose(channel, 
                            axes=(0,2,1)))), axis=0)
    
    return trho

def tstate_nu(channel, channel_dag, state):
    """ 
    Transformed state and/or set: rho -> E(rho)

    :param channel: left undiagonalized krauss operators of the quantum channel
    :type channel: List[np.ndarray], float
    :param channel_dag: right undiagonalized krauss operators of the quantum channel
    :type channel_dag: List[np.ndarray], float
    :param state: density matrix of a single state (pure or mixed)
    :type state: np.ndarray, float
    :return: transformed state
    :rtype: np.ndarray, float
    """

    trho = np.sum(np.matmul(np.matmul(channel, state), channel_dag), axis=0)
    return trho



