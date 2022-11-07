
import numpy as np
from state_tomography import *
from scipy.optimize import minimize
from misc_utilities import rho
class QPTparaopt(object):
    """
    A routine for computing coefficients that go with an arbitrary physically allowed 
    decomposition of an input quantum state. This will be useful for doing parametrizble 
    QPT.
    """
    def __init__(self, qubit_number=1, qparalist=None, tolerance=1e-6, no_of_dec_terms=2):
        """
        :param qubit_number: number of qubits
        :type qubit_number: int
        :param qparalist: which input state needs its basis changed
        :type qparalist: List, float
        :param tolerance: floating point accuracy
        :type tolerance: float, Defaults to 1e-6
        :param no_of_dec_terms: number of decomposable terms for the coefficients
        :type no_of_dec_terms: int, >= 2, Defaults to 2
        """

        self.qubits = qubit_number
        self.qubit_params_list = qparalist
        self.tol = tolerance
        self.terms = no_of_dec_terms


    def f(self, coeffs, off_diag):
        """
        Define the optimization function for computing coefficients for an arbitrary 
        basis input state, that will be used in QPT. One or more basis elements need 
        to be specified in ``qubit_params_list`` and the remaining are filled in through 
        the optimization routine. This is done to reconstruct an off-diagonal "coherent" 
        element that forms an input state during channel/process tomography reconstruction.

        :param coeffs: coefficients to be computed
        :type coeffs: np.ndarray or List, float
        :param off_diag: |n><m| where n != m that is decomposed into basis elements 
                        whose corresponding coefficients are computed
        :type off_diag: np.ndarray, float
        :return: coefficients parametrizing the off-diagonal element
        :rtype: np.ndarray, float
        """
        coeffs = np.array(coeffs, dtype='complex128')
        coeffs = coeffs[:int(len(coeffs)/2)] + 1j*coeffs[int(len(coeffs)/2):]
    #    print(coeffs)
        dim = int(2 ** self.qubits)
        res = np.zeros(dim * dim, dtype='complex128').reshape(dim, dim)
        
        for i, param in enumerate(self.qubit_params_list):
            res += coeffs[i]*rho(param[0], param[1])
    #    print(res)
        for i, param in enumerate(self.qubit_params_list):
            res -= coeffs[i]*np.diag(np.diag((rho(param[0], param[1])))) # diag mat of diag of mat
    #    print(res)   
        return np.sum(np.abs(np.real(res)-np.real(off_diag))) + np.sum(np.abs(np.imag(res)-np.imag(off_diag)))


    def f_qt(self, coeffs, rho_basis_vec):
        """
        Optimization objective function.Same as f but for a parameter list that 
        parametrizes 4 input states, coeffs will have 8 parameters. Rho_basis_vec 
        replaces off-diag as there is now no special diagonal input basis.

        :param coeffs: coefficients to be computed
        :type coeffs: np.ndarray or List, float
        :param rho_basis_vec: |n><m| for any n,m: arbitrary basis element
        :type rho_basis_vec: np.ndarray, float
        :return: coefficients parametrizing the density matrix basis vector
        :rtype: np.ndarray, float
        """
        coeffs = np.array(coeffs, dtype='complex128')
        coeffs = coeffs[:int(len(coeffs)/2)] + 1j*coeffs[int(len(coeffs)/2):]
        dim = int(2 ** self.qubits)
        res = np.zeros(dim * dim, dtype='complex128').reshape(dim, dim)
        
        for i, param in enumerate(self.qubit_params_list):
            res += coeffs[i]*rho(param[0], param[1])
        
        return np.sum(np.abs(np.real(res)-np.real(rho_basis_vec))) + np.sum(np.abs(np.imag(res)-np.imag(rho_basis_vec)))



    def coeffs(self, f, offdiag, random=False):
        """
        Optimization routine using sequential least squares minimization for f above.
        A random initialization of coefficients is used. If no qubit 
        parametrization is provided, then random should be set to True
        and a random qplist parametrization will be chosen. More terms can also be added 
        for each input basis parametrization that can be solved forbut the bare 
        needed is 2.

        :param f: objective function
        :type f: bounded method, self.func
        :param offdiag: off-diagonal element whose coefficient needs to be estimated
        :type offdiag: np.ndarray, float
        :param random: switch to use random input vector basis configuration
        :type random: bool, defaults to False
        :raises Exception: if a diagonal is specified, as computational basis is fixed.
        :raises NotImplementedError: if the routine is not successful after 1000 maxiterations
        :return: Optimized coefficients parametrizing the density matrix basis vector
        :rtype: np.ndarray, float
        """

        if np.allclose(np.trace(offdiag), 0) != True:
            raise Exception("Make sure the element whose coeffs are to be estimated is actually off-diagonal!")
        
        if random == True:
            self.qubit_params_list = [[np.random.random(1)*np.pi, np.random.random(1)*2*np.pi] for _ in range(self.terms)]
            
        opt = minimize(self.f, np.random.random(len(self.qubit_params_list)*2), 
                    args=(offdiag), method='SLSQP', 
                    options={'maxiter': 1000, 'ftol': 1e-15, 'eps': 1.4906e-12})
        
        if opt.fun > self.tol:
            raise NotImplementedError(
                    "The optimization wasn't successful as {} is less than specified tol {}"
                    .format(opt.fun, self.tol)
                    )
        c = opt.x
        
        cs = c[:int(len(c)/2)] + 1j*c[int(len(c)/2):]  # combine the complex with reals
        
        if random==True:
            return cs, self.qubit_params_list
        
        return cs


    def coeffs_qt(self, f_qt, rho_basis_vec, random=False):
        """
        Optimization routine using sequential least squares minimization for f_qt above.
        The basis is unfixed so no special diagonal vector exists.

        :param f: objective function
        :type f: bounded method, self.func
        :param rho_basis_vec: density matrix element whose coefficient needs to be estimated
        :type rho_basis_vec: np.ndarray, float
        :param random: switch to use random input vector basis configuration
        :type random: bool, defaults to False
        :raises NotImplementedError: if the routine is not successful after 1000 maxiterations
        :return: Optimized coefficients parametrizing the density matrix basis vector
        :rtype: np.ndarray, float
        """

        if random == True:
            self.qubit_params_list = [[np.random.random(1)*np.pi, np.random.random(1)*2*np.pi] for _ in range(self.terms)]
            
        opt = minimize(f_qt, np.random.random(len(self.qubit_params_list)*2), 
                    args=(rho_basis_vec), method='SLSQP', 
                    options={'maxiter': 1000, 'ftol': 1e-15, 'eps': 1.4906e-12})
        
        if opt.fun > self.tol:
            raise NotImplementedError("The optimization wasn't successful as {} is less than specified tol {}".format(opt.fun, self.tol))
        

        c = opt.x
        
        cs = c[:int(len(c)/2)] + 1j*c[int(len(c)/2):]  # combine the complex with reals
        
        if random==True:
            return cs, self.qubit_params_list
        
        return cs

    


    def test_offdiag(self, coeffs, offdiag):
        """
        Test function for the fixed optimization routine. 
        Raises exception if unsuccessful. Use it for debugging.
        """

        
        dim = int(2 ** self.qubits)
        res = np.zeros(dim * dim, dtype='complex128').reshape(dim, dim)
        
        diags = np.matmul(vec_basis(self.qubits), np.transpose(vec_basis(self.qubits), axes=(0, 2, 1)))
        
        diag_coeffs = np.zeros(dim, dtype='complex128')
        
        for i in range(len(self.qubit_params_list)):
            ie = coeffs[i] * rho(self.qubit_params_list[i][0], self.qubit_params_list[i][1])
            
            res += ie
            
            diag_coeffs += -1*coeffs[i]*np.diag(rho(self.qubit_params_list[i][0], 
                                                self.qubit_params_list[i][1]))
            
        for j in range(len(diags)):
            res += diag_coeffs[j]*diags[j]
            
        if np.allclose(res, np.array(offdiag, dtype='complex128'), atol=1e-04) == False:
            print(res)
            raise Exception("offdiag not matched correctly -> flag raised")
        
        x = coeffs.copy()
        
        x = np.append(x, diag_coeffs)    
        #print(res) 
        return x



    def test_qt(self, coeffs, target, atol=1e-8, check=False):
        """
        Test function for the unfixed optimization routine. 
        Raises an exception if unsuccessful. Use it for debugging.
        """

        dim = int(2 ** self.qubits)
        res = np.zeros(dim * dim, dtype='complex128').reshape(dim, dim)
        
        for i in range(len(self.qubit_params_list)):
            ie = coeffs[i] * rho(self.qubit_params_list[i][0], self.qubit_params_list[i][1])
            res += ie
            
        if np.allclose(res, np.array(target, dtype='complex128'), atol) == False:
            print(res)
            raise Exception("offdiag not matched correctly -> flag raised")
            
        if check==True:
            print(res)

        return coeffs
   
    def cgen(self):
        """
        Package the optimization routine to compute coefficients for a whole
        input vector basis set. Computational basis is still fixed

        :raises NotImplementedError: if any sub-opt routine is unsuccessful.
        :return: all the coefficients for the basis vector configuration 
        :rtype: np.ndarray, float
        """
        cs = []
        input_basis_vectors = vec_basis(self.qubits)
        for n in input_basis_vectors:
            for m in input_basis_vectors:
                if False in (n == m):
                    try:
                        nm = np.matmul(n, np.conjugate(m).T)
                        
                        cos = self.coeffs(self.f, nm, random=False)
                        
                        new_cos = self.test_offdiag(cos, nm)
                        cs.append(new_cos)
                    except NotImplementedError as e:
                        base = f"exception ({e}) caught for param values "
                        bad_param_stringer = lambda x,y: f"(theta, phi)=({x,y})"
                        bad_pstring=""
                        for param in cos:
                            bad_pstring += bad_param_stringer(param[0], param[1])
                        raise NotImplementedError(
                             base + bad_pstring
                        )
        return cs    

    def cgen_qt(self):
        """
        Package the optimization routine to compute coefficients for a whole
        input vector basis set. Computational basis is unfixed.

        :raises NotImplementedError: if any sub-opt routine is unsuccessful.
        :return: all the coefficients for the basis vector configuration 
        :rtype: np.ndarray, float
        """
        cs = []
        input_basis = rho_basis(self.qubits)
        for basis in input_basis:
                try:
                    nm = basis
                    cos = self.coeffs_qt(self.f_qt, nm)
                    cs.append(cos)
                except NotImplementedError as e:
                    raise NotImplementedError(
                            "exception ({}) caught for param values (theta, phi1 = {}, {})".format(
                                        e, (np.array(self.qubit_params_list)[:, 0] / (np.pi)), 
                                        (np.array(self.qubit_params_list)[:, 1]/ (np.pi))))
        return cs   