"""
These routines generalize for n-qubits but the computational complexity 
grows exponentially. I essentially came up with a way to do state tomography 
by sampling from multinomial distributions by exploiting the linearity of the Trace.
Further TODO: Try sparse tensor algebra for compute speed-up and storage
              Try more realistic noise mixing models. See report.
"""

from tensor_algebra import *

def state_tomography(state, qubits, 
                     cheat=False, noise_level=0, 
                     noisy_axis=(False,False,False), 
                     noise_model='simple', measurements=100):
    """
    Helper function. Non-noisy or Noisy tomography that returns the Krauss 
    basis coefficients for arbitrary number of qubits using the cartesian 
    product basis of single qubit projectors in the computational basis. 
    This is the exact infinite measurement limit of the actual tomography.

    :param state: quantum state in density matrix format
    :type state: np.ndarray, float
    :param qubits: number of qubits
    :type qubits: int
    :param cheat: turn noise off for debugging
    :type cheat: bool
    :param noise_level: percentage of noise between 0 and 1 for each axis
    :type noise_level: float or (float,float,float)
    :param noisy_axis: for the single qubit, set 3 independently noisy axes
    :type noisy_axis: (bool, bool, bool)
    :param noise_model: choices of linear or gaussian or binomial: ``simple``or 
                        ``gtou`` or ``binomial``
    :type noise_model: str, defaults to 'simple'
    :raises Exception: if state is not physical
    :raises TypeError: if an incorrect noise model is specified
    :return: eigenvalues of the physical observables 
             (e.g. X, Y, Z measurements for the single qubit corresponding 
                to spin or polarization measurements)
    :rtype: np.ndarray, float
    
    """
    
    qeigvals, qeigbasis = np.linalg.eig(ko(1))  # eigendecomposition
    qprojectors = np.matmul(qeigbasis[:,:,:, np.newaxis],
                            np.transpose(np.conjugate(qeigbasis[:,:,:, np.newaxis]), 
                            axes=(0,1,3,2)))
    
    #print(qeigvals)
    #print(qprojectors)
    if np.allclose(np.trace(state), 1) != True:
        raise Exception(
              "Make sure the state has unit trace to ensure conservation of probability"
              )
    
    
    qprojectors[0] = np.array([0.5*np.eye(2), 0.5*np.eye(2)])  # identity should not reveal any info
    
    
    if True in noisy_axis:
        noisy_qprojectors = qprojectors.copy()
        noisy_qeigvals = qeigvals.copy()
        
        # replace valid projectors with maximally mixed identity projectors for axis noise
        
        if np.shape(noise_level) == ():
            noise_level = np.array([noise_level]*3)
            
        for i in range(3):   
            if noisy_axis[i] == True:
                e = noise_level[i]
                if noise_model == 'gtou':
                    err = e - np.mean(np.random.normal(0,1, measurements)
                                        + e)*(1-np.exp(-1*(1-e) * measurements**1/100))
                
                elif noise_model == 'binomial':
                    err_pvec = np.random.binomial(measurements, [1-e, e])
                    err_pvec = measurements*e
                
                elif noise_model == 'simple':
                    err = e
                else:
                    raise TypeError("Please specify an allowed noise model.")
                    
                    
                noisy_qprojectors[i+1] = ((1-np.abs(err))*noisy_qprojectors[i+1] + 
                                        np.abs(err)*np.array([0.5*np.eye(2), 0.5*np.eye(2)]))
                # noisy_qeigvals[i+1] = qeigvals[i]    because noise is only in outcome, not in 
                #                                    knowledge of outcome? unnecessary complication for now
    
    if qubits == 1:
        states = sensembler(len(qprojectors)*int(2**qubits), state).reshape(qprojectors.shape)
        
        bin_vecs = np.trace(np.matmul(qprojectors, states), axis1=2, axis2=3)
#        print(bin_vecs)
        op_addition = np.multiply(bin_vecs, qeigvals) # define addition algebra for the operators
       
        # information on op_addition and bin_vecs then completes the tomography procedure, but for testing:
        if cheat == True and True not in noisy_axis: # return the reconstructed state for comparison
            print("!")
            return 0.5*np.sum(np.multiply(np.sum(op_addition, axis=1)[:, np.newaxis, np.newaxis], 
                    ko(1)), axis=0)

        elif True in noisy_axis:  # repeat above to get another noisy op_addition
            
            bin_vecs_noise = np.trace(np.matmul(noisy_qprojectors, states), axis1=2, axis2=3)
#            print(bin_vecs_noise)
            op_addition_noise = np.multiply(bin_vecs_noise, noisy_qeigvals) 
            
            if cheat == True: 
                print("!")
                return np.sum(np.multiply(np.sum(op_addition_noise, axis=1)[:, np.newaxis, np.newaxis], 
                                      ko(qubits)), axis=0) / (2 ** (qubits))
            
            return op_addition , op_addition_noise 
        
        return op_addition / (2 ** (qubits-1))   # normalized

    if qubits > 1:  # define projectors in terms of single qubit projectors above
        projectors = np.array([tp(i) for i in itertools.product(qprojectors, repeat=qubits)]) 
        states = sensembler(len(projectors)*int(2**qubits), state).reshape(projectors.shape)
        
        bin_vecs = np.trace(np.matmul(projectors, states), axis1=2, axis2=3)
        eigvals = np.array([tp(i) for i in itertools.product(qeigvals, repeat=qubits)])
        op_addition = np.multiply(bin_vecs, eigvals) # define addition algebra for the operators
        
        
        if (cheat == True) and (True not in noisy_axis):
            # return the reconstructed state for comparison
            print("!")
            return np.sum(np.multiply(np.sum(op_addition, axis=1)[:, np.newaxis, np.newaxis], 
                                      ko(qubits)), axis=0) / (2 ** (qubits))
        
        elif True in noisy_axis:  # repeat above to get another noisy op_addition
            
            noisy_projectors = np.array([tp(i) for i in itertools.product(noisy_qprojectors, repeat=qubits)])
            
            bin_vecs_noise = np.trace(np.matmul(noisy_projectors, states), axis1=2, axis2=3)
            noisy_eigvals = np.array([tp(i) for i in itertools.product(noisy_qeigvals, repeat=qubits)])
            op_addition_noise = np.multiply(bin_vecs_noise, noisy_eigvals) 
            norm_noise = np.sum(np.abs(op_addition_noise), axis=1)
            
            if cheat == True:
            # return the reconstructed state for comparison
                print("!")
                return np.sum(np.multiply(np.sum(op_addition_noise, 
                      axis=1)[:, np.newaxis, np.newaxis], ko(qubits)), 
                                                axis=0) / (2 ** (qubits))
            
            return op_addition, op_addition_noise
        
        
        return op_addition # normalized
    
    
    
def noisy_state_tomography(state, qubits, measurements, 
                           noise_level=0.0, noise_model='simple', 
                           noisy_axis=(False, False, False), 
                           pseudopure=False):
    
    """
    Implementing quantum noise by sampling from a multinomial distribution 
    of physical observables mixed with a prior e.g. a classical noise model. 
    This uses the "helper" function above. Embarrasingly, this function is quite
    long but the length is mainly due to brute-force-edly enumerating nearly all 
    specificities associated with defining the problem e.g. different noise 
    levels, different number of axis measurements that would have taken the same
    number of lines otherwise. I did not have time to refactor in an OOP version
    as the math/physics was occupying most of my time. More importantly, this was 
    originally written with speed of execution for data collection in mind instead 
    of code brevity. 
    
    :param state: quantum state in density matrix format
    :type state: np.ndarray, float
    :param qubits: number of qubits
    :type qubits: int
    :param measurements: measurements for each axis
    :type measurements: int or (int, int, int)
    :param noise_level: percentage of noise between 0 and 1 for each axis
    :type noise_level: float or (float,float,float)
    :param noisy_axis: for the single qubit, set 3 independently noisy axes
    :type noisy_axis: (bool, bool, bool)
    :param noise_model: choices of linear or gaussian or binomial: ``simple``or 
                        ``gtou`` or ``binomial``
    :type noise_model: str, defaults to 'simple'
    :param pseudopure: if unnormalized states should be propagated, for testing
    :type pseudopure: bool
    :raises Exception: if integer noisy measurements are not possible
    :raises Exception: if double axis noise is specified for qubits > 1 
                       (no correlational noise implementation yet)
    :raises Exception: if vector of measurements does not have shape (3,)
    :return: measurement course-grained noisy eigenvalues of the physical observables 
             (e.g. X, Y, Z measurements for the single qubit corresponding 
                to spin or polarization measurements)
    :rtype: np.ndarray, float
    
    """

    
    scalar = False
    if np.shape(measurements) == () and np.shape(noise_level) == (): # both scalars
        
        noisy_measurements = noise_level * measurements
        
        if np.int(noisy_measurements) != noisy_measurements:
            raise Exception(
                  "select a noise_level or number of measurements st. n*M is a whole number"
                  )
        
        noisy_measurements = np.int(noisy_measurements)
        notnoisy_measurements = measurements - noisy_measurements
        
        scalar = True
            
    
    if True in noisy_axis:
        if pseudopure == True:
            m  = [measurements]*4
            noisy_axes = [i+1 for i in range(len(noisy_axis)) if noisy_axis[i]==True]
            for i in noisy_axes:
                m[i] = int((1-noise_level)*measurements)
            #print(m)
            pro_expecs = state_tomography(state, qubits) # true
            bin_vecs = np.abs(pro_expecs)

            norm = np.sum(np.abs(pro_expecs), axis=1)[:, np.newaxis]          
            
            
            outcome_vector = np.array([np.random.multinomial(m[i], bin_vecs[i] / norm[i])*norm[i] 
                              for i in range(len(bin_vecs))]) / np.array(m)[:, np.newaxis]
            
#            print(outcome_vector)
            
            signs = pro_expecs / np.where( bin_vecs != 0, bin_vecs, 1)  # addition_algebra
            
#            print(pro_expecs)
#            print(bin_vecs)
            #print(np.multiply(outcome_vector, signs))
            return np.multiply(outcome_vector, signs)
        
        
        pro_expecs, noisy_pro_expecs = state_tomography(state, qubits, 
                                 noise_level=noise_level, noisy_axis=noisy_axis, 
                                 noise_model=noise_model )
        
        norm, norm_noisy = (np.sum(np.abs(pro_expecs), axis=1)[: np.newaxis], 
                            np.sum(np.abs(noisy_pro_expecs), axis=1)[: np.newaxis]
                            )
        bin_vecs, bin_vecs_noise = np.abs(pro_expecs), np.abs(noisy_pro_expecs)
#        print(bin_vecs_noise)
#        print(bin_vecs, bin_vecs_noise)          
#        print(outcome_vector_notnoisy)
        if np.shape(measurements) == ():
            outcome_vector_noisy = (
                np.array([np.random.multinomial(measurements, 
                        bin_vecs_noise[i] / norm_noisy[i]) * norm_noisy[i] 
                        for i in range(len(bin_vecs_noise))]) / np.where(measurements != 0, 
                         measurements, 1)
            )
        else:
            outcome_vector_noisy = (
                    np.array([np.random.multinomial(measurements[i], 
                    bin_vecs_noise[i] / norm_noisy[i]) * norm_noisy[i] 
                    for i in range(len(bin_vecs_noise))]) / np.where(np.array(measurements) != 0, 
                                                    np.array(measurements), 1)[:, np.newaxis]
            )

        if np.shape(noise_level) == ():
            noise_level = np.array([noise_level]*3)
            
        ms = np.zeros(len(bin_vecs_noise))
        indices = cartesian_product([0,1,2,3], repeat=qubits)
        
        noisy_axes = [i+1 for i in range(len(noisy_axis)) if noisy_axis[i]==True]
        for j in noisy_axes:    # find the corrupted axes
            nums = []
            new_indices = [i for i in indices if j in i]  # for x,y,z i.e. 1, 2, 3
            for i in range(len(new_indices)):
                index = 0
                c=0
                while c < qubits:
                    index += 4**(c) * new_indices[i][qubits-c-1]
                    c += 1
                nums.append(index)
                
            ms[nums] = np.ones(len(nums))*noise_level[j-1]  # index of noise
                
                
        outcome_vector_noisy = (outcome_vector_noisy - 0.5*ms[:, np.newaxis]) / (1-ms[:, np.newaxis])
    
    
        signs_noisy = noisy_pro_expecs / np.where(bin_vecs_noise != 0, bin_vecs_noise, 1)
        fov = np.multiply(outcome_vector_noisy, signs_noisy)
    
        return fov
              
    else:   # in case there is no noise

        if scalar == True:   # scalar input case
            pro_expecs = state_tomography(state, qubits)
            bin_vecs = np.abs(pro_expecs)

            norm = np.sum(np.abs(pro_expecs), axis=1)[:, np.newaxis]          
            #copies = ensembler([measurements for _ in range(len(bin_vecs))], bin_vecs)
            
            
            outcome_vector = np.array([np.random.multinomial(measurements, 
                             bin_vecs[i] / norm[i])*norm[i] 
                             for i in range(len(bin_vecs))]) / measurements
            
#            print(outcome_vector)
            
            signs = pro_expecs / np.where( bin_vecs != 0, bin_vecs, 1)  # addition_algebra
            
#            print(pro_expecs)
#            print(bin_vecs)
            #print(np.multiply(outcome_vector, signs))
            return np.multiply(outcome_vector, signs)
        
        else:   # vector input case
            if qubits == 1:
                ms = measurements 

                pro_expecs = state_tomography(state, qubits)
                bin_vecs = np.abs(pro_expecs)

            
                outcome_vector = np.array([np.random.multinomial(ms[i], bin_vecs[i]) 
                for i in range(len(bin_vecs))]) / np.where(ms != 0, ms, 1)[:, np.newaxis]

#               print(outcome_vector)
    
                signs = pro_expecs / np.where( bin_vecs != 0, bin_vecs, 1)  # addition_algebra
                
                return np.multiply(outcome_vector, signs)  
            else:  # if qubits > 1
                 # assuming all joint measurements are also specified
                 
                 ms = measurements   # specify the identiy measurement as a convention
                 pro_expecs = state_tomography(state, qubits)
                 bin_vecs = np.abs(pro_expecs)
                 
                 norm = np.sum(np.abs(pro_expecs), axis=1)[:, np.newaxis]  

                 outcome_vector = np.array([np.random.multinomial(ms[i], bin_vecs[i] / norm[i])*norm[i] 
                                                for i in range(len(bin_vecs))]) / np.where(ms != 0, ms, 1)[:, np.newaxis]

#                 print(outcome_vector)
    
                 signs = pro_expecs / np.where( bin_vecs != 0, bin_vecs, 1)  # addition_algebra
                
                 return np.multiply(outcome_vector, signs) 
                 
                
                
                
            
            
def noisy_rho(state, qubits, measurements, 
              noise_level=0.0, noisy_axis=(False, False, False), 
              noise_model='simple'):
    """
    Constructing a density matrix rho of tomography results using a trick based 
    on the linearity of the trace (see report). Combining above. Returns a noisy rho 
    based on finite number of measurements and tunable axis noise.

    :param state: quantum state in density matrix format
    :type state: np.ndarray, float
    :param qubits: number of qubits
    :type qubits: int
    :param measurements: measurements for each axis
    :type measurements: int or (int, int, int)
    :param noise_level: percentage of noise between 0 and 1 for each axis
    :type noise_level: float or (float,float,float)
    :param noisy_axis: for the single qubit, set 3 independently noisy axes
    :type noisy_axis: (bool, bool, bool)
    :param noise_model: choices of linear or gaussian or binomial: ``simple``or 
                        ``gtou`` or ``binomial``
    :type noise_model: str, defaults to 'simple'

    :return: n-qubit reconstructed noisy density matrix
    :rtype: np.ndarray, float
        
    """
     
    op_addition = noisy_state_tomography(state, qubits, measurements, 
                                          noise_level, noise_model, noisy_axis,)
     
    return np.sum(np.multiply(np.sum(op_addition, axis=1)[:, np.newaxis, np.newaxis], 
                                      ko(qubits)), axis=0) / (2 ** (qubits))
     