def rotation(theta, axis='x'):
    """anticlockwise SO(3) rotations. Note that the sign is flipped in case the rotation is around the
        y-axis rotates clockwise when you consider the preservation of the left-handedness of xyz coordinates
    """
    
    if axis == 'z':
        return np.array([[np.cos(theta), -np.sin(theta), 0 ],[np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]]).reshape(3, 3)
        
    if axis == 'y':
        return np.array([[np.cos(theta), 0, np.sin(theta) ],[0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]]).reshape(3, 3)
        
    
    if axis == 'x':
        return np.array([[1, 0, 0], [ 0, np.cos(theta), -1*np.sin(theta) ],
                          [0, np.sin(theta), np.cos(theta)]]).reshape(3, 3)
    
    
def bloch_vec(theta, phi):
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]).reshape(3, 1)
    


def ma(array, window=5): 
    "just a simple convolution with const weights function i.e. 1"
    
    return np.convolve(np.array(array), np.ones((window,))/window, mode='valid')


def rot_angles(theta, phi):
    "take as arguments theta, phi -> alpha, beta that parametrize rotations along two orthogonal Y, Z axes in 3D"
    bvec = bloch_vec(theta, phi).flatten()
    arg_alpha = bvec[0] / (1-bvec[1]*bvec[1])
    arg_beta = bvec[1]
#    if abs(arg_alpha) > 1:# or abs(arg_beta) > 1:
#        return "Nan"
    alpha = np.arccos(arg_alpha) # parametrizes y rotation
    beta = np.arcsin(arg_beta)  # parametrizses z rotation
    
    return np.array([alpha, beta])



def hsip(mat1, mat2):
    "Hilbert-Schmidt inner product that generalizes the inner product in higher dimensions of CES"
    return np.trace(np.matmul(np.conjugate(mat1.T), mat2)) 
    
              
        
    
def depol(param, state):
     return np.matmul((1-param)*np.eye(len(state)), state) + 0.5*(param)*np.eye(len(state))   
        
def dep(param, qubits=1):
    return np.multiply(np.array([np.sqrt(1-0.75*param), 0.5*np.sqrt(param), 0.5*np.sqrt(param), 
                                 0.5*np.sqrt(param)])[:, np.newaxis, np.newaxis],np.array(ko(qubits)))     


def qubit(theta, phi):
    "use the standard definition of a qubit here for scalable parametrization. 0 < theta < pi and 0 < phi < 2*pi"
    
    if (theta > np.pi or theta < 0) or (phi < 0 or phi > np.pi*2):
        raise Exception("theta and phi must obey the boundary conditions specified by the param")
        
    return np.array([np.cos(theta / 2.), np.sin(theta / 2.)*np.exp(1j*phi)], dtype='complex128').reshape(2, 1)


def rho(theta, phi):
    
    return np.matmul(qubit(theta, phi), np.conjugate(qubit(theta, phi)).T)


def rotcon(state, theta, axis='y'):
    bvec = bloch_vec(state[0], state[1])
    bvecr = np.matmul(rotation(theta, axis), bvec)
    
    def nf(x, bvecr):
        return np.sum(np.abs(bloch_vec(x[0], x[1])-bvecr))
    
    opt = minimize(nf, np.random.random(2), 
                   args=(bvecr), method='SLSQP', 
                   options={'maxiter': 1000, 'ftol': 1e-15, 'eps': 1.4906e-12})
    return opt.x
        

### bloch sphere for some good poster illustrations about quantum channels 
    
%matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def blochify(theta_list= np.linspace(0, np.pi, 40), phi_list= np.linspace(0, 2*np.pi, 30) ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c = np.array([bloch_vec(theta, phi) for theta in theta_list for phi in phi_list])

    ax.scatter(c[:, 0], c[:, 1], c[:, 2], c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()