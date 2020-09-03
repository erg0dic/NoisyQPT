"""
Rough script. Playing around with parallelizing on local machine while datagenjob.sh
gets munched by the cluster. 
"""

from joblib import Parallel, delayed
import multiprocessing
from qpt import Sqpt_protocol
import param_optimizer as po

def ptest_i(maxis, counter, noise, measurements, axis_1_index):
    """
    use this to parallelize the for loop over basis vectors during datagen
    """

    try:
        cs = po.QPTparaopt(qubit_number=1, qparalist=maxis).cgen()
        c1n, c2n, c3n = [], [], []
        for index, channel in enumerate([damp, dephase, dep(0.2)]):
            if noise == 0:
                booll = False
            else:
                booll = True
            
            for i in measurements:
                protocol = Sqpt_protocol(channel, qubits=1, 
                                    measurements=i, noise_level=noise, 
                                    noisy_axis=(booll, True, True))
                oic = protocol.oics(qplist=maxis, cs)

    #                if oic == []:## in case it fails
    #                    break
                c, cd = protocol.sqpt(oic)
                if index == 0:
                    c1n.append(dn(c, cd, channel, 1))
                if index == 1:
                    c2n.append(dn(c, cd, channel, 1))
                if index == 2:
                    c3n.append(dn(c, cd, channel, 1))
    #            if ((len(c1n) != len(measurements)) or (len(c2n) != len(measurements)) or (len(c1n) != len(measurements))):
    #                if noise==0:
    #                    failed_indices.append(counter)
            
        with open("ax1_{}_ampdamp{}n_{}maxis4.pickle".format(axis_1_index, int(noise*100), counter), "wb") as handle:
            pickle.dump(c1n, handle)
        with open("ax1_{}_dephase{}n_{}maxis4.pickle".format(axis_1_index, int(noise*100), counter), "wb") as handle:
            pickle.dump(c2n, handle)
            
        with open("ax1_{}_depol{}n_{}maxis4.pickle".format(axis_1_index, int(noise*100), counter), "wb") as handle:
            pickle.dump(c3n, handle)
    except NotImplementedError as e:
        print(e)

qplist = [[[theta, phi], [theta2, phi2]] for theta in np.linspace(0, np.pi*0.5, 10) for phi in np.linspace(0, np.pi, 10)]
dict_qplist = {i:j for i,j in enumerate(qplist)}
failed_indices = []
measurements = np.arange(10,10000, 20)

def ptest(measurements, noise, qplist, axis_1_index):
num_cores = multiprocessing.cpu_count()
inputs = [[qplist.get(maxis), maxis, noise, measurements, axis_1_index] for maxis in qplist ]

Parallel(n_jobs=num_cores)(delayed(ptest_i)(maxis=i[0], counter=i[1], noise=i[2], measurements=i[3], axis_1_index=i[4]) for i in inputs)


#num_cores = multiprocessing.cpu_count()
#inputs = [[measurements, noise, qplist] for noise in [0, 0.1, 0.2]]
#print(Parallel(n_jobs=num_cores)(delayed(ptest)(measurements=i[0], noise=i[1], qplist=i[2]) for i in inputs))
#measurements = np.arange(10,10000, 20)
#for noise in [0.3]:
#    counter = 0
#    for maxis in qplist:
#        try:
#            cs = cgen(qubits=1, qplist=maxis)
#            c1n = []
#            for index, channel in enumerate([x_rotation]):
#                if noise == 0:
#                    booll = False
#                else:
#                    booll = True
#                
#                for i in measurements:
#                    oic = oics(channel, qplist = maxis, c = cs , qubits=1, measurements=i, noise_level=noise, noisy_axis=(booll, False, False))
##                if oic == []:## in case it fails
##                    break
#                    c, cd = sqpt(oic, 1)
#                    if index == 0:
#                        c1n.append(dn(c, cd, channel, 1))
#
#
##            if ((len(c1n) != len(measurements)) or (len(c2n) != len(measurements)) or (len(c1n) != len(measurements))):
##                if noise==0:
##                    failed_indices.append(counter)
#                
#            with open("cbiasnoise/ampdamp{}n_{}maxis7.pickle".format(int(noise*100), counter), "wb") as handle:
#                pickle.dump(c1n, handle)
#
#        except NotImplementedError as e:
#            print(e)
#            if noise==0:
#                failed_indices.append(counter)
#        counter += 1
#        print(counter)


#res=20
#qplist2 = [[theta, phi] for theta in np.linspace(0, np.pi*0.5, res) for phi in np.linspace(0, np.pi, res)]
#dict_qplist2 = {i:j for i,j in enumerate(qplist2)}
#
#def qp(k, res=res):
#    l =  [[dict_qplist2.get(k), [theta, phi]] for theta in np.linspace(0, np.pi*0.5, res) for phi in np.linspace(0, np.pi, res)]
#    return {i:j for i,j in enumerate(l)}
#noise=0.2
#ins = [[measurements, noise, qp(k), k] for k in dict_qplist2]
#
#from datetime import datetime
#from tqdm import tqdm
#
#Parallel(n_jobs = multiprocessing.cpu_count())(delayed(ptest)(measurements=ins[i][0], 
#         noise=ins[i][1], qplist=ins[i][2], axis_1_index=ins[i][3]) for i in tqdm(range(50)))




#### prelim data generation for noise model
    
    
#measurements = np.arange(10,10000, 20)
#for noise in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#    if noise == 0:
#        booll = False
#    else:
#        booll = True
#    c1n = []
#    for index, channel in enumerate([x_rotation]):
#        for i in measurements:
#            c, cd = sqpt(ics(channel, vec_basis(1), 1, i, noise_level=noise, noisy_axis=(booll, False, False)), 1)
#            if index == 0:
#                c1n.append(dn(c, cd, channel, 1))
#
#    with open("rotationx90_fixede{}n.pickle".format(int(noise*100)), "wb") as handle:
#        pickle.dump(c1n, handle)
#    
#    print("Done noise = {}".format(noise))
    