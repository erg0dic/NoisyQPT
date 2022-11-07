"""
Rough script. Playing around with parallelizing on local machine while datagenjob.sh
gets munched by the cluster. 
"""

from joblib import Parallel, delayed
import multiprocessing
from qpt import Sqpt_protocol
import param_optimizer as po
from misc_utilities import *
from qpt import *
import pickle

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
                oic = protocol.oics(qplist=maxis, c=cs)

                c, cd = protocol.sqpt(oic)
                if index == 0:
                    c1n.append(dn(c, cd, channel, 1))
                if index == 1:
                    c2n.append(dn(c, cd, channel, 1))
                if index == 2:
                    c3n.append(dn(c, cd, channel, 1))

            
        with open("ax1_{}_ampdamp{}n_{}maxis4.pickle".format(axis_1_index, int(noise*100), counter), "wb") as handle:
            pickle.dump(c1n, handle)
        with open("ax1_{}_dephase{}n_{}maxis4.pickle".format(axis_1_index, int(noise*100), counter), "wb") as handle:
            pickle.dump(c2n, handle)
            
        with open("ax1_{}_depol{}n_{}maxis4.pickle".format(axis_1_index, int(noise*100), counter), "wb") as handle:
            pickle.dump(c3n, handle)
    except NotImplementedError as e:
        print(e)