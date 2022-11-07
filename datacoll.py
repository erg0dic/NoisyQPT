# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 12:17:47 2020

@author: Irtaza
"""

from qpt import *
import os
import pickle
import param_optimizer as po
import qpt as q
import numpy as np
import matplotlib.pyplot as plt
from misc_utilities import *

orts = [[np.pi*0.5, np.pi], [0, 0], [np.pi*0.5, 0], [np.pi*0.5, np.pi*0.5]]

## TODO: modify cgen_qt and calling oics as both are now part of classes (done, now test it)

class Config(object):
    
    def __init__(self, channel_name, channel, noise_level, 
                noisy_axis, input_state_config, config_name, 
                measurements, pathroot, qubits=1, iterations=1):
        """_summary_

        Args:
            
            iterations (int, optional): Number of experimental repetitions. Defaults to 1.
        """
        self.channel = channel
        self.noise_level = noise_level
        self.noisy_axis = noisy_axis
        self.input_state_config = input_state_config
        self.measurements = np.array(measurements)
        self.qubits = qubits
        self.channel_name = channel_name
        self.config_name = config_name
        self.iterations = iterations
        
        if not os.path.exists(pathroot): # memoize the data collection
            os.makedirs(pathroot)
        
        self.pathroot = pathroot
    
    def set_iterations(self, iteration):
        self.iterations = iteration
        
    def record(self):
        dns = []
        config = self.input_state_config
        channel= self.channel
        qubits = self.qubits
        
        break_condition = False
        while not break_condition:
            try:
                cs = po.QPTparaopt(self.qubits, qparalist=config, no_of_dec_terms=4).cgen_qt()
                # cs = cgen_qt(1, config) refactoring remnant: remove after debugging
                break_condition = True
                
            except NotImplementedError as e:
                print(e)
            
        for iteration in range(1,self.iterations+1,1):
            dns = []
            path = self.config_name + "_" + self.channel_name + str(int(100*self.noise_level)) + "n" + "_it_" + str(iteration)
            #print(iteration)
            if self.noisy_axis[0] == True:
                    path += 'x'
                
            if self.noisy_axis[1] == True:
                    path += 'y'
                
            if self.noisy_axis[2] == True:
                    path += 'z'
#            print(iteration)        
            if not os.path.exists("{}/{}".format(self.pathroot, path)):
                
                for measurement in self.measurements:
                    protocol = q.Sqpt_protocol(channel, qubits, 
                                    measurement, self.noise_level, self.noisy_axis)
                    rhos = protocol.oics_qt(config, cs)
                    # rhos = oics_qt(channel, config, cs, qubits, measurement, 
                    #        noise_level=self.noise_level, noisy_axis=self.noisy_axis) # refactored
                    c, cd = protocol.sqpt(rhos)
                    dns.append(dn(c, cd, channel, 1))
    
                with open("{}/{}".format(self.pathroot, path), "wb") as handle:
                    pickle.dump(dns, handle)
                    
    def change_noise(self, new_level, axis):
        self.noise_level = new_level
        self.noisy_axis = axis
        
    def change_channel(self, new_channel, new_channel_name):
        self.channel_name = new_channel_name
        self.channel = new_channel
        
    def change_config(self, new_config, new_config_name):
        self.config_name = new_config_name
        self.input_state_config = new_config
     
    def path_projector(self):
        path = self.config_name + "_" + self.channel_name + str(int(100*self.noise_level)) + "n" + "_it_" + "{}"
        if self.noisy_axis[0] == True:
            path += 'x'
                
        if self.noisy_axis[1] == True:
            path += 'y'
                
        if self.noisy_axis[2] == True:
            path += 'z'
            
        return path
    
    def averaging(self):
        ca = np.array(np.zeros_like(ma(self.measurements,1)))
        ca2 = np.array(np.zeros_like(ma(self.measurements,1)))    
        for iteration in range(1,self.iterations+1,1):
            path = self.path_projector().format(iteration)
            with open("{}/{}".format(self.pathroot,path), "rb") as handle:
                l = np.array(pickle.load(handle))
            ca += l
            ca2 += l*l
                
        ca /= self.iterations
        ca2 /= self.iterations
        #print(ca, ca2)
        err = np.sqrt(ca2 - ca*ca)
        
        return ca, err
    
    def plot(self):
        plt.figure()
        err, y = self.averaging()
        plt.plot(self.measurements, y)
        
    
    
class PlotterofData(Config):
    def __init__(self, channels, configs):
        self.channels = channels
        self.configs = configs
        self.pathroot = ""

    def already_recorded(self, path):
       "Check if the file has already been recoreded"
       return os.path.exists(self.pathroot+path)
     


channels = {"ampdamp":damp, "depol":dep(0.2), "dephase":dephase, 
            "rotx90":x_rotation, "roty90":y_rotation, "rotz90":z_rotation}
zzxy = [[np.pi, 0], [0, 0], [np.pi*0.5, 0], [np.pi*0.5, np.pi*0.5]]
xxyz = [[np.pi*0.5, np.pi], [0, 0], [np.pi*0.5, 0], [np.pi*0.5, np.pi*0.5]]

yyxz = [[np.pi*0.5, np.pi*1.5], [0, 0], [np.pi*0.5, 0], [np.pi*0.5, np.pi*0.5]]

zzxyrot18x = [[np.pi-0.1*np.pi, 0], [0+0.1*np.pi, np.pi], [np.pi*0.5-np.pi*0.1, 0], [np.pi*0.5, np.pi*0.5]]
nonorth = [[np.pi-np.pi*0.15, 0], [0+0.15*np.pi, np.pi], [np.pi*0.5, 0], [np.pi*0.5, np.pi*0.5]]

bipyramedal = [[0,0], [np.pi, 0], [np.pi*0.5, np.pi], [np.pi*0.5, np.pi*0.5], [np.pi*0.5, 0]]

hexagonal = [[0,0], [np.pi, 0], [np.pi*0.5, np.pi], [np.pi*0.5, np.pi*0.5], [np.pi*0.5, 0], [np.pi*0.5, np.pi*1.5]]

tets = [[np.arcsin(np.sqrt(2/3)), np.pi*0.25], [np.arcsin(np.sqrt(2/3)), np.pi+np.pi*0.25], 
         [np.pi-np.arcsin(np.sqrt(2/3)), np.pi-np.pi*0.25], 
         [np.pi-np.arcsin(np.sqrt(2/3)), 2*np.pi-np.pi*0.25] ]


configs = {"tets":tets, "zzxy2":zzxy, "xxyz": xxyz, "yyxz": yyxz, 
            "zzxyrot18x":zzxyrot18x, "nonorth":nonorth,
           "bipyramedal2":bipyramedal, "hexagonal":hexagonal}

noise=0
for config_name in configs:
    for channel_name in channels:
        m = int(40000 / len(configs[config_name]))
        x = Config(channel_name, channels[channel_name], noise, 
                    (False, False, False),configs[config_name], 
                    config_name, np.arange(10,m,100), "cob")        
        x.set_iterations(100)
        x.record()
    
noise=0.3    
for config_name in configs:
    for channel_name in channels:
        m = int(40000 / len(configs[config_name]))
        x = Config(channel_name, channels[channel_name], noise, 
            (False, False, True),configs[config_name], config_name, 
            np.arange(10,m,100), "cob")        
        x.set_iterations(100)
        x.record()
    
            
        

        