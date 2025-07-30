#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from network import network
from diffusion_simulation import diffusion_simulation

import argparse

DIR = 'C:/Users/Maxgroup/Documents/Yunduan/np_diffusion/instances'

class IM():
    def __init__(self, sim, params):
        self.sim = sim
        self.K = int(params['K'])
        self.sim_id = params['sim_id']
    def optimize(self):
        n = self.sim.network_G.n
        cnt_adp = np.zeros(n)
        seed = np.zeros(n, dtype=bool)
        seed_order = []
        for i in range(self.K):
            print(i)
            max_adp = 0
            ind = 0
            for j in range(n):
                if ~seed[j]:
                    cur_seed = seed.copy()
                    cur_seed[j] = True
                    mu,_ = self.sim.run_simulation(seed=cur_seed)
                    cnt_adp[j] = np.sum(mu)
                    if cnt_adp[j]>max_adp:
                        max_adp = cnt_adp[j]
                        ind = j

            seed[ind] = True
            seed_order.append(ind)

        df = pd.DataFrame({'seed':seed_order})
        df.to_csv(f'{DIR}/{self.sim_id}/seed.csv',index=None)
        print(np.sum(mu))

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('sim_id', help='The id number of current simulation')
    parser.add_argument('alpha', help='Utility coefficient for external influence')
    parser.add_argument('beta', help='Utility coefficient for network effect')
    parser.add_argument('noise_dist', help='The distribution of noise')
    parser.add_argument('noise_dist_param', help='The parameter for noise distribution')
    parser.add_argument('cnt_iter', help='Iteration number of each simulation')
    parser.add_argument('cnt_simulation', help='Simulation times')
    parser.add_argument('K', help='The number of seeds')

    parser.add_argument('--ell', help='Coefficient for uniform noise')
    parser.add_argument('--is_network_given', action='store_true', help='Use a given network or not')
    parser.add_argument('--network_dir', help='Directory of network file')
    parser.add_argument('--network_size', help='The number of users in the network')
    parser.add_argument('--edge_prob', help='The probability of an edge existing in the network')

    parser.add_argument('--is_value_given', action='store_true', help='Use given intrinsic values or not')
    parser.add_argument('--value_dir', help='Directory of value file')
    parser.add_argument('--v_dist', help='The rule for intrinsic value distribution')
    parser.add_argument('--v_dist_param', help='The parameter for intrinsic value distribution')


    args_dict = vars(parser.parse_args())
    network_G = network(args_dict)
    sim = diffusion_simulation(network_G,args_dict)

    instance = IM(sim,args_dict)
    instance.optimize()
