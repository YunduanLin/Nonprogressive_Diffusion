#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.stats import rv_discrete
import networkx as nx

# Define the power-law distribution
class power_law(rv_discrete):
    # power_law distribution
    def _pmf(self, x, x_min, x_max, expon):
        return ((x/x_min)**(1-expon)-((x+1)/x_min)**(1-expon)) / (1-(x_max/x_min)**(1-expon))

class network():

    def __init__(self, params):
            
        if params['is_network_given']:
            self.network_dir = params['network_dir']
            self.read_network_from_csv()
        else:
            self.n = int(params['network_size'])
            if params['network_type']=='ER':
                self.p = float(params['ER_prob'])
                self.generate_ER_network()
            elif params['network_type']=='PL':
                self.expon = float(params['PL_exponent'])
                self.corr = float(params['PL_corr'])
                self.x_min = float(params['PL_xmin'])
                self.generate_PL_network()
            else:
                raise Exception("The network type has not been implemented")

            l_edge = np.where(self.mat_adj == 1)
            self.df_edge = pd.DataFrame({'from_node': l_edge[0], 'to_node': l_edge[1]})

        print(f'Instance generated with {self.n} nodes.')

        # network characteristics
        self.in_degree = np.sum(self.mat_adj, axis=0)  # indegree column
        self.out_degree = np.sum(self.mat_adj, axis=1)  # out-neighbor
        self.in_degree_adj = self.in_degree.copy()  # adjusted in-neighbor, to exclude the 0 in-degree nodes
        self.in_degree_adj[self.in_degree == 0] = 1

        # self.n_min = np.min(self.n_in)
        self.in_neighbors = [np.where(i > 0)[0] for i in self.mat_adj.T]
        self.out_neighbors = [np.where(i > 0)[0] for i in self.mat_adj]

        self.y = np.zeros(self.n)  # adoption status

        # define notation for centrality
        D_inv = np.zeros((self.n,self.n))
        in_d_inv = 1/self.in_degree_adj
        in_d_inv[self.in_degree==0] = 0
        np.fill_diagonal(D_inv, in_d_inv)
        self.A = D_inv @ self.mat_adj.T
        self.b = D_inv @ np.ones(self.n)

        # intrinsic value
        if params['is_value_given']:
            self.value_dir = params['value_dir']
            self.read_value_from_csv()
        else:
            self.v_dist = params['v_dist']
            self.v_dist_param = float(params['v_dist_param'])
            self.generate_random_value()

    def read_network_from_csv(self):
        # columns: ['from_node', 'to_node']
        self.df_edge = pd.read_csv(self.network_dir)
        l_edge = self.df_edge.values

        self.n = np.max(l_edge) + 1
        self.l_node = np.arange(0,self.n,1)

        mat_adj = np.zeros((self.n, self.n))
        mat_adj[tuple(l_edge.T)] = 1
        np.fill_diagonal(mat_adj, 0)
        self.mat_adj = mat_adj.astype('float32')

    # generate ER random network according to some rules
    def generate_ER_network(self):

        self.l_node = np.arange(0,self.n,1)
        mat_adj = (np.random.rand(self.n, self.n) < self.p).astype(int)
        np.fill_diagonal(mat_adj, 0)
        self.mat_adj = mat_adj

    # generate PL random network according to some rules
    def generate_PL_network(self):
        self.l_node = np.arange(0,self.n,1)
        pl = power_law(a=self.x_min, b=self.n, name='power-law')
        in_degree = np.sort(pl.rvs(x_min=self.x_min, x_max=self.n, expon=self.expon, size=self.n))[::-1]
        Z = np.random.choice([0,1], size=self.n, p=[1-abs(self.corr), abs(self.corr)])
        I_0, I_1 = np.where(Z==0)[0], np.where(Z==1)[0]

        if self.corr>=0:
            out_degree = in_degree.copy()
            out_degree[I_0] = np.random.permutation(out_degree[I_0])
        else:
            out_degree = in_degree.copy()[::-1]
            out_degree[I_1] = np.random.permutation(out_degree[I_1])

        nx_D = nx.directed_configuration_model(in_degree, out_degree)
        nx_D = nx.DiGraph(nx_D)
        nx_D.remove_edges_from(nx.selfloop_edges(nx_D))

        self.mat_adj = nx.to_numpy_array(nx_D)

    def generate_random_value(self):
        if self.v_dist=='normal':
            self.v = np.random.normal(0, self.v_dist_param, self.n)
        elif self.v_dist=='uniform':
            self.v = np.random.uniform(-self.v_dist_param, self.v_dist_param, self.n)
        elif self.v_dist=='uniform_neg':
            self.v = np.random.uniform(-self.v_dist_param, 0, self.n)
        elif self.v_dist=='uniform_pos':
            self.v = np.random.uniform(0, self.v_dist_param, self.n)
        self.df_v = pd.DataFrame({'v':self.v})

    def read_value_from_csv(self):
        # columns: ['v']
        self.df_v = pd.read_csv(self.value_dir)
        self.v = self.df_v['v'].values.astype('float32')
        self.v = self.v[:self.n]

    def reset(self):
        self.y = np.zeros(self.n)

    def cal_mean_inv_indeg(self):
        inv_indeg = 1/self.in_degree_adj
        inv_indeg = inv_indeg[self.in_degree>0]
        self.ave_inv_indeg = np.sum(inv_indeg)/self.n
