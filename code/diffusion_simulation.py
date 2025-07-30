#!/usr/bin/env python
# coding: utf-8

# DIR = 'C:/Users/Maxgroup/Documents/Yunduan/np_diffusion/instances'
DIR = '../instances'

import numpy as np
import numpy.linalg as la
import pandas as pd
import time

from network import network


THRESH = 1e-5


class diffusion_simulation():

    def __init__(self, G, params):
        self.G = G
        self.alpha, self.beta = float(params['alpha']), float(params['beta'])
        self.noise_dist = params['noise_dist']
        self.noise_dist_param = float(params['noise_dist_param'])
        if 'L' in params:
            self.L = float(params['L'])
        self.cnt_iter, self.t_warmup = int(params['cnt_iter']), int(params['t_warmup'])
        self.is_traj = params['is_traj']
        self.output_dir = DIR + '/' + params['sim_id']

        self.df_output = pd.DataFrame()
        self.df_output_traj = pd.DataFrame()
        self.t_sim, self.t_fp, self.t_exact = 0, 0, 0

    def start_diffusion(self, init=[], seed=[], p=0):
        # seed is for IM and p is for pricing

        n = self.G.n
        v, mu = self.G.v, self.G.y
        in_neighbors, in_degree = self.G.in_neighbors, self.G.in_degree_adj
        
        if len(init)>0:
            y = init
        else:
            y = np.zeros(self.G.n)
            
        y[seed] = 1  # initialization for IM problem

        agg_y, agg_prob = np.zeros(n), np.zeros(n)
        t_start = time.time()

        for t in range(self.t_warmup + self.cnt_iter):
            
            y_neighbor = np.array([y[i].sum() for i in in_neighbors])
            
            if self.noise_dist == 'logistic':
                epsilon = np.random.logistic(0, self.noise_dist_param, n)
            elif self.noise_dist == 'uniform':
                epsilon = np.random.uniform(-self.noise_dist_param, self.noise_dist_param, n)
            else:
                epsilon = 0

            u = v + self.beta * y_neighbor / in_degree - p + epsilon
            x = -(v + self.beta * y_neighbor / in_degree - p)
            prob = 1 - 1 / (1 + np.exp(-x / self.noise_dist_param))
            
            y = (u >= 0).astype(int)

            y[seed] = 1  # keep y invariant for seed users

            if t >= self.t_warmup:
                agg_y += y
                agg_prob += prob
                if (self.is_traj) & (t % 500 == 0):  # record the trajectory
                    t_int = int(t - self.t_warmup) + 1
                    self.df_output_traj[f'sim_{t_int}'] = agg_y / t_int
                    self.df_output_traj[f'p_{t_int}'] = agg_prob / t_int
                    self.df_output_traj[f't_{t_int}'] = time.time() - t_start

        self.t_sim = time.time() - t_start

        self.df_output['sim'] = agg_y / self.cnt_iter
        self.y_last = y

    def cal_exact(self, seed=[], p=0):
        n, v = self.G.n, self.G.v
        cnt_s = 2 ** n
        in_neighbors, in_degree = self.G.in_neighbors, self.G.in_degree_adj

        t_start = time.time()

        P = np.zeros((cnt_s, cnt_s))  # transition matrix
        for s1 in range(cnt_s):
            s1_bin = bin(s1)[2:]
            y_origin = np.array((n - len(s1_bin)) * [0] + list(s1_bin), dtype=int)
            y_neighbor = np.array([y_origin[i].sum() for i in in_neighbors])

            x = -(v + self.beta * y_neighbor / in_degree - p)

            if self.noise_dist == 'logistic':
                y_to1 = 1 - 1 / (1 + np.exp(-x / self.noise_dist_param))
                y_to0 = 1 / (1 + np.exp(-x / self.noise_dist_param))
            else:
                raise Exception("The distribution type has not been implemented")

            for s2 in range(cnt_s):
                s2_bin = bin(s2)[2:]
                y_dest = np.array((n - len(s2_bin)) * [0] + list(s2_bin), dtype=int)

                prob = np.zeros(n)
                prob[y_dest == 0] = y_to0[y_dest == 0]
                prob[y_dest == 1] = y_to1[y_dest == 1]

                P[s1, s2] = np.prod(prob)

        self.P = P
        eig_vals, eig_vecs = la.eig(P.T)
        eig_vecs1 = eig_vecs[:, np.isclose(eig_vals, 1)]
        eig_vecs1 = eig_vecs1[:, 0]
        stationary = eig_vecs1 / eig_vecs1.sum()
        stationary = stationary.astype(float)

        q = np.zeros(n)
        for s in range(cnt_s):
            s_bin = bin(s)[2:]
            y = np.array((n - len(s_bin)) * [0] + list(s_bin), dtype=int)
            q += stationary[s] * y

        self.t_exact = time.time() - t_start

        self.q_star = q
        self.df_output['exact'] = self.q_star

    def run_fixed_point(self, seed=[], p=0):
        # seed is for IM and p is for pricing

        v, mu = self.G.v, self.G.y
        in_neighbors, in_degree = self.G.in_neighbors, self.G.in_degree_adj

        mu[seed] = 1  # initialization for IM problem

        t_start = time.time()
        for t in range(self.cnt_iter):
            mu_neighbor = np.array([mu[i].sum() for i in in_neighbors])
            x = -(v + self.beta * mu_neighbor / in_degree - p)
            mu_last = mu

            if self.noise_dist == 'logistic':
                mu = 1 - 1 / (1 + np.exp(-x / self.noise_dist_param))
            elif self.noise_dist == 'uniform':
                tmp = self.L * x + 0.5
                tmp[tmp > 1] = 1
                tmp[tmp < 0] = 0
                mu = 1 - tmp

            mu[seed] = 1  # keep mu invariant for seed users

            if np.linalg.norm(mu_last - mu, 1) < THRESH:
                break

        self.t_fp = time.time() - t_start

        self.mu_star = mu
        self.df_output['fp'] = self.mu_star

    def re_cal(self, seed=[], p=0):
        mu = self.mu_star
        
        n = self.G.n
        v = self.G.v
        in_neighbors, in_degree = self.G.in_neighbors, self.G.in_degree_adj

        agg_y, agg_prob = np.zeros(n), np.zeros(n)

        t_iter = 1000
        for t in range(t_iter):

            y = np.random.binomial(size=n, n=1, p=mu)

            y_neighbor = np.array([y[i].sum() for i in in_neighbors])
            
            if self.noise_dist == 'logistic':
                epsilon = np.random.logistic(0, self.noise_dist_param, n)
            elif self.noise_dist == 'uniform':
                epsilon = np.random.uniform(-self.noise_dist_param, self.noise_dist_param, n)
            else:
                epsilon = 0

            u = v + self.beta * y_neighbor / in_degree - p + epsilon
            x = -(v + self.beta * y_neighbor / in_degree - p)
            prob = 1 - 1 / (1 + np.exp(-x / self.noise_dist_param))
            
            y = (u >= 0).astype(int)

            y[seed] = 1  # keep y invariant for seed users

            
            agg_y += y
            agg_prob += prob

        self.df_output[f'sim_re'] = agg_y / t_iter
        self.df_output[f'p_re'] = agg_prob / t_iter
        self.df_output['in_deg'] = self.G.in_degree
        

    def output(self):
        self.df_output.to_csv(self.output_dir + '/results.csv', index=None)
        if self.is_traj:
            self.df_output_traj.to_csv(self.output_dir + '/results_traj.csv', index=None)
        self.G.df_edge.to_csv(self.output_dir + '/edge.csv', index=None)
        self.G.df_v.to_csv(self.output_dir + '/v.csv', index=None)
        pd.DataFrame({'sim': [self.t_sim], 'fp': [self.t_fp], 'exact': [self.t_exact]}).to_csv(
            self.output_dir + '/t.csv', index=None)


