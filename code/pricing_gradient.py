#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import time

THRESH = 1e-5

class grad():

    def __init__(self, sim, params):
        self.sim = sim
        self.lr_mu, self.lr_p = params['lr_mu'],params['lr_p']
        self.W = params['W']
        self.K = params['K']
        self.alpha = params['alpha']

        # self.a, self.b, self.x = params['a'], params['b'], params['x']


    def solve_mu_grad(self):
        lr = self.lr_mu
        n = self.sim.G.n
        v = self.sim.G.v
        in_neighbors, in_degree = self.sim.G.in_neighbors, self.sim.G.in_degree_adj
        out_neighbors = self.sim.G.out_neighbors
        
        beta = self.sim.beta
        mu = np.ones(n)*0.5

        last_mu = np.ones(n)-THRESH
        last_profit = 0

        while np.linalg.norm(mu-last_mu,np.inf)>1e-5:
            last_mu = mu
            y_neighbor_mu = np.array([mu[i].sum() for i in in_neighbors])
            y_neighbor_mu_inv = np.array([(mu[i]/in_degree[i]).sum() for i in out_neighbors])
            grad_mu = (v + beta*y_neighbor_mu/in_degree + np.log((1-mu)/mu) -mu/(1-mu)-1 + beta*y_neighbor_mu_inv)/self.alpha

            mu = mu+grad_mu*lr
            mu[mu<=0] = THRESH
            mu[mu>=1] = 1-THRESH

            p = (v + beta * y_neighbor_mu / in_degree + np.log((1 - mu) / mu))/self.alpha
            if np.sum(mu*p)<last_profit:
                lr = lr/2
            last_profit = np.sum(mu*p)
            # print(last_profit)

        self.mu = mu

    def solve_mu_grad_myopic(self, mu_0):
        lr = self.lr_mu
        n = self.sim.G.n
        v = self.sim.G.v
        in_neighbors, in_degree = self.sim.G.in_neighbors, self.sim.G.in_degree_adj
        out_neighbors = self.sim.G.out_neighbors
        
        beta = self.sim.beta
        mu = np.ones(n)*0.5

        last_mu = np.ones(n)-THRESH
        last_profit = 0

        while np.linalg.norm(mu-last_mu,np.inf)>1e-5:
            last_mu = mu
            y_neighbor_mu = np.array([mu_0[i].sum() for i in in_neighbors])
            
            grad_mu = (v + beta*y_neighbor_mu/in_degree + np.log((1-mu)/mu) -mu/(1-mu)-1)/self.alpha

            mu = mu+grad_mu*lr
            mu[mu<=0] = THRESH
            mu[mu>=1] = 1-THRESH

            p = (v + beta * y_neighbor_mu / in_degree + np.log((1 - mu) / mu))/self.alpha
            if np.sum(mu*p)<last_profit:
                lr = lr/2
            last_profit = np.sum(mu*p)
            # print(last_profit)

        self.mu = mu
    

    def solve_mu_grad_transient(self, mu_0, t):
        lr = self.lr_mu
        n = self.sim.G.n
        v = self.sim.G.v
        in_neighbors, in_degree = self.sim.G.in_neighbors, self.sim.G.in_degree_adj
        out_neighbors = self.sim.G.out_neighbors
        
        beta = self.sim.beta
        mu = np.ones((t+1, n))*0.5
        grad_mu, p = np.zeros((t+1,n)), np.zeros((t+1,n))
        mu[0,:] = mu_0

        last_mu = np.ones(n)-THRESH
        last_profit = 0

        while np.linalg.norm(mu-last_mu,np.inf)>1e-5*t:
            # print(np.linalg.norm(mu-last_mu,np.inf))
            last_mu = mu
            
            for ite in range(1,t+1):
                y_neighbor_mu = np.array([mu[ite-1,i].sum() for i in in_neighbors])
                grad_mu[ite, :] = (v + beta*y_neighbor_mu/in_degree + np.log((1-mu[ite,:])/mu[ite,:]) -mu[ite,:]/(1-mu[ite,:])-1)/self.alpha
                if ite<t:
                    y_neighbor_mu_inv = np.array([(mu[ite+1,i]/in_degree[i]).sum() for i in out_neighbors])
                    grad_mu[ite, :] = grad_mu[ite, :] + (beta*y_neighbor_mu_inv)/self.alpha
            
            mu = mu+grad_mu*lr
            mu[mu<=0] = THRESH
            mu[mu>=1] = 1-THRESH

            for ite in range(1,t+1):
                y_neighbor_mu = np.array([mu[ite-1,i].sum() for i in in_neighbors])
                p[ite, :] = (v + beta * y_neighbor_mu / in_degree + np.log((1 - mu[ite,:]) / mu[ite,:]))/self.alpha
            if np.sum(mu*p)<last_profit:
                lr = lr/2
            last_profit = np.sum(mu*p)
            # print(last_profit)

        self.mu = mu

    def solve_p_grad(self,start_pos,ub):
        lr = self.lr_p
        n = self.sim.G.n
        v = self.sim.G.v
        in_neighbors, in_degree = self.sim.G.in_neighbors, self.sim.G.in_degree_adj
        beta = self.sim.beta

        p = start_pos
        p_user = self.W @ p
        mu = self.sim.run_fixed_point(p_user)

        last_p = -np.ones(self.K)
        last_profit = 0
        while np.linalg.norm(p-last_p,np.inf)>1e-2:
            last_p = p
            y_neighbor_mu = np.array([mu[i].sum() for i in in_neighbors])
            exp = np.exp(v + beta * y_neighbor_mu / in_degree-self.alpha*p_user)
            part_h_p = - self.alpha * self.W.T * (exp / (1+exp)**2).T
            part_h_mu = self.sim.G.mat_adj * (exp / (1+exp)**2 * beta/in_degree).T

            inv = np.identity(self.sim.G.n) + part_h_mu + part_h_mu@part_h_mu
            grad_pi_p = part_h_p @ inv @ self.W @ p + self.W.T @ mu
            # print(grad_pi_p)
            p = p + grad_pi_p * lr
            p[p < 0] = 0
            p[p > ub] = ub[p>ub]
            p_user = self.W @ p
            mu = self.sim.run_fixed_point(p_user)

            if np.sum(mu*p_user)<last_profit:
                lr = lr/2
            last_profit = np.sum(mu*p_user)
            # print(last_profit)
        self.p = p

    
    def solve_p_grad_myopic(self,mu_0,start_pos,ub):
        lr = self.lr_p
        n = self.sim.G.n
        v = self.sim.G.v
        in_neighbors, in_degree = self.sim.G.in_neighbors, self.sim.G.in_degree_adj
        beta = self.sim.beta

        p = start_pos
        p_user = self.W @ p
        y_neighbor_mu = np.array([mu_0[i].sum() for i in in_neighbors])
        exp = np.exp(v + beta * y_neighbor_mu / in_degree-self.alpha*p_user)
        mu = 1-1/(1+exp)

        last_p = -np.ones(self.K)
        last_profit = 0
        while np.linalg.norm(p-last_p,np.inf)>1e-2:
            last_p = p
            y_neighbor_mu = np.array([mu_0[i].sum() for i in in_neighbors])
            exp = np.exp(v + beta * y_neighbor_mu / in_degree-self.alpha*p_user)
            part_h_p = - self.alpha * self.W.T * (exp / (1+exp)**2).T
            
            grad_pi_p = part_h_p @ self.W @ p + self.W.T @ mu
            # print(grad_pi_p)
            p = p + grad_pi_p * lr
            p[p < 0] = 0
            p[p > ub] = ub[p>ub]
            p_user = self.W @ p
            exp = np.exp(v + beta * y_neighbor_mu / in_degree-self.alpha*p_user)
            mu = 1-1/(1+exp)

            if np.sum(mu*p_user)<last_profit:
                lr = lr/2
            last_profit = np.sum(mu*p_user)
            # print(last_profit)
        self.p = p

    
    def solve_p_grad_transient(self,mu_0,start_pos,ub):
        lr = self.lr_p
        n = self.sim.G.n
        v = self.sim.G.v
        in_neighbors, in_degree = self.sim.G.in_neighbors, self.sim.G.in_degree_adj
        beta = self.sim.beta

        p = start_pos
        p_user = self.W @ p
        y_neighbor_mu = np.array([mu_0[i].sum() for i in in_neighbors])
        exp = np.exp(v + beta * y_neighbor_mu / in_degree-self.alpha*p_user)
        mu = 1-1/(1+exp)

        last_p = -np.ones(self.K)
        last_profit = 0
        while np.linalg.norm(p-last_p,np.inf)>1e-2:
            last_p = p
            y_neighbor_mu = np.array([mu_0[i].sum() for i in in_neighbors])
            exp = np.exp(v + beta * y_neighbor_mu / in_degree-self.alpha*p_user)
            part_h_p = - self.alpha * self.W.T * (exp / (1+exp)**2).T
            
            grad_pi_p = part_h_p @ self.W @ p + self.W.T @ mu
            # print(grad_pi_p)
            p = p + grad_pi_p * lr
            p[p < 0] = 0
            p[p > ub] = ub[p>ub]
            p_user = self.W @ p
            exp = np.exp(v + beta * y_neighbor_mu / in_degree-self.alpha*p_user)
            mu = 1-1/(1+exp)

            if np.sum(mu*p_user)<last_profit:
                lr = lr/2
            last_profit = np.sum(mu*p_user)
            # print(last_profit)
        self.p = p

    def solve_p_grad_nodiff(self,start_pos,ub):
        lr = self.lr_p
        n = self.sim.G.n
        v = self.sim.G.v
        in_neighbors, in_degree = self.sim.G.in_neighbors, self.sim.G.in_degree_adj

        p = start_pos
        p_user = self.W @ p
        mu = 1 - 1/(1+np.exp(v-self.alpha * p_user))

        last_p = -np.ones(self.K)
        last_profit = 0
        while np.linalg.norm(p-last_p,np.inf)>1e-2:
            last_p = p
            exp = np.exp(v - self.alpha*p_user)
            part_h_p = - self.alpha * self.W.T * (exp / (1+exp)**2).T

            grad_pi_p = part_h_p @ self.W @ p + self.W.T @ mu
            p = p + grad_pi_p * lr
            p[p <= 0] = 0
            p[p > ub] = ub[p>ub]
            p_user = self.W @ p
            mu = 1 - 1/(1+np.exp(v-self.alpha * p_user))

            if np.sum(mu*p_user)<last_profit:
                lr = lr/2
            last_profit = np.sum(mu*p_user)
            # print(last_profit)
        self.p_nodiff = p

    def solve_p_grad_all(self, start_pos, ub):
        lr = self.lr_p
        n = self.sim.G.n
        v = self.sim.G.v
        in_neighbors, in_degree = self.sim.G.in_neighbors, self.sim.G.in_degree_adj
        beta = self.sim.beta

        p = start_pos
        p_user = p.copy()
        mu = self.sim.run_fixed_point(p_user)

        last_p = -np.ones(self.K)
        last_profit = 0
        while np.linalg.norm(p - last_p, np.inf) > 1e-2:
            last_p = p
            y_neighbor_mu = np.array([mu[i].sum() for i in in_neighbors])
            exp = np.exp(v + beta * y_neighbor_mu / in_degree - self.alpha * p_user)
            part_h_p = - self.alpha * self.W.T * (exp / (1 + exp) ** 2).T
            part_h_mu = self.sim.G.mat_adj * (exp / (1 + exp) ** 2 * beta / in_degree).T

            inv = np.identity(n) + part_h_mu + part_h_mu@part_h_mu
            grad_pi_p = part_h_p @ inv @ p + mu
            # print(grad_pi_p)
            p = p + grad_pi_p * lr
            p[p < 0] = 0
            p[p > ub] = ub[p > ub]
            p_user = p.copy()
            mu = self.sim.run_fixed_point(p_user)

            if np.sum(mu * p_user) < last_profit:
                lr = lr / 2
            last_profit = np.sum(mu * p_user)
            # print(last_profit)
        self.p = p

    def solve_p_grad_all_transient(self, mu_0, start_pos, ub):
        lr = self.lr_p
        n = self.sim.G.n
        v = self.sim.G.v
        in_neighbors, in_degree = self.sim.G.in_neighbors, self.sim.G.in_degree_adj
        beta = self.sim.beta

        p = start_pos
        p_user = p.copy()
        y_neighbor_mu = np.array([mu_0[i].sum() for i in in_neighbors])
        exp = np.exp(v + beta * y_neighbor_mu / in_degree-self.alpha*p_user)
        mu = 1-1/(1+exp)

        last_p = -np.ones(self.K)
        last_profit = 0
        while np.linalg.norm(p - last_p, np.inf) > 1e-2:
            last_p = p
            y_neighbor_mu = np.array([mu_0[i].sum() for i in in_neighbors])
            exp = np.exp(v + beta * y_neighbor_mu / in_degree - self.alpha * p_user)
            part_h_p = - self.alpha * self.W.T * (exp / (1 + exp) ** 2).T
            
            grad_pi_p = part_h_p @ p + mu
            # print(grad_pi_p)
            p = p + grad_pi_p * lr
            p[p < 0] = 0
            p[p > ub] = ub[p > ub]
            p_user = p.copy()
            exp = np.exp(v + beta * y_neighbor_mu / in_degree-self.alpha*p_user)
            mu = 1-1/(1+exp)

            if np.sum(mu * p_user) < last_profit:
                lr = lr / 2
            last_profit = np.sum(mu * p_user)
            # print(last_profit)
        self.p = p

    def solve_p_grad_nodiff_all(self,start_pos,ub):
        lr = self.lr_p
        n = self.sim.G.n
        v = self.sim.G.v
        in_neighbors, in_degree = self.sim.G.in_neighbors, self.sim.G.in_degree_adj

        p = start_pos
        p_user = p.copy()
        mu = 1 - 1/(1+np.exp(v-self.alpha * p_user))

        last_p = -np.ones(self.K)
        last_profit = 0
        while np.linalg.norm(p-last_p,np.inf)>1e-2:
            last_p = p
            exp = np.exp(v - self.alpha*p_user)
            part_h_p = - self.alpha * self.W.T * (exp / (1+exp)**2).T

            grad_pi_p = part_h_p @ p + mu
            p = p + grad_pi_p * lr
            p[p <= 0] = 0
            p[p > ub] = ub[p>ub]
            p_user = p.copy()
            mu = 1 - 1/(1+np.exp(v-self.alpha * p_user))

            if np.sum(mu*p_user)<last_profit:
                lr = lr/2
            last_profit = np.sum(mu*p_user)
            # print(last_profit)
        self.p_nodiff = p

