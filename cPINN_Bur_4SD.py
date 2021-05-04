#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:17:51 2019

@author:  Dr. Ameya D. Jagtap, Division of Applied Mathematics, Brown University, USA

References: 
1. A. D. Jagtap, E. Kharazmi and G.E. Karnidakis, Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward and inverse problems Computer Methods in Applied Mechanics and Engineering, 365, 113028, 2020.

2. A.D. Jagtap, G.E. Karniadakis, Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations, Communications in Computational Physics, 28 (5), 2002-2041, 2020
"""

import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from plotting import newfig, savefig
import time
import matplotlib.gridspec as gridspec


np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class

    def __init__(self, X_u_train_total, u_train_total, X_f_train_total, X_f_inter_total, \
                              NN_layers_total, beta, nu, Num_interface, Num_subdomain, x_interface):
        
        
        self.x_u_total       = X_u_train_total
        self.u_total         = u_train_total
        self.x_f_total       = X_f_train_total
        self.x_f_inter_total = X_f_inter_total
        self.layers          = NN_layers_total
        self.Num_interface   = Num_interface
        self.Num_subdomain   = Num_subdomain
        self.x_interface     = x_interface

        self.lb1 = x_interface[0]
        self.ub1 = 1.0
        self.lb2 = x_interface[1]
        self.ub2 = 1.0
        self.lb3 = x_interface[2]
        self.ub3 = 1.0
        self.lb4 = x_interface[3]
        self.ub4 = 1.0      
    
        self.x_u1 = X_u_train_total[0][:,0:1]
        self.t_u1 = X_u_train_total[0][:,1:2]
        self.u1   = u_train_total[0]
        self.x_f1 = X_f_train_total[0][:,0:1]
        self.t_f1 = X_f_train_total[0][:,1:2]
        #++++++++++++++++++++++++++++++++++++
        self.x_u2 = X_u_train_total[1][:,0:1]
        self.t_u2 = X_u_train_total[1][:,1:2]
        self.u2   = u_train_total[1]
        self.x_f2 = X_f_train_total[1][:,0:1]
        self.t_f2 = X_f_train_total[1][:,1:2]        
        #++++++++++++++++++++++++++++++++++++
        self.x_u3 = X_u_train_total[2][:,0:1]
        self.t_u3 = X_u_train_total[2][:,1:2]
        self.u3   = u_train_total[2]
        self.x_f3 = X_f_train_total[2][:,0:1]
        self.t_f3 = X_f_train_total[2][:,1:2]

        self.x_u4 = X_u_train_total[3][:,0:1]
        self.t_u4 = X_u_train_total[3][:,1:2]
        self.u4   = u_train_total[3]
        self.x_f4 = X_f_train_total[3][:,0:1]
        self.t_f4 = X_f_train_total[3][:,1:2]
     



        self.x_fi1 = X_f_inter_total[0][:,0:1]
        self.t_fi1 = X_f_inter_total[0][:,1:2]
        self.x_fi2 = X_f_inter_total[1][:,0:1]
        self.t_fi2 = X_f_inter_total[1][:,1:2]

        self.x_fi3 = X_f_inter_total[2][:,0:1]
        self.t_fi3 = X_f_inter_total[2][:,1:2]
       




        self.beta = beta
        self.nu = nu
       

        self.layers1 = self.layers[0]
        self.weights1, self.biases1, self.a1 = self.initialize_NN(self.layers1)
        self.layers2 = self.layers[1]
        self.weights2, self.biases2, self.a2 = self.initialize_NN(self.layers2)
        self.layers3 = self.layers[2]
        self.weights3, self.biases3, self.a3 = self.initialize_NN(self.layers3)
        self.layers4 = self.layers[3]
        self.weights4, self.biases4, self.a4 = self.initialize_NN(self.layers4)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_u1_tf = tf.placeholder(tf.float64, shape=[None, self.x_u1.shape[1]])
        self.t_u1_tf = tf.placeholder(tf.float64, shape=[None, self.t_u1.shape[1]])        
        self.u1_tf   = tf.placeholder(tf.float64, shape=[None, self.u1.shape[1]])
        self.x_f1_tf = tf.placeholder(tf.float64, shape=[None, self.x_f1.shape[1]])
        self.t_f1_tf = tf.placeholder(tf.float64, shape=[None, self.t_f1.shape[1]])
        self.x_fi1_tf = tf.placeholder(tf.float64, shape=[None, self.x_fi1.shape[1]])
        self.t_fi1_tf = tf.placeholder(tf.float64, shape=[None, self.t_fi1.shape[1]])
        
        
        self.x_u2_tf = tf.placeholder(tf.float64, shape=[None, self.x_u2.shape[1]])
        self.t_u2_tf = tf.placeholder(tf.float64, shape=[None, self.t_u2.shape[1]])        
        self.u2_tf   = tf.placeholder(tf.float64, shape=[None, self.u2.shape[1]])
        self.x_f2_tf = tf.placeholder(tf.float64, shape=[None, self.x_f2.shape[1]])
        self.t_f2_tf = tf.placeholder(tf.float64, shape=[None, self.t_f2.shape[1]])
        self.x_fi2_tf = tf.placeholder(tf.float64, shape=[None, self.x_fi2.shape[1]])
        self.t_fi2_tf = tf.placeholder(tf.float64, shape=[None, self.t_fi2.shape[1]])
        
        self.x_u3_tf = tf.placeholder(tf.float64, shape=[None, self.x_u3.shape[1]])
        self.t_u3_tf = tf.placeholder(tf.float64, shape=[None, self.t_u3.shape[1]])        
        self.u3_tf   = tf.placeholder(tf.float64, shape=[None, self.u3.shape[1]])
        self.x_f3_tf = tf.placeholder(tf.float64, shape=[None, self.x_f3.shape[1]])
        self.t_f3_tf = tf.placeholder(tf.float64, shape=[None, self.t_f3.shape[1]])
        self.x_fi3_tf = tf.placeholder(tf.float64, shape=[None, self.x_fi3.shape[1]])
        self.t_fi3_tf = tf.placeholder(tf.float64, shape=[None, self.t_fi3.shape[1]])
        
        self.x_u4_tf = tf.placeholder(tf.float64, shape=[None, self.x_u4.shape[1]])
        self.t_u4_tf = tf.placeholder(tf.float64, shape=[None, self.t_u4.shape[1]])        
        self.u4_tf   = tf.placeholder(tf.float64, shape=[None, self.u4.shape[1]])
        self.x_f4_tf = tf.placeholder(tf.float64, shape=[None, self.x_f4.shape[1]])
        self.t_f4_tf = tf.placeholder(tf.float64, shape=[None, self.t_f4.shape[1]])

        self.u1_pred = self.net_u1(self.x_u1_tf, self.t_u1_tf)  
        self.u2_pred = self.net_u2(self.x_u2_tf, self.t_u2_tf)
        self.u3_pred = self.net_u3(self.x_u3_tf, self.t_u3_tf)
        self.u4_pred = self.net_u4(self.x_u4_tf, self.t_u4_tf)

        self.f1_pred, self.f2_pred, self.f3_pred, self.f4_pred, self.fi1_pred, self.fi2_pred, self.fi3_pred,  \
        self.uavgi1_pred, self.uavgi2_pred, self.uavgi3_pred, \
        self.u1i1_pred, self.u2i1_pred, self.u2i2_pred, self.u3i2_pred, self.u3i3_pred, self.u4i3_pred\
        = self.net_f(self.x_f1_tf, self.t_f1_tf,self.x_f2_tf, self.t_f2_tf,\
                     self.x_f3_tf, self.t_f3_tf,self.x_f4_tf, self.t_f4_tf,\
                     self.x_fi1_tf, self.t_fi1_tf,self.x_fi2_tf, self.t_fi2_tf,\
                     self.x_fi3_tf, self.t_fi3_tf)
      
        self.loss1 = 20*(tf.reduce_mean(tf.square(self.u1_tf - self.u1_pred))) \
                    +tf.reduce_mean(tf.square(self.f1_pred))+ 20*tf.reduce_mean(tf.square(self.fi1_pred))\
                    + 20*tf.reduce_mean(tf.square(self.u1i1_pred - self.uavgi1_pred))
                    
        self.loss2 = 20*(tf.reduce_mean(tf.square(self.u2_tf - self.u2_pred)))\
                    +tf.reduce_mean(tf.square(self.f2_pred)) + 20*(tf.reduce_mean(tf.square(self.fi1_pred))+tf.reduce_mean(tf.square(self.fi2_pred)))\
                    + 20*tf.reduce_mean(tf.square(self.u2i1_pred - self.uavgi1_pred)) + 20*tf.reduce_mean(tf.square(self.u2i2_pred - self.uavgi2_pred))
                    
        self.loss3 = 20*(tf.reduce_mean(tf.square(self.u3_tf - self.u3_pred)))\
                    +tf.reduce_mean(tf.square(self.f3_pred))+20*(tf.reduce_mean(tf.square(self.fi2_pred))+tf.reduce_mean(tf.square(self.fi3_pred)))\
                    + 20*tf.reduce_mean(tf.square(self.u3i2_pred - self.uavgi2_pred)) + 20*tf.reduce_mean(tf.square(self.u3i3_pred - self.uavgi3_pred))
                                 
                    
        self.loss4 = 20*(tf.reduce_mean(tf.square(self.u4_tf - self.u4_pred)))\
                    +tf.reduce_mean(tf.square(self.f4_pred))+20*tf.reduce_mean(tf.square(self.fi3_pred))\
                    + 20*tf.reduce_mean(tf.square(self.u4i3_pred - self.uavgi3_pred))

      
        self.optimizer_Adam = tf.train.AdamOptimizer(0.0008)
        self.train_op_Adam1 = self.optimizer_Adam.minimize(self.loss1) 
        self.train_op_Adam2 = self.optimizer_Adam.minimize(self.loss2)        
        self.train_op_Adam3 = self.optimizer_Adam.minimize(self.loss3)
        self.train_op_Adam4 = self.optimizer_Adam.minimize(self.loss4)

        init = tf.global_variables_initializer()
        self.sess.run(init)




    def F_transform(self, fun):
        
        F_T = np.fft.fft(fun)       
        return F_T
    
                
               
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float64), dtype=tf.float64)
            a = tf.Variable(0.05, dtype=tf.float64)
            weights.append(W)
            biases.append(b)  

        return weights, biases, a
         
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.to_double(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev)), dtype=tf.float64)
    
    def neural_net(self, X, lb, ub, weights, biases, a):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - lb)/(ub - lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l] 
            H = tf.tanh(20*a*tf.add(tf.matmul(H, W), b)) 
            # POWER WEIGHTS & BIASES
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H,W), b)
        return Y
            
    def net_u1(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.lb1, self.ub1, self.weights1, self.biases1, self.a1)
        return u
    
    def net_u2(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.lb2, self.ub2, self.weights2, self.biases2, self.a2)
        return u

    def net_u3(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.lb3, self.ub3, self.weights3, self.biases3, self.a3)
        return u

    def net_u4(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.lb4, self.ub4, self.weights4, self.biases4, self.a4)
        return u


    def net_f(self, x1, t1, x2, t2, x3, t3, x4, t4, xi1, ti1, xi2, ti2, xi3, ti3):

        u1    = self.net_u1(x1,t1) 
        u1_t  = tf.gradients(u1, t1)[0]
        u1_x  = tf.gradients(u1, x1)[0]
        u1_xx = tf.gradients(u1_x, x1)[0]
        u1i1   = self.net_u1(xi1,ti1) 
        u1i1_x = tf.gradients(u1i1, xi1)[0]
        u2    = self.net_u2(x2,t2) 
        u2_t  = tf.gradients(u2, t2)[0]
        u2_x  = tf.gradients(u2, x2)[0]
        u2_xx = tf.gradients(u2_x, x2)[0]
        u2i1   = self.net_u2(xi1,ti1) 
        u2i1_x = tf.gradients(u2i1, xi1)[0]
        u2i1_xx = tf.gradients(u2i1_x, xi1)[0]
        
        u2i2   = self.net_u2(xi2,ti2) 
        u2i2_x = tf.gradients(u2i2, xi2)[0]
        u3    = self.net_u3(x3,t3) 
        u3_t  = tf.gradients(u3, t3)[0]
        u3_x  = tf.gradients(u3, x3)[0]
        u3_xx = tf.gradients(u3_x, x3)[0]

        u3i2   = self.net_u3(xi2,ti2) 
        u3i2_x = tf.gradients(u3i2, xi2)[0]
        u3i2_xx = tf.gradients(u3i2_x, xi2)[0]

        u3i3   = self.net_u3(xi3,ti3) 
        u3i3_x = tf.gradients(u3i3, xi3)[0]
        u4    = self.net_u4(x4,t4) 
        u4_t  = tf.gradients(u4, t4)[0]
        u4_x  = tf.gradients(u4, x4)[0]
        u4_xx = tf.gradients(u4_x, x4)[0]
        u4i3   = self.net_u4(xi3,ti3) 
        u4i3_x = tf.gradients(u4i3, xi3)[0]
        u4i3_xx = tf.gradients(u4i3_x, xi3)[0]
        uavgi1 = (u1i1 + u2i1)/2
        uavgi2 = (u2i2 + u3i2)/2
        uavgi3 = (u3i3 + u4i3)/2


        f1 = u1_t + u1*u1_x - self.nu*u1_xx
        f2 = u2_t + u2*u2_x - self.nu*u2_xx
        f3 = u3_t + u3*u3_x - self.nu*u3_xx
        f4 = u4_t + u4*u4_x - self.nu*u4_xx

        fi1 = u1i1**2/2-self.nu*u1i1_x - (u2i1**2/2-self.nu*u2i1_xx)
        fi2 = u2i2**2/2-self.nu*u2i2_x - (u3i2**2/2-self.nu*u3i2_xx)
        fi3 = u3i3**2/2-self.nu*u3i3_x - (u4i3**2/2-self.nu*u4i3_xx)

        

        return f1, f2, f3, f4, fi1, fi2, fi3, uavgi1, uavgi2, uavgi3,\
                u1i1, u2i1, u2i2, u3i2, u3i3, u4i3

       
    def train(self,nIter,X_star1, X_star2, X_star3, X_star4, u1_star,u2_star,u3_star,u4_star):#, X_star, X, T, Exact):

        tf_dict = {self.x_u1_tf: self.x_u1, self.t_u1_tf: self.t_u1, self.u1_tf: self.u1,
                   self.x_u2_tf: self.x_u2, self.t_u2_tf: self.t_u2, self.u2_tf: self.u2,
                   self.x_u3_tf: self.x_u3, self.t_u3_tf: self.t_u3, self.u3_tf: self.u3,
                   self.x_u4_tf: self.x_u4, self.t_u4_tf: self.t_u4, self.u4_tf: self.u4,
                   self.x_f1_tf: self.x_f1, self.t_f1_tf: self.t_f1,
                   self.x_f2_tf: self.x_f2, self.t_f2_tf: self.t_f2,
                   self.x_f3_tf: self.x_f3, self.t_f3_tf: self.t_f3,
                   self.x_f4_tf: self.x_f4, self.t_f4_tf: self.t_f4,
                   self.x_fi1_tf: self.x_fi1, self.t_fi1_tf: self.t_fi1,
                   self.x_fi2_tf: self.x_fi2, self.t_fi2_tf: self.t_fi2,
                   self.x_fi3_tf: self.x_fi3, self.t_fi3_tf: self.t_fi3}
                                                                  

        MSE1_history=[]
        MSE2_history=[]
        MSE3_history=[]
        MSE4_history=[]
        a1_history=[]
        a2_history=[]
        a3_history=[]
        a4_history=[]
        L2error_u = []
        
        u_star = np.concatenate([u1_star, u2_star, u3_star, u4_star])
        for it in range(nIter):
            self.sess.run(self.train_op_Adam1, tf_dict)
            self.sess.run(self.train_op_Adam2, tf_dict)
            self.sess.run(self.train_op_Adam3, tf_dict)
            self.sess.run(self.train_op_Adam4, tf_dict)
            
            if it %10 == 0:
                #elapsed = time.time() - start_time
                loss1_value = self.sess.run(self.loss1, tf_dict)
                loss2_value = self.sess.run(self.loss2, tf_dict)
                loss3_value = self.sess.run(self.loss3, tf_dict)
                loss4_value = self.sess.run(self.loss4, tf_dict)
                
                a1_value = self.sess.run(self.a1, tf_dict)
                a2_value = self.sess.run(self.a2, tf_dict)
                a3_value = self.sess.run(self.a3, tf_dict)
                a4_value = self.sess.run(self.a4, tf_dict)
                

                print('It: %d, Loss1: %.3e, Loss2: %.3e, Loss3: %.3e, Loss4: %.3e' \
                      %(it, loss1_value, loss2_value, loss3_value, loss4_value))
                
                
                u1_pred, u2_pred, u3_pred, u4_pred = model.predict(X_star1, X_star2, X_star3, X_star4)
                u_pred = np.concatenate([u1_pred, u2_pred, u3_pred, u4_pred])
                error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
 
                L2error_u.append(error_u)   
                #start_time = time.time()
                MSE1_history.append(loss1_value)
                MSE2_history.append(loss2_value)
                MSE3_history.append(loss3_value)
                MSE4_history.append(loss4_value)

                a1_history.append(a1_value)
                a2_history.append(a2_value)
                a3_history.append(a3_value)
                a4_history.append(a4_value)
        

        return  MSE1_history, MSE2_history, MSE3_history, MSE4_history, \
                a1_history, a2_history , a3_history , a4_history, L2error_u


    def predict(self, X1_star, X2_star, X3_star, X4_star):
                
        u1_star = self.sess.run(self.u1_pred, {self.x_u1_tf: X1_star[:,0:1], self.t_u1_tf: X1_star[:,1:2]})  
        u2_star = self.sess.run(self.u2_pred, {self.x_u2_tf: X2_star[:,0:1], self.t_u2_tf: X2_star[:,1:2]})
        u3_star = self.sess.run(self.u3_pred, {self.x_u3_tf: X3_star[:,0:1], self.t_u3_tf: X3_star[:,1:2]})
        u4_star = self.sess.run(self.u4_pred, {self.x_u4_tf: X4_star[:,0:1], self.t_u4_tf: X4_star[:,1:2]})        
            
        return u1_star, u2_star, u3_star, u4_star
    
if __name__ == "__main__": 
     
    beta = 1e-7
    noise = 0.0  
    nu = 0.01/np.pi#0.0025 
    Max_iter = 15001  
   

    data = scipy.io.loadmat('../DATA/burgers_shock.mat')
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T    
    X, T  = np.meshgrid(x,t)
    u_star= Exact.flatten()[:,None]
    X_star  = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))


    x_interface = np.array([-1, -0.6, 0.2 , 0.5, 1])
    idx_x_interface = np.floor((x_interface+1)/2*len(x)).astype(int)
    Num_interface = len(x_interface) - 2
    Num_subdomain = len(x_interface) - 1
    NN_depth = [4, 6, 6, 4]
    NN_width = [20, 20, 20, 20]
    NN_layers_total = []
    for sd in range(Num_subdomain):
        NN_layer_sd = [2] + [NN_width[sd]] * NN_depth[sd] + [1]
        NN_layers_total.append(NN_layer_sd)
    N_u_boundary        = min(len(t), 4)
    N_u_subdomain_total = idx_x_interface[1:]-idx_x_interface[0:-1] - 25
    N_f_interface       = 99
    N_f_interface_total = Num_interface * [N_f_interface]
    N_f                 = 3000
    N_f_total           = Num_subdomain * [N_f]

    X_u_train_total = []
    u_train_total = []
    X_star_total = []
    u_star_total = []
    X_sd_total = []
    T_sd_total = []
    for sd in range(Num_subdomain):
        t_sd = data['t'].flatten()[:,None]
        x_sd = x[idx_x_interface[sd]:idx_x_interface[sd+1]].flatten()
        x_sd_2 = np.linspace(x_interface[sd],x_interface[sd+1], 100)
        u_sd_star = Exact[:, idx_x_interface[sd]:idx_x_interface[sd+1]].flatten()[:,None]
        u_star_total.append(u_sd_star)
        X_sd, T_sd = np.meshgrid(x_sd,t_sd)
        X_sd_total.append(X_sd)
        T_sd_total.append(T_sd)
        X_star_sd  = np.hstack((X_sd.flatten()[:,None], T_sd.flatten()[:,None]))
        X_star_total.append(X_star_sd)
        xx_sd_init = np.hstack((X_sd[0:1,:].T, T_sd[0:1,:].T))
        uu_sd_init = Exact[0:1, idx_x_interface[sd]:idx_x_interface[sd+1]].T
        
        if sd != 0 and sd!= Num_subdomain-1:
            idx_sd = np.random.choice(xx_sd_init.shape[0], N_u_subdomain_total[sd], replace=False)
            X_u_sd_train = xx_sd_init[idx_sd, :]
            u_sd_train = uu_sd_init[idx_sd,:]
            X_u_train_total.append(X_u_sd_train)
            u_train_total.append(u_sd_train)
        elif sd == 0:
            xx1bdy  = np.hstack((X[:,0:1], T[:,0:1]))
            uu1bdy  = Exact[:,0:1]
            X_u1_train = np.vstack([xx_sd_init, xx1bdy])
            u1_train = np.vstack([uu_sd_init, uu1bdy])
            idx_sd = np.random.choice(X_u1_train.shape[0], N_u_subdomain_total[sd] + N_u_boundary + 100, replace=False)
            X_u_sd_train = X_u1_train[idx_sd, :]
            u_sd_train = u1_train[idx_sd,:]
            X_u_train_total.append(X_u_sd_train)
            u_train_total.append(u_sd_train)
        elif sd == Num_subdomain-1:
            xx2bdy  = np.hstack((X[:,-1:], T[:,-1:]))
            uu2bdy  = Exact[:,-1:]
            X_u2_train = np.vstack([xx_sd_init, xx2bdy])
            u2_train = np.vstack([uu_sd_init, uu2bdy])
            idx_sd = np.random.choice(X_u2_train.shape[0], N_u_subdomain_total[sd] + N_u_boundary + 100, replace=False)
            X_u_sd_train = X_u2_train[idx_sd, :]
            u_sd_train = u2_train[idx_sd,:]
            X_u_train_total.append(X_u_sd_train)
            u_train_total.append(u_sd_train)

    X_f_train_total = []
    for sd in range(Num_subdomain):
        X_f_sd_train_temp = lhs(2, N_f_total[sd])
        X_f_sd_train_x = x_interface[sd] + (x_interface[sd+1] - x_interface[sd])*X_f_sd_train_temp[:,0]
        X_f_sd_train_t = X_f_sd_train_temp[:,1]
        X_f_sd_train   = np.hstack([X_f_sd_train_x[:,None], X_f_sd_train_t[:,None]])
        X_f_train_total.append(X_f_sd_train)
        

    u_star_inter_total = []
    X_f_inter_total = []
    for inter in range(Num_interface):
        t_inter = data['t'].flatten()[:,None]
        x_inter = x[idx_x_interface[inter+1]:idx_x_interface[inter+1]+1].flatten()
        u_star_inter = Exact[:, idx_x_interface[sd]:idx_x_interface[sd]+1].flatten()[:,None]
        u_star_inter_total.append(u_star_inter)
        X_inter, T_inter = np.meshgrid(x_inter,t_inter)
        X_star_inter  = np.hstack((X_inter.flatten()[:,None], T_inter.flatten()[:,None]))
        idx_inter = np.random.choice(X_star_inter.shape[0], N_f_interface_total[inter], replace=False)
        X_f_inter_train = X_star_inter[idx_inter,:]
        X_f_inter_total.append(X_f_inter_train)

       
    model = PhysicsInformedNN(X_u_train_total, u_train_total, X_f_train_total, X_f_inter_total, \
                              NN_layers_total, beta, nu, Num_interface, Num_subdomain, x_interface)

    start_time = time.time()                
    MSE1_hist, MSE2_hist, MSE3_hist, MSE4_hist,a1_hist, a2_hist, a3_hist, a4_hist,L2error_u = model.train(Max_iter,X_star_total[0],X_star_total[1],X_star_total[2],X_star_total[3],u_star_total[0],u_star_total[1],u_star_total[2], u_star_total[3])
 
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))

#%%  
        
    X_f1_train = X_f_train_total[0]
    X_f2_train = X_f_train_total[1]
    X_f3_train = X_f_train_total[2]
    X_f4_train = X_f_train_total[3]
    
    X_fi1_train = X_f_inter_total[0]
    X_fi2_train = X_f_inter_total[1]
    X_fi3_train = X_f_inter_total[2]
    
    X_u1_train = X_u_train_total[0]
    X_u2_train = X_u_train_total[1]
    X_u3_train = X_u_train_total[2]
    X_u4_train = X_u_train_total[3]
    
    
#    fig, ax = newfig(1.0, 1.1)
    fig = plt.figure()
    ax = plt.subplot2grid((1,1), (0,0))
    
    ax.scatter(X_f1_train[:,1], X_f1_train[:,0], color = 'b')
    ax.scatter(X_f2_train[:,1], X_f2_train[:,0], color = 'r')
    ax.scatter(X_f3_train[:,1], X_f3_train[:,0], color = 'g')
    ax.scatter(X_f4_train[:,1], X_f4_train[:,0], color = 'c')
    
    ax.scatter(X_fi1_train[:,1], X_fi1_train[:,0], color = 'k')
    ax.scatter(X_fi2_train[:,1], X_fi2_train[:,0], color = 'k')
    ax.scatter(X_fi3_train[:,1], X_fi3_train[:,0], color = 'k')
        
    ax.scatter(X_u1_train[:,1], X_u1_train[:,0], color = 'c')
    ax.scatter(X_u2_train[:,1], X_u2_train[:,0], color = 'g')
    ax.scatter(X_u3_train[:,1], X_u3_train[:,0], color = 'b')
    ax.scatter(X_u4_train[:,1], X_u4_train[:,0], color = 'y')
    
    X_star1 = X_star_total[0]
    X_star2 = X_star_total[1]
    X_star3 = X_star_total[2]
    X_star4 = X_star_total[3]


    u1_star = u_star_total[0]
    u2_star = u_star_total[1]
    u3_star = u_star_total[2]
    u4_star = u_star_total[3]

    
    u1_pred, u2_pred, u3_pred, u4_pred = model.predict(X_star1, X_star2, X_star3, X_star4)
            

   
    X1, T1 = X_sd_total[0], T_sd_total[0]
    X2, T2 = X_sd_total[1], T_sd_total[1]
    X3, T3 = X_sd_total[2], T_sd_total[2]
    X4, T4 = X_sd_total[3], T_sd_total[3]

    U1_pred = griddata(X_star1, u1_pred.flatten(), (X1, T1), method='cubic')
    U2_pred = griddata(X_star2, u2_pred.flatten(), (X2, T2), method='cubic')
    U3_pred = griddata(X_star3, u3_pred.flatten(), (X3, T3), method='cubic')
    U4_pred = griddata(X_star4, u4_pred.flatten(), (X4, T4), method='cubic')
    
    U_star  = griddata(X_star, u_star.flatten(), (X, T), method='cubic')
    U1_star = griddata(X_star1, u1_star.flatten(), (X1, T1), method='cubic')
    U2_star = griddata(X_star2, u2_star.flatten(), (X2, T2), method='cubic') 
    U3_star = griddata(X_star3, u3_star.flatten(), (X3, T3), method='cubic') 
    U4_star = griddata(X_star4, u4_star.flatten(), (X4, T4), method='cubic') 

    u1_err = abs(u1_pred - u1_star)
    u2_err = abs(u2_pred - u2_star)
    u3_err = abs(u3_pred - u3_star)
    u4_err = abs(u4_pred - u4_star)    
    U1_err = griddata(X_star1, u1_err.flatten(), (X1, T1), method='cubic')
    U2_err = griddata(X_star2, u2_err.flatten(), (X2, T2), method='cubic')
    U3_err = griddata(X_star3, u3_err.flatten(), (X3, T3), method='cubic')
    U4_err = griddata(X_star4, u4_err.flatten(), (X4, T4), method='cubic')
    
############################ Plotting #################>>>>>>>>>>>>>>>>>>
    
    fig, ax = newfig(1.0, 1.1)
#    fig = plt.figure()
    gridspec.GridSpec(1,1)
    
    ax = plt.subplot2grid((1,1), (0,0))
    maxLevel = max(max(u1_star),max(max(u2_star),max(max(u3_star),max(u4_star))))[0]
    minLevel = min(min(u1_star),min(min(u2_star), min(min(u3_star), min(u4_star)) ))[0]
    levels = np.linspace(minLevel-0.01, maxLevel+0.01, 200)
    CS_ext1 = ax.contourf(T, X, U_star, levels=levels, cmap='jet', origin='lower')
    cbar = fig.colorbar(CS_ext1)
#                        , ticks=[-1, -0.5, 0, 0.5, 1])
#    cbar.ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'])
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(-1.01, 1.02)
    #ax_pred.locator_params(nbins=5)
    #ax_pred.set_xticklabels(np.linspace(0,1,5), rotation=0, fontsize=18)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$ u^{Exact} $')
#    for xc in x_interface:
#        ax.axhline(y=xc, linewidth=1, color = 'w')
    
    #fig.tight_layout()
    fig.set_size_inches(w=15,h=8) 
    savefig('./figures/BurExact_4sd')#KdV3SD_PredPlot')
    
#    plt.show()

    fig, ax = newfig(1.0, 1.1)
#    fig = plt.figure()
    gridspec.GridSpec(1,1)
    
    ax = plt.subplot2grid((1,1), (0,0))
    maxLevel = max(max(u1_star),max(max(u2_star),max(max(u3_star),max(u4_star))))[0]
    minLevel = min(min(u1_star),min(min(u2_star), min(min(u3_star), min(u4_star)) ))[0] 
    levels = np.linspace(minLevel-0.01, maxLevel+0.01, 200)
    CS_pred1 = ax.contourf(T1, X1, U1_pred, levels=levels, cmap='jet', origin='lower')
    CS_pred2 = ax.contourf(T2, X2, U2_pred, levels=levels, cmap='jet', origin='lower')
    CS_pred3 = ax.contourf(T3, X3, U3_pred, levels=levels, cmap='jet', origin='lower')
    CS_pred4 = ax.contourf(T4, X4, U4_pred, levels=levels, cmap='jet', origin='lower')
    cbar = fig.colorbar(CS_pred1)
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(-1.01, 1.02)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$ u^{prediction} $')

    for xc in x_interface:
        ax.axhline(y=xc, linewidth=1, color = 'w')
    

    fig.set_size_inches(w=15,h=8)

   
    savefig('./figures/BurPred_4sd')#KdV3SD_PredPlot')
    
#%%   
    fig, ax = newfig(1.0, 1.1)
#    fig = plt.figure()
    gridspec.GridSpec(1,1)
    
    ax = plt.subplot2grid((1,1), (0,0))
    maxerr = max(max(u1_err),max(max(u2_err),max(max(u3_err), max(u4_err))))[0]
    levels = np.linspace(0, maxerr, 200)
#    levels = np.linspace(0, 0.05, 100)
    CS_err1 = ax.contourf(T1, X1, U1_err, levels=levels, cmap='jet', origin='lower')
    CS_err2 = ax.contourf(T2, X2, U2_err, levels=levels, cmap='jet', origin='lower')
    CS_err3 = ax.contourf(T3, X3, U3_err, levels=levels, cmap='jet', origin='lower')
    CS_err4 = ax.contourf(T4, X4, U4_err, levels=levels, cmap='jet', origin='lower')
    cbar = fig.colorbar(CS_err1)
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(-1.01, 1.02)
    #ax.locator_params(nbins=5)
    ax.set_aspect(0.25)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$ point-wise \,\, error $')
    for xc in x_interface:
        ax.axhline(y=xc, linewidth=1, color = 'w')
    
    
    #fig.tight_layout()
    fig.set_size_inches(w=15,h=8)
#    plt.show()
    savefig('./figures/BurError_4sd')#KdV3SD_errorPlot')
    
    ############################# Plotting ###############################
  
    fig, ax = newfig(1.0, 1.1)
#    fig = plt.figure()
    
    a1_hist = np.reshape(a1_hist, (-1, 1)) 
    a2_hist = np.reshape(a2_hist, (-1, 1)) 
    a3_hist = np.reshape(a3_hist, (-1, 1)) 
    a4_hist = np.reshape(a4_hist, (-1, 1)) 
    
    plt.plot(range(1,Max_iter-1,10),20*a1_hist[0:-1],  'r.-', linewidth = 1,  label = '$a_1$')  
    plt.plot(range(1,Max_iter-1,10),20*a2_hist[0:-1],  'b--', linewidth = 1,  label = '$a_2$')  
    plt.plot(range(1,Max_iter-1,10),20*a3_hist[0:-1],  'g:', linewidth = 1,  label = '$a_3$') 
    plt.plot(range(1,Max_iter-1,10),20*a4_hist[0:-1],  'k-', linewidth = 1,  label = '$a_4$') 
    plt.legend(loc='lower right')
    plt.xlabel('$\#$ iterations')

    savefig('./figures/Bur_Ahist_4sd')#KdV3SD_Ahistory') 
    
    fig, ax = newfig(1.0, 1.1)
    plt.plot(range(1,Max_iter-1,10), L2error_u[0:-1],  'b-', linewidth = 1) 
    plt.xlabel('$\#$ iterations')
    plt.ylabel('$L_2$-error')
    plt.yscale('log')
    savefig('./figures/Bur_L2err_4sd')#KdV3SD_Ahistory') 
    
    print(L2error_u[-1])
    
    with open('L2error_Bur4SD_200Wi.mat','wb') as f:
        scipy.io.savemat(f, {'L2error_u': L2error_u})
    
    
    fig, ax = newfig(1.0, 1.1)
#    fig = plt.figure()

    #ax.plot(MSE_hist,  'r-', linewidth = 1)    

    plt.plot(range(1,Max_iter-1,10), MSE1_hist[0:-1],  'r.-', linewidth = 1,  label = 'Sub-PINN-1') 
    plt.plot(range(1,Max_iter-1,10), MSE2_hist[0:-1],  'b--', linewidth = 1,  label = 'Sub-PINN-2')  
    plt.plot(range(1,Max_iter-1,10), MSE3_hist[0:-1],  'g-', linewidth = 1,  label = 'Sub-PINN-3')  
    plt.plot(range(1,Max_iter-1,10), MSE4_hist[0:-1],  'k-', linewidth = 1,  label = 'Sub-PINN-4')  
    
    plt.xlabel('$\#$ iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(loc='upper right')
        
    savefig('./figures/Bur_MSEdomain_4sd')#KdV3SD_MSEhistory') 

    
 
