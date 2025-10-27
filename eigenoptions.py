import sys
import os
# home_dir = os.getenv('HOME')
# sys.path = ['/home/aaron/Documents/l1-coverability'+'/gym-fork'] + sys.path #change this to your local folder

import time
from datetime import datetime
import logging

import copy

from matplotlib.pylab import f
import numpy as np
import scipy.stats
from scipy.interpolate import interp2d
from scipy.interpolate import UnivariateSpline #changed this
from scipy.stats import norm

import gymnasium as gym
from sympy import unflatten

from cart_entropy_policy import CartEntropyPolicy
import base_utils
import plotting

import torch
from torch.distributions import Normal
import random

import pickle

from itertools import islice
import plotlib

torch.backends.cudnn.enabled = False

#reshapes the reward to be between -1 and 0, 
def reward_shaping(reward_fn):
    r_max = np.max(reward_fn)
    r_min = np.min(reward_fn)
    new_reward = reward_fn
    new_reward -= r_min
    new_reward /= (r_max - r_min)
    return new_reward -1


def flatten_m(matrix):

    a = matrix.shape[0]
    b = matrix.shape[1]


    mat = np.zeros(shape=(a*b, a*b))


    for i in range(a * b):
        for j in range(a * b):
            mat[i][j] = matrix[unflatten_idx(i) + unflatten_idx(j)]

    return mat

def flatten_state(state):
    a,b = base_utils.num_states
    mat = np.zeros(shape=(a*b))

    for i in range(a*b):
        mat[i] = state[unflatten_idx(i)]

    return mat

def unflatten_state(state):
    a, b = base_utils.num_states

    mat = np.zeros(shape=(a, b))

    for i in range(a*b):
        mat[unflatten_idx(i)] = state[i]

    return mat



            

def flatten_idx(x, v):
    return x + base_utils.num_states[0]*v 

def unflatten_idx(i):
    return (i% base_utils.num_states[0], i//base_utils.num_states[0])




def compute_SR(transitions, gamma=0.99):
    size = base_utils.num_states[0] * base_utils.num_states[1]
    transitions = flatten_m(transitions)
    transitions = transitions + transitions.T
    p_pi = np.zeros(shape=(size, size))
    plotlib.plot_heatmap(transitions, save_path="out1/transitions_random.png")
    for i in range(size):
        if np.sum(transitions[i, :]) != 0:
            p_pi[i, :] = transitions[i, :]/np.sum(transitions[i,:])
        else:
            p_pi[i,i] = 1.0

    print(np.all(p_pi >= 0), np.all(p_pi <= 1))

    SR = np.linalg.inv(np.eye(size)-gamma*p_pi)
    print(np.all(SR >= 0))

    plotlib.plot_heatmap(SR, save_path="out1/sr_random.png")

    eigenvalues, eigenvectors = np.linalg.eig(SR)

    idx = np.argsort(eigenvalues.real)
    eigenvalues = np.real_if_close(eigenvalues, tol=1e5)[idx]
    eigenvectors = np.real_if_close(eigenvectors, tol=1e5)[:, idx]

    return SR, eigenvectors, eigenvalues





def rod_cycle(env, T, gamma=0.99, lr=1e-3, num_rollouts=1, epochs=10):
    options = []
    zero_reward = np.zeros(shape=(tuple(base_utils.num_sa)))


    rand_policy = CartEntropyPolicy(env, gamma, lr, base_utils.obs_dim, base_utils.action_dim)
    # mu, _, _, transitions = rand_policy.execute(T, zero_reward, sa_reward=False, num_rollouts=10)
    mu, _, _, transitions = rand_policy.execute_random(1000, zero_reward, num_rollouts=20,  initial_state=[]) 
    SR, eigenvectors, eigenvalues = compute_SR(transitions, gamma)

    plotlib.plot_heatmap(mu, save_path=f"out1/visitaion_random.png")

    print(eigenvalues)
    

    

    for i in range(len(eigenvectors)):
        print(eigenvalues[i])

        # if eigenvalues[i] != 1:

        eig = unflatten_state(eigenvectors[i])



        plotlib.plot_heatmap(eig, save_path=f"out1/eigenvector{i}.png")


            
            # reward = reward_shaping(eig)
            # option = CartEntropyPolicy(env, gamma, lr, base_utils.obs_dim, base_utils.action_dim)
            # option.learn_policy(reward, sa_reward=False)

            # mu, _, _, transitions = option.execute(T, eig, sa_reward=False, num_rollouts=10)
            # print(mu)

            # plotlib.plot_heatmap(mu, save_path=f"out1/eig{i}_visitation.png")
            
            # break
        









    





if __name__ == "__main__":
    # test flatten idx

    # Suppress scientific notation.
    np.set_printoptions(suppress=True, edgeitems=100)
    np.set_printoptions(precision=3, suppress=True)

    env = "MountainCarContinuous-v0"
    # Make environment.
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    # env.seed(int(time.time())) # seed environment

    rod_cycle(env, 1000,  gamma=0.99, lr=1e-3,)


    env.close()
    print("DONE")