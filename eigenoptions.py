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
from cart_entropy_policy1 import CartEntropyPolicy1
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




            

def sr_from_transitions(transitions, gamma=0.99, disc_fn = base_utils.discretize_state, step_size=0.01):
    a,b = base_utils.num_states
    sr = np.zeros(shape=(a*b, a*b))
    for trajectory in transitions:
        for sars in trajectory:
            # print(sars)
            # print(disc_fn(sars[0]))

            prev_s = flatten_idx(*disc_fn(sars[0]))
            next_s = flatten_idx(*disc_fn(sars[3]))
            for i in range(a * b):
                delta =  gamma * sr[next_s, i] - sr[prev_s,i] + (1 if i == prev_s else 0)
                sr[prev_s, i] += step_size * delta
    return sr


                
                 






                




# OLD
def compute_SR(transitions, gamma=0.99):
    size = base_utils.num_states[0] * base_utils.num_states[1]
    transitions = flatten_m(transitions)
    plotlib.plot_heatmap(transitions, save_path="out1/transitions_random.png")

    transitions = transitions + transitions.T
    p_pi = np.zeros(shape=(size, size))
    for i in range(size):
        if np.sum(transitions[i, :]) != 0:
            p_pi[i, :] = transitions[i, :]/np.sum(transitions[i,:])
        else:
            p_pi[i,i] = 1.0

    assert np.all(p_pi >= 0) and np.all(p_pi <= 1)

    SR = np.linalg.inv(np.eye(size)-gamma*p_pi)
    assert np.all(SR >= 0)

    plotlib.plot_heatmap(SR, save_path="out1/sr_random.png")
    eigenvalues, eigenvectors = np.linalg.eig(SR)
    idx = np.argsort(eigenvalues.real)
    eigenvalues = np.real_if_close(eigenvalues, tol=1e5)[idx]
    eigenvectors = np.real_if_close(eigenvectors, tol=1e5)[:, idx]

    return SR, eigenvectors, eigenvalues


def p_from_transitions(transitions, disc_fn = base_utils.discretize_state):
    p = np.zeros(shape=base_utils.num_states)
    for trajectory in transitions:
        p[tuple(disc_fn(trajectory[0][0]))] += 1 # add init state 
        for sars in trajectory:
            p[tuple(disc_fn(sars[3]))] += 1
    return p/np.sum(p)




def rod_cycle(env, T, gamma=0.99, lr=1e-3, num_rollouts=10, epochs=10):
    options = []
    zero_reward = np.zeros(shape=(tuple(base_utils.num_sa)))


    # step 0 collect random walk transitions
    rand_policy = CartEntropyPolicy1(env, gamma, lr, base_utils.obs_dim, base_utils.action_dim)
    transitions = rand_policy.execute_random(1000, zero_reward, num_rollouts=num_rollouts,  initial_state=[]) 
    mu = p_from_transitions(transitions)
    SR = sr_from_transitions(transitions, gamma)
    SR += SR.T
    eigenvalues, eigenvectors = np.linalg.eig(SR)
    idx = np.argsort(-eigenvalues.real)
    eigenvalues = np.real_if_close(eigenvalues, tol=1e5)[idx]
    eigenvectors = np.real_if_close(eigenvectors, tol=1e5)[:, idx]
    plotlib.plot_heatmap(SR, save_path=f"out1/random_SR.png")
    plotlib.plot_heatmap(mu, save_path=f"out1/random_mu.png")

    all_transitions = transitions



    for epoch in range(epochs):


        for i in range(3):
            # get eigenvector and use it to learn an option
            eig = unflatten_state(eigenvectors[:, i])
            if np.dot(eigenvectors[:, i], flatten_state(mu)) > 0:
                eig = -eig
            reward = reward_shaping(eig)
            option = CartEntropyPolicy1(env, gamma, lr, base_utils.obs_dim, base_utils.action_dim)
            option.learn_policy(reward, sa_reward=False)

            # collect rollouts from option
            transitions = option.execute(T, eig, sa_reward=False, num_rollouts=num_rollouts)
            mu = p_from_transitions(transitions)
            # plot  
            plotlib.plot_heatmap(reward, save_path=f"out1/epoch{epoch}_eigenvector{i}.png")
            plotlib.plot_heatmap(mu, save_path=f"out1/{epoch}eigenvector{i}-visitation.png")

            all_transitions += transitions

        # after each iteration, recompute sr, eigenvectors, and eigenvalues
        SR = sr_from_transitions(all_transitions)
        eigenvalues, eigenvectors = np.linalg.eig(SR)
        idx = np.argsort(-eigenvalues.real)
        eigenvalues = np.real_if_close(eigenvalues, tol=1e5)[idx]
        eigenvectors = np.real_if_close(eigenvectors, tol=1e5)[:, idx]
        plotlib.plot_heatmap(p_from_transitions(all_transitions), save_path=f"out1/{epoch}_visitation.png")
        plotlib.plot_heatmap(SR, save_path=f"out1/epoch{epoch}_sr.png")
        SR = sr_from_transitions(all_transitions)
        SR += SR.T
        eigenvalues, eigenvectors = np.linalg.eig(SR)
        idx = np.argsort(-eigenvalues.real)
        eigenvalues = np.real_if_close(eigenvalues, tol=1e5)[idx]
        eigenvectors = np.real_if_close(eigenvectors, tol=1e5)[:, idx]

        

    

        








    





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