import sys
import os
# home_dir = os.getenv('HOME')
# sys.path = ['/home/aaron/Documents/l1-coverability'+'/gym-fork'] + sys.path #change this to your local folder

import time
from datetime import datetime
import logging

import copy

import numpy as np
import scipy.stats
from scipy.interpolate import interp2d
from scipy.interpolate import UnivariateSpline #changed this
from scipy.stats import norm

import gymnasium as gym

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


def rod_cycle(env, T, gamma=0.99, lr=1e-3, num_rollouts=1000, epochs=10):

    options = []


    # how to handle state self loops?

    zero_reward = np.zeros(shape=(tuple(base_utils.num_sa)))
    for _ in range(epochs):
        p_pi = np.zeros(shape=(tuple(base_utils.num_states) + tuple(base_utils.num_states)))

        next_option = CartEntropyPolicy(env, gamma, lr, base_utils.obs_dim, base_utils.action_dim)
        _, _, _, transitions = next_option.execute_random(T, zero_reward, num_rollouts=num_rollouts, initial_state=[]) 

        size = base_utils.num_states[0] * base_utils.num_states[1]

        transitions = transitions.reshape(size, size)
        transitions += transitions.T

        row_sums = transitions.sum(axis=1, keepdims=True)
        p_pi = np.divide(transitions, row_sums, where=row_sums!=0)

        N_BINS=40
        plotlib.plot_transitions(p_pi.reshape(40, 40,40, 40), 40, 40)
        SR = np.linalg.inv(np.eye(size)-gamma*p_pi)

        eigenvalues, eigenvectors = np.linalg.eig(SR)
        idx = np.argsort(eigenvalues.real)
        eigenvalues = np.real_if_close(eigenvalues, tol=1e5)[idx]
        eigenvectors = np.real_if_close(eigenvectors, tol=1e5)[:, idx]

        eigenvectors = eigenvectors.reshape((-1,) +tuple(base_utils.num_states) )

        for i in range(len(eigenvalues)):
            plotlib.plot_heatmap(eigenvectors[i], save_path=f"out1/eigenvector{i}.png")



            






        break





    





if __name__ == "__main__":
    # Suppress scientific notation.
    np.set_printoptions(suppress=True, edgeitems=100)
    np.set_printoptions(precision=3, suppress=True)

    env = "MountainCarContinuous-v0"
    # Make environment.
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    # env.seed(int(time.time())) # seed environment

    rod_cycle(env, 1000,  gamma=0.99, lr=1e-3,)


    env.close()
    print("DONE")