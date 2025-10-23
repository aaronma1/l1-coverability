import numpy as np
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal

import gymnasium as gym
from gymnasium import wrappers
import base_utils
import copy

import gc

# Get the initial zero-state for the env.
def init_state(env):
    if env == "Pendulum-v0":
        return [np.pi, 0] 
    elif env == "MountainCarContinuous-v0":
        return [-0.50, 0]

def get_obs(state):
    if base_utils.args.env == "Pendulum-v0":
        theta, thetadot = state
        return np.array([np.cos(theta), np.sin(theta), thetadot])
    elif base_utils.args.env == "MountainCarContinuous-v0":
        return np.array(state)

class CartEntropyPolicy(nn.Module):
    def __init__(self, env, gamma, lr, obs_dim, action_dim):
        super(CartEntropyPolicy, self).__init__()

        self.affine1 = nn.Linear(obs_dim, 128)
        self.middle = nn.Linear(128, 128)
        self.affine2 = nn.Linear(128, action_dim)

        torch.nn.init.xavier_uniform_(self.affine1.weight)
        torch.nn.init.xavier_uniform_(self.middle.weight)
        torch.nn.init.xavier_uniform_(self.affine2.weight)

        self.saved_log_probs = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.eps = np.finfo(np.float32).eps.item()

        self.env = env
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.init_state = np.array(init_state(base_utils.args.env))
        # self.env.seed(int(time.time())) # seed environment

    def init(self, init_policy):
        print("init to policy")
        self.load_state_dict(init_policy.state_dict())

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.middle(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def get_probs(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        return probs

    def select_action_no_grad(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()

        if (action.item() == 1):
            return [0]
        elif (action.item() == 0):
            return [-1]
        return [1]

    def select_action(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))

        if (action.item() == 1):
            return [0]
        elif (action.item() == 0):
            return [-1]
        return [1]

    def update_policy(self):
        R = 0
        policy_loss = [] #
        rewards = []

        #Get discounted rewards from the episode.
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)

        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward.float())

        self.optimizer.zero_grad(set_to_none=True)
        policy_loss = torch.cat(policy_loss).sum() 
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]

        return policy_loss

    def get_initial_state(self):
        if base_utils.args.env == "Pendulum-v0":
            self.env.unwrapped.state = [np.pi, 0] 
            theta, thetadot = self.env.unwrapped.state
            return np.array([np.cos(theta), np.sin(theta), thetadot])
        elif base_utils.args.env == "MountainCarContinuous-v0":
            self.env.unwrapped.state = [-0.50, 0]
            return np.array(self.env.unwrapped.state)

    def get_obs(self):
        if base_utils.args.env == "Pendulum-v0":
            theta, thetadot = self.env.unwrapped.state
            return np.array([np.cos(theta), np.sin(theta), thetadot])
        elif base_utils.args.env == "MountainCarContinuous-v0":
            return np.array(self.env.unwrapped.state)

    def learn_policy(self, reward_fn, det_initial_state=False, sa_reward=True, true_reward=False,
        episodes=1000, train_steps=1000, 
        initial_state=[], start_steps=10000):

        if det_initial_state:
            initial_state = self.init_state

        running_reward = 0
        running_loss = 0
        for i_episode in range(episodes):
            if det_initial_state:
                self.env.unwrapped.reset_state = initial_state
            self.env.reset()
            state = self.get_obs()
            ep_reward = 0
            for t in range(train_steps):  # Don't infinite loop while learning
                action = self.select_action(state)
                if true_reward:
                    state, reward, done, _ = self.env.step(action)
                else:
                    tmp = copy.deepcopy(base_utils.discretize_state(state))
                    tmp.append(action[0])
                    if sa_reward:
                        reward = reward_fn[tuple(tmp)] #reward fn is a fn state-action pairs. applied before.
                        state, _, done, _ = self.env.step(action)
                    else:
                        state, _, done, _ = self.env.step(action)
                        reward = reward_fn[tuple(base_utils.discretize_state(state))] #reward fn is a fn of states. applied after.
                    del tmp
                ep_reward += reward
                self.rewards.append(reward)
                if done:
                    if det_initial_state:
                        self.env.unwrapped.reset_state = initial_state
                    self.env.reset()
                    state = self.get_obs()

            running_reward = running_reward * (1-0.05) + ep_reward * 0.05
            if (i_episode == 0):
                running_reward = ep_reward
            
            loss = self.update_policy()
            running_loss = running_loss * (1-.005) + loss*0.05

            gc.collect()

            # Log to console.
            if (i_episode) % 100 == 0:
                print('Episode {}\tEpisode reward {:.2f}\tRunning reward: {:.2f}\tLoss: {:.2f}'.format(
                    i_episode, ep_reward, running_reward, running_loss))

    #collects one rollout of current policy, returns reward and occupancy data. trajectory is of length T (or termination, whichever comes first)
    def execute_internal(self, env, T, reward_fn, sa_reward, true_reward, state):
        p = np.zeros(shape=(tuple(base_utils.num_states)))
        p_sa = np.zeros(shape=(tuple(base_utils.num_sa)))
        total_reward = 0.
        exited = False

        count_sr = np.zeros(shape=(tuple(base_utils.num_states + base_utils.num_states)))
        last_state = state


        for t in range(T):  
            p[tuple(base_utils.discretize_state(state))] += 1    
            action = self.select_action_no_grad(state)[0]
            tmp = copy.deepcopy(base_utils.discretize_state(state))
            tmp.append(action)
            p_sa[tuple(tmp)] +=1 
            if true_reward:
                state, reward, done, _ = env.step([action]) 
            else:
                if sa_reward:
                    reward = reward_fn[tuple(tmp)] #reward fn is a fn state-action pairs. applied before.
                    state, _, done, _ = env.step([action]) 
                else:
                    state, _, done, _ = env.step([action]) 
                    reward = reward_fn[tuple(base_utils.discretize_state(state))]  #reward fn is a fn of states. applied after update. 

            #compute sr
            last_state_features = copy.deepcopy(base_utils.discretize_state(last_state))
            count_sr[tuple(last_state_features) + tuple(tmp[:-2]) ] += 1
            last_state = state

            total_reward += reward
            del tmp
            del last_state_features
            
            if done:
                exited = True
                final_t = t + 1
                break

        env.close()

        if exited:
            return p/float(final_t), p_sa/float(final_t), total_reward, count_sr
        else:
            return p/float(T), p_sa/float(T), total_reward, count_sr

    #collect T rollouts from current policy
    def execute(self, T, reward_fn, sa_reward=True,true_reward=False,num_rollouts=1, initial_state=[], video_dir=''):

        p = np.zeros(shape=(tuple(base_utils.num_states)))
        p_sa = np.zeros(shape=(tuple(base_utils.num_sa)))

    
        p_pi = np.zeros(shape=(tuple(base_utils.num_states + base_utils.num_states)))
        

        total_reward = 0.

        for r in range(num_rollouts):
            if len(initial_state) == 0:
                initial_state = self.env.reset() # get random starting location
            else:
                self.env.unwrapped.reset_state = initial_state 
                state = self.env.reset()
                state = self.get_obs()

                p_int,p_sa_int,total_reward_int, p_pi_ep = self.execute_internal(self.env, T, reward_fn,sa_reward,true_reward,state)
                p += p_int
                p_sa += p_sa_int
                p_pi += p_pi_ep 
                total_reward += total_reward_int

        return p/float(num_rollouts), p_sa/float(num_rollouts), total_reward/float(num_rollouts) , p_pi/ (T*num_rollouts)

    def execute_random_internal(self, env, T, reward_fn, state):
        p = np.zeros(shape=(tuple(base_utils.num_states)))
        p_sa = np.zeros(shape=(tuple(base_utils.num_sa)))
        total_reward = 0.
        exited = False


        transitions = np.zeros(shape=(tuple(base_utils.num_states + base_utils.num_states)))
        last_state = state

        for t in range(T):  
            p[tuple(base_utils.discretize_state(state))] += 1
            r = random.random()
            action = -1
            if (r < 1/3.):
                action = 0
            elif r < 2/3.:
                action = 1

            state_features = copy.deepcopy(base_utils.discretize_state(state))
            reward = reward_fn[tuple(state_features) + (action,)] #reward fn is a fn state-action pairs
            total_reward += reward
            p_sa[tuple(state_features) + (action, )] +=1 

            last_state_features = copy.deepcopy(base_utils.discretize_state(last_state))
            transitions[tuple(last_state_features) + tuple(state_features)] += 1
            last_state = state

            del state_features
            del last_state_features

            state, reward, terminated, truncated, info = self.env.step([action])

            print(state, action)

            done = terminated or truncated



            if done:
                exited = True
                final_t = t + 1
                break
            
        if exited:
            return p/float(final_t), p_sa/float(final_t), total_reward, transitions
        else:
            return p/float(T), p_sa/float(T), total_reward, transitions

    def execute_random(self, T, reward_fn, num_rollouts=1, initial_state=[]):
        p = np.zeros(shape=(tuple(base_utils.num_states)))
        p_sa = np.zeros(shape=(tuple(base_utils.num_sa)))
        total_reward = 0.

        transitions = np.zeros(shape=(tuple(base_utils.num_states + base_utils.num_states)))

        for r in range(num_rollouts):
            if len(initial_state) == 0:
                initial_state = self.init_state

            else:
                self.env.unwrapped.reset_state = initial_state 
                state = self.env.reset()
                state = self.get_obs()

                p_int, p_sa_int, total_reward_int, ep_transitions = self.execute_random_internal(self.env, T, reward_fn, state)
                p += p_int
                p_sa += p_sa_int
                total_reward += total_reward_int

                transitions += ep_transitions


        return p/float(num_rollouts), p_sa/float(num_rollouts), total_reward/float(num_rollouts), transitions

    def save(self, filename):
        self.env.close()
        torch.save(self, filename)