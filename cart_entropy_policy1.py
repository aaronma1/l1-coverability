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

class CartEntropyPolicy1(nn.Module):
    def __init__(self, env, gamma, lr, obs_dim, action_dim):
        super(CartEntropyPolicy1, self).__init__()
        self.device = "cuda"

        hidden_dim=16

        self.affine1 = nn.Linear(obs_dim, hidden_dim, device=self.device)
        self.middle = nn.Linear(hidden_dim, hidden_dim, device=self.device)
        self.affine2 = nn.Linear(hidden_dim, action_dim, device=self.device)

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
        x = x.to(self.device)
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
                done = False
                if true_reward:
                    state, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                else:
                    last_state_features = copy.deepcopy(base_utils.discretize_state(state))
                    last_state_features.append(action[0])
                    if sa_reward:
                        reward = reward_fn[tuple(last_state_features)] #reward fn is a fn state-action pairs. applied before.
                        state, _, terminated, truncated, info = self.env.step(action)
                        done = terminated or truncated
                    else:
                        state, _, terminated, truncated, info = self.env.step(action)
                        reward = reward_fn[tuple(base_utils.discretize_state(state))] #reward fn is a fn of states. applied after.
                        done = terminated or truncated
                    del last_state_features
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
    def execute_internal(self, env, T, reward_fn, sa_reward, true_reward, init_state):
        transitions = []

        state = init_state
        last_state = init_state 

        for t in range(T):  
            # select action
            action = [self.select_action_no_grad(state)[0]]

            # sa reward
            state, reward, terminated, truncated, info = self.env.step(action)
            if not true_reward:
                state_features = copy.deepcopy(base_utils.discretize_state(state))
                if sa_reward:
                    reward = reward_fn[tuple(state_features) + tuple(action)]  #reward fn is a fn of states. applied after update. 
                else:
                    reward = reward_fn[tuple(state_features)]
                del state_features
            done = terminated or truncated
                
            transitions.append([copy.deepcopy(last_state), copy.deepcopy(action), copy.deepcopy(reward), copy.deepcopy(state)])
            last_state = state
            
            if done:
                exited = True
                final_t = t + 1
                break
        env.close()

        return transitions

    def p_from_transitions(transitions, disc_fn = base_utils.discretize_state):
        p = np.zeros(shape=base_utils.num_states)
        
        for trajectory in transitions:

            print(disc_fn(trajectory[0][0]))

            p[disc_fn(trajectory[0][0])] += 1 # add init state 
            for sars in trajectory:
                p[disc_fn(sars[3])] += 1
        return p/np.sum(p)
    


    #collect T rollouts from current policy
    def execute(self, T, reward_fn, sa_reward=True,true_reward=False,num_rollouts=1, initial_state=[], video_dir=''):
        transitions = []

        for r in range(num_rollouts):
            if len(initial_state) == 0:
                initial_state = self.env.reset() # get random starting location
            else:
                self.env.unwrapped.reset_state = initial_state 
                self.env.reset()
                state = self.get_obs()
                transitions.append(self.execute_internal(self.env,T, reward_fn,sa_reward,true_reward,state))
        return transitions

    def execute_random_internal(self, env, T,reward_fn, start_state, true_reward=False, sa_reward=False):
        transitions = []

        state = start_state
        last_state = start_state 

        for t in range(T):  

            # select action
            r = random.random()
            action = [-1]
            if (r < 1/3.):
                action = [0]
            elif r < 2/3.:
                action = [1]

            # env step, 
            state, reward, terminated, truncated, info = self.env.step(action)
            if not true_reward:
                state_features = copy.deepcopy(base_utils.discretize_state(state))
                if sa_reward:
                    reward = reward_fn[tuple(state_features) + tuple(action)]  #reward fn is a fn of states. applied after update. 
                else:
                    reward = reward_fn[tuple(state_features)]
                del state_features
            done = terminated or truncated

            transitions.append([copy.deepcopy(last_state), copy.deepcopy(action), copy.deepcopy(reward), copy.deepcopy(state)])
            last_state = state
            
            if done:
                exited = True
                final_t = t + 1
                break
        env.close()

        return transitions


    #collect T rollouts from current policy

    def execute_random(self, T, reward_fn, num_rollouts=1, initial_state=[], true_reward=False, sa_reward=False):
        transitions = []
        for r in range(num_rollouts):
            self.env.reset()
            state = self.get_obs()
            transitions.append(self.execute_random_internal(self.env, T, reward_fn, state))
        return transitions

    def save(self, filename):
        self.env.close()
        torch.save(self, filename)

if __name__ == "__main__":
    # Suppress scientific notation.
    np.set_printoptions(suppress=True, edgeitems=100)
    np.set_printoptions(precision=3, suppress=True)

    env = "MountainCarContinuous-v0"
    # Make environment.
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    # env.seed(int(time.time())) # seed environment

    zero_reward = np.zeros(shape=(tuple(base_utils.num_sa)))
    rand_policy = CartEntropyPolicy(env, 0.99, 1e-3, base_utils.obs_dim, base_utils.action_dim)
    transitions = rand_policy.execute_random(1000, zero_reward, num_rollouts=10,  initial_state=[]) 