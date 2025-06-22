
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def update_model(source, target, tau):
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=[64 for _ in range(2)], device=None):
        super(Actor, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layers = nn.ModuleList()
        input_dims = [obs_dim] + hidden
        output_dims = hidden + [action_dim]
        for in_dim, out_dim in zip(input_dims[:-1], output_dims[:-1]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Linear(input_dims[-1], output_dims[-1]))
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        for layer in self.layers:
            x = layer(x)
        return F.softmax(x, dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, output_dim=1, hidden=[64 for _ in range(2)], device=None):
        super(Critic, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layers = nn.ModuleList()
        input_dims = [state_dim + action_dim] + hidden
        output_dims = hidden + [output_dim]
        for in_dim, out_dim in zip(input_dims[:-1], output_dims[:-1]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Linear(input_dims[-1], output_dims[-1]))
        self.to(self.device)

    def forward(self, state, action):
        state = state.to(self.device)
        action = action.to(self.device)
        x = torch.cat([state, action], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.memory)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards, dones = zip(*transitions)
        states = torch.tensor(np.vstack(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.vstack(actions), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.vstack(next_states), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.vstack(rewards), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.vstack(dones), dtype=torch.float32).to(self.device)
        return states, actions, next_states, rewards, dones

class MADDPG(nn.Module):
    def __init__(self, n_agents, obs_dims, action_dim, lr=1e-4, gamma=0.95, batch_size=64, tau=1e-3):
        super(MADDPG, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_agents = n_agents
        self.obs_dims = obs_dims
        self.action_dim = action_dim
        self.lr_base = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.init_actors()
        self.init_critics()
        self.init_optimizers()
        self.mse_loss = nn.MSELoss()
        self.memory = ReplayMemory(capacity=2000000)
        self.eps = 0.9
        self.eps_decay = 0.9993
        self.eps_threshold = 0.01
        self.step = 0
        self.to(self.device)

    def init_actors(self):
        self.actors = nn.ModuleList([Actor(self.obs_dims[i], self.action_dim, device=self.device) for i in range(self.n_agents)])
        self.target_actors = nn.ModuleList([Actor(self.obs_dims[i], self.action_dim, device=self.device) for i in range(self.n_agents)])
        for i in range(self.n_agents):
            update_model(self.actors[i], self.target_actors[i], tau=1.0)

    def init_critics(self):
        total_obs_dim = sum(self.obs_dims)
        self.critics = nn.ModuleList([Critic(total_obs_dim, self.action_dim * self.n_agents, device=self.device) for _ in range(self.n_agents)])
        self.target_critics = nn.ModuleList([Critic(total_obs_dim, self.action_dim * self.n_agents, device=self.device) for _ in range(self.n_agents)])
        for i in range(self.n_agents):
            update_model(self.critics[i], self.target_critics[i], tau=1.0)

    def init_optimizers(self):
        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=self.lr_base) for i in range(self.n_agents)]
        self.critic_optimizers = [optim.Adam(self.critics[i].parameters(), lr=self.lr_base) for i in range(self.n_agents)]

    def set_lr(self, is_emergency, episode):
        lr = self.lr_base
        for optimizer in self.actor_optimizers + self.critic_optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def select_action(self, obs, i):
        obs = torch.from_numpy(obs).float().to(self.device)
        if random.random() < self.eps:
            action = np.random.randint(self.action_dim)
            action_prob = np.zeros(self.action_dim)
            action_prob[action] = 1.0
        else:
            action_prob = self.actors[i](obs).detach().cpu().numpy()
            action = np.argmax(action_prob)
        return action, action_prob

    def push(self, transition):
        before_state, actions, state, rewards, dones = transition
        before_state_flat = np.concatenate([s.flatten() for s in before_state])
        state_flat = np.concatenate([s.flatten() for s in state])
        actions_flat = np.concatenate([np.array(a).flatten() for a in actions])
        transition = (before_state_flat, actions_flat, state_flat, rewards, dones)
        self.memory.push(transition)

    def train_start(self):
        return len(self.memory) >= self.batch_size

    def train_model(self, i):
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
        current_q = self.critics[i](states, actions)

        obs_cumsum = np.cumsum([0] + self.obs_dims)
        next_actions = []
        for j in range(self.n_agents):
            agent_next_states = next_states[:, obs_cumsum[j]:obs_cumsum[j+1]]
            action = self.target_actors[j](agent_next_states)
            next_actions.append(action)
        next_actions = torch.hstack(next_actions)

        next_q = self.target_critics[i](next_states, next_actions).detach()
        target_q = rewards[:, i].view(self.batch_size, 1) + self.gamma * (1.0 - dones) * next_q
        value_loss = self.mse_loss(target_q, current_q)

        self.critic_optimizers[i].zero_grad()
        value_loss.backward()
        self.critic_optimizers[i].step()

        actions = []
        for j in range(self.n_agents):
            agent_states = states[:, obs_cumsum[j]:obs_cumsum[j+1]]
            action = self.actors[j](agent_states)
            actions.append(action)
        actions = torch.hstack(actions)

        policy_loss = -self.critics[i](states, actions).mean()
        self.actor_optimizers[i].zero_grad()
        policy_loss.backward()
        self.actor_optimizers[i].step()

        update_model(self.actors[i], self.target_actors[i], self.tau)
        update_model(self.critics[i], self.target_critics[i], self.tau)
        return policy_loss.item(), value_loss.item()

    def update_eps(self, episode):
        self.step += 1
        if episode < 250:
            if self.step % 250 == 0:
                self.eps = max(0.01, self.eps * 0.995)
        else:
            self.eps = 0

    def update_eps_emergency(self, episode):
        self.step += 1
        if episode < 250:
            if self.step % 250 == 0:
                self.eps = max(0.05, self.eps * 0.995)
        else:
            self.eps = 0

    def save_model(self, file_name):
        torch.save(self.state_dict(), file_name)

    def load_model(self, file_name):
        self.load_state_dict(torch.load(file_name))
