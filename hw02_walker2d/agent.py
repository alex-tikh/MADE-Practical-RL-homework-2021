import random
import numpy as np
import os
import torch

from torch import nn
from torch.distributions import Normal

HIDDEN_DIM = 256

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super().__init__()
        # Advice: use same log_sigma for all states to improve stability
        # You can do this by defining log_sigma as nn.Parameter(torch.zeros(...))
        self.model = nn.Sequential(
            nn.Linear(state_dim, 2*HIDDEN_DIM),
            nn.ELU(),
            nn.Linear(2*HIDDEN_DIM, HIDDEN_DIM),
            nn.ELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ELU(),
            nn.Linear(HIDDEN_DIM, action_dim)
        )
        self.device = device
        self.sigma = nn.Parameter(torch.zeros(action_dim), requires_grad=True).to(self.device)
        
    def compute_proba(self, state, action):
        # Returns probability of action according to current policy and distribution of actions
        mu = self.model(state)
        sigma = torch.exp(self.sigma)
        dist = Normal(mu, sigma)
        return torch.exp(dist.log_prob(action).sum(-1)), dist
        
    def act(self, state):
        # Returns an action (with tanh), not-transformed action (without tanh) and distribution of non-transformed actions
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        mu = self.model(state)
        sigma = torch.exp(self.sigma)
        dist = Normal(mu, sigma)
        action = dist.sample()
        tanh_action = torch.tanh(action)
        return tanh_action, action, dist

class Agent:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        # self.model = torch.load(__file__[:-8] + "/agent.pkl").to(self.device)
        self.model = Actor(
                state_dim=22,
                action_dim=6,
                device=self.device)
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pt"))
        self.model.to(self.device)
        self.model.eval()
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float().to(self.device)
            action, _, _ = self.model.act(state)
        return action.cpu().numpy()

    def reset(self):
        pass

