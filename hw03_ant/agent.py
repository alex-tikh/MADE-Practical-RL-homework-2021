import random
import numpy as np
import os
import torch
from torch import nn


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.model(state)
    

class Agent:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        # self.model = torch.load(__file__[:-8] + "/agent.pkl").to(self.device)
        self.model = Actor(
                state_dim=28,
                action_dim=8,
                device=self.device)
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pt"))
        self.model.to(self.device)
        self.model.eval()
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float().to(self.device)
            action = self.model(state)
        return action.cpu().numpy()

    def reset(self):
        pass

