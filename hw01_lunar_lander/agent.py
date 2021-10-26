import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = torch.load(__file__[:-8] + "/agent.pkl").to(self.device)
        
    def act(self, state):
        state = np.array(state, dtype=np.float32)
        with torch.no_grad():
            pred = self.model(torch.from_numpy(state).to(self.device))
            return pred.argmax(-1).cpu().numpy()

