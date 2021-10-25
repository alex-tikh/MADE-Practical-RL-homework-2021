from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import random
import copy

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 5e-4


class DQN:
    def __init__(self, state_dim, action_dim, device):
        self.steps = 0 # Do not change
        self.device = device
        
        self.state_buffer = deque([np.zeros(state_dim) for i in range(INITIAL_STEPS)], maxlen=INITIAL_STEPS)
        self.next_state_buffer = deque([np.zeros(state_dim) for i in range(INITIAL_STEPS)], maxlen=INITIAL_STEPS)
        self.action_buffer = deque([0. for i in range(INITIAL_STEPS)], maxlen=INITIAL_STEPS)
        self.reward_buffer = deque([0. for i in range(INITIAL_STEPS)], maxlen=INITIAL_STEPS)
        self.done_buffer = deque([True for i in range(INITIAL_STEPS)], maxlen=INITIAL_STEPS)
        
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),   
            nn.Linear(64, action_dim)
        ).to(self.device)
        
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = Adam(self.model.parameters(), LEARNING_RATE)

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        state, action, next_state, reward, done = transition
        self.state_buffer.append(state)
        self.next_state_buffer.append(next_state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        batch_idx = np.random.choice(len(self.state_buffer), BATCH_SIZE, replace=False)
        
        torch_state_batch = torch.from_numpy(
            np.array(self.state_buffer)[batch_idx]
        ).to(self.device)
        torch_next_state_batch = torch.from_numpy(
            np.array(self.next_state_buffer)[batch_idx]
        ).to(self.device)
        
        torch_action_batch = torch.from_numpy(
            np.array(self.action_buffer)[batch_idx]
        ).reshape(-1, 1).to(self.device)
        torch_reward_batch = torch.from_numpy(
            np.array(self.reward_buffer)[batch_idx]
        ).reshape(-1, 1).to(self.device)
        torch_done_batch = torch.from_numpy(
            np.array(self.done_buffer)[batch_idx]
        ).reshape(-1, 1).to(self.device)
        
        return torch_state_batch, torch_action_batch, torch_next_state_batch, torch_reward_batch, torch_done_batch
        
    def train_step(self, batch):
        # Use batch to update DQN's network.
        state, action, next_state, reward, done = batch
        
        with torch.no_grad():
            
            target_q = self.target_model(next_state)
            target_q = torch.max(target_q, dim=1).values.reshape(-1,1)
            target_q[done] = 0
            target_q = reward + GAMMA * target_q
        
        q = self.model(state).gather(1, action)
        
        loss = F.mse_loss(q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), 5)
        self.optimizer.step()
        
        
    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or 
        # assign a values of network parameters via PyTorch methods.
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = np.array(state)
        state = torch.tensor(state, device=self.device)
        with torch.no_grad():
            pred = self.model(state)
        return pred.argmax(-1).cpu().numpy()

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model, "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns

def main(device):
    env = make("LunarLander-v2")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, device=device)
    eps = 0.1
    state = env.reset()
    
    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
    
    for i in range(TRANSITIONS):
        #Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
        if (i + 1) % (TRANSITIONS//100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            dqn.save()

if __name__ == "__main__":
    main(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    