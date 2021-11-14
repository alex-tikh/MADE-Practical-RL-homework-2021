import pybullet_envs
# Don't forget to install PyBullet!
from gym import make
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim import Adam
import random

ENV_NAME = "Walker2DBulletEnv-v0"

LAMBDA = 0.95
GAMMA = 0.99

ACTOR_LR = 2e-4
CRITIC_LR = 1e-4

CLIP = 0.2
ENTROPY_COEF = 1e-2
BATCHES_PER_UPDATE = 64
BATCH_SIZE = 64

MIN_TRANSITIONS_PER_UPDATE = 2048
MIN_EPISODES_PER_UPDATE = 4

ITERATIONS = 1000

HIDDEN_DIM = 128

    
def compute_lambda_returns_and_gae(trajectory):
    lambda_returns = []
    gae = []
    last_lr = 0.
    last_v = 0.
    for _, _, r, _, v in reversed(trajectory):
        ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
        last_lr = ret
        last_v = v
        lambda_returns.append(last_lr)
        gae.append(last_lr - v)
    
    # Each transition contains state, action, old action probability, value estimation and advantage estimation
    return [(s, a, p, v, adv) for (s, a, _, p, _), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]
    


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
        return tanh_action, action, dis
        
        
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
        
    def get_value(self, state):
        return self.model(state)


class PPO:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.actor = Actor(state_dim, action_dim, self.device).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), CRITIC_LR)
        self.clip = CLIP

    def update(self, trajectories):
        transitions = [t for traj in trajectories for t in traj] # Turn a list of trajectories into list of transitions
        state, action, old_prob, target_value, advantage = zip(*transitions)
        state = np.array(state)
        action = np.array(action)
        old_prob = np.array(old_prob)
        target_value = np.array(target_value)
        advantage = np.array(advantage)
        advnatage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        
        for _ in range(BATCHES_PER_UPDATE):
            idx = np.random.randint(0, len(transitions), BATCH_SIZE) # Choose random batch
            s = torch.tensor(state[idx]).float().to(self.device)
            a = torch.tensor(action[idx]).float().to(self.device)
            op = torch.tensor(old_prob[idx]).float().to(self.device) # Probability of the action in state s.t. old policy
            v = torch.tensor(target_value[idx]).float().to(self.device) # Estimated by lambda-returns 
            adv = torch.tensor(advantage[idx]).float().to(self.device) # Estimated by generalized advantage estimation 
            
            # TODO: Update actor here
            proba, dist = self.actor.compute_proba(s, a)
            ratio = torch.exp(torch.log(proba + 1e-10) - torch.log(op + 1e-10))
            clip1 = ratio * adv
            clip2 = torch.clamp(ratiom, 1 - self.clip, 1 + self.clip) * adv
            actor_loss = -torch.mean(torch.min(clip1, clip2))
            
            entropy = dist.entropy().mean()
            actor_loss -= ENTROPY_COEF * entropy
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            
            # TODO: Update critic here
            critic_loss = F.smooth_l1_loss(self.critic.get_value(s).view(-1), v)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            
            
    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float().to(self.device)
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float().to(self.device)
            action, pure_action, distr = self.actor.act(state)
            prob = torch.exp(distr.log_prob(pure_action).sum(-1))
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], prob.cpu().item()

    def save(self):
        torch.save(self.actor, "agent.pkl")


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)
    return returns
   

def sample_episode(env, agent):
    s = env.reset()
    d = False
    trajectory = []
    while not d:
        a, pa, p = agent.act(s)
        v = agent.get_value(s)
        ns, r, d, _ = env.step(a)
        trajectory.append((s, pa, r, p, v))
        s = ns
    return compute_lambda_returns_and_gae(trajectory)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = make(ENV_NAME)
    ppo = PPO(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device
    )
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0
    
    for i in range(ITERATIONS):
        trajectories = []
        steps_ctn = 0
        
        while len(trajectories) < MIN_EPISODES_PER_UPDATE or steps_ctn < MIN_TRANSITIONS_PER_UPDATE:
            traj = sample_episode(env, ppo)
            steps_ctn += len(traj)
            trajectories.append(traj)
        episodes_sampled += len(trajectories)
        steps_sampled += steps_ctn

        ppo.update(trajectories)        
        
        if (i + 1) % (ITERATIONS//100) == 0:
            rewards = evaluate_policy(env, ppo, 5)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}, Episodes: {episodes_sampled}, Steps: {steps_sampled}")
            ppo.save()


if __name__ == "__main__":
    main()
