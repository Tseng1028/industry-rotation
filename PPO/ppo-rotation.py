import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.distributions import Categorical

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(self, batch_size):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ActorNetwork, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

class CriticNetwork(nn.Module):
    def __init__(self, input_dims):
        super(CriticNetwork, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        return value

class PPOAgent:
    def __init__(
            self, 
            input_dims,
            n_actions,
            gamma=0.99,
            alpha=0.0003,
            gae_lambda=0.95,
            policy_clip=0.2,
            batch_size=64,
            n_epochs=10):
        
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size

        self.actor = ActorNetwork(input_dims, n_actions)
        self.critic = CriticNetwork(input_dims)
        self.memory = PPOMemory()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha)

    def choose_action(self, observation):
        state = torch.FloatTensor(observation).unsqueeze(0)
        
        dist = self.actor(state)
        value = self.critic(state)
        
        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches(self.batch_size)

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            advantage = torch.tensor(advantage)
            values = torch.tensor(values)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float)
                old_probs = torch.tensor(old_prob_arr[batch])
                actions = torch.tensor(action_arr[batch])

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.memory.clear_memory()

class IndustryRotationEnvironment:
    def __init__(self, financial_data, cluster_data, returns_data, lookback=126):
        self.financial_data = financial_data
        self.cluster_data = cluster_data
        self.returns_data = returns_data
        self.lookback = lookback
        self.current_step = lookback
        
    def reset(self):
        self.current_step = self.lookback
        state = self._get_state()
        return state
        
    def step(self, action):
        # Get the returns for the chosen cluster
        next_return = self.returns_data.iloc[self.current_step+1].mean()
        
        # Calculate reward (can be modified based on specific requirements)
        reward = next_return
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.returns_data) - 1
        
        # Get new state
        next_state = self._get_state()
        
        return next_state, reward, done
        
    def _get_state(self):
        # Combine financial indicators and cluster data for the state
        financial_features = self.financial_data.iloc[self.current_step].values
        cluster_features = self.cluster_data.iloc[self.current_step].values
        state = np.concatenate([financial_features, cluster_features])
        return state

def train_rotation_model(env, agent, n_episodes=100):
    best_reward = float('-inf')
    scores = []
    
    for episode in range(n_episodes):
        observation = env.reset()
        done = False
        score = 0
        
        while not done:
            action, prob, val = agent.choose_action(observation)
            next_observation, reward, done = env.step(action)
            
            agent.memory.store_memory(observation, action, prob, val, reward, done)
            score += reward
            
            if len(agent.memory.states) >= agent.batch_size:
                agent.learn()
                
            observation = next_observation
            
        scores.append(score)
        
        avg_score = np.mean(scores[-100:])
        if avg_score > best_reward:
            best_reward = avg_score
            
        print(f'Episode {episode} Score: {score:.2f} Avg Score: {avg_score:.2f}')
    
    return scores

# Helper function to prepare data
def prepare_data(financial_data, cluster_data, returns_data):
    # Ensure all dataframes are aligned by date
    common_dates = sorted(set(financial_data.index) & 
                        set(cluster_data.index) & 
                        set(returns_data.index))
    
    financial_data = financial_data.loc[common_dates]
    cluster_data = cluster_data.loc[common_dates]
    returns_data = returns_data.loc[common_dates]
    
    return financial_data, cluster_data, returns_data
