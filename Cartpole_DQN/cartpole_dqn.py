import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import matplotlib.pylab as plt

class NeuralNetwork(nn.Module):

    def __init__(self, state_dim, action_size, hidden_layer=100):

        super(NeuralNetwork, self).__init__()
        self.input_size = state_dim
        self.action_size = action_size

        self.l1 = nn.Linear(state_dim, hidden_layer)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_layer, action_size)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)


    def forward(self, x):
        out1 = self.l1(x)
        out2 = self.relu(out1)
        self.q_state = self.l2(out2)
        return self.q_state
    
    def q_state_action(self, q_state, action_in):
        q_state_action = q_state[action_in.item()]
        return q_state_action

    def update(self, state, action_in, target_q):
        q_state = self.forward(state)
        q_state_action = self.q_state_action(q_state, action_in)

        loss = self.criterion(target_q, q_state_action)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().numpy()


class DQNagent():
    def __init__ (self, env):
        self.state_dim = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.q_network = NeuralNetwork(self.state_dim, self.action_size)

        self.gamma = 0.9
        self.epsilon = 1.0

    def get_action(self, state):
        q_state = self.q_network.forward(torch.from_numpy(state).float())
        action_greedy = torch.argmax(q_state).item()
        action_random = np.random.randint(self.action_size)
        action = action_random if random.random() < self.epsilon else action_greedy
        
        return action
    
    def train(self, state, action, next_state, reward, done):
        q_next_state = self.q_network.forward(torch.from_numpy(next_state).float())
        q_next_state = (1-done) * q_next_state 
        target_q = reward + self.gamma * torch.max(q_next_state).item()
        loss = self.q_network.update(torch.from_numpy(state).float(), torch.tensor(action), torch.tensor(target_q).float())

        if done: self.epsilon = max(0.1, 0.99*self.epsilon)

        return loss


env = gym.make('CartPole-v1', render_mode = 'human')
print("Observation space:", env.observation_space)
print("Action Space:", env.action_space)


agent = DQNagent(env)
num_episodes = 500
Tot_rewards = []
losses = []

for ep in range(num_episodes):
    (state, info) = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        episode_loss = agent.train(state, action, next_state, reward, done)
        env.render()
        total_reward += reward
        state = next_state
    Tot_rewards.append(total_reward)
    losses.append(episode_loss)
    print(f"episode no: {ep}, total reward: {total_reward:.4f}")

plt.figure(1)
plt.plot(range(len(Tot_rewards)), Tot_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title('Reward vs Episode')

plt.savefig('reward_vs_episode.png')

plt.figure(2)
plt.plot(range(len(losses)), losses)
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title('Loss vs Episode')

plt.savefig('loss_vs_episode.png')

plt.show()

