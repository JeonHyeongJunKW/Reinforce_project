ENV ='CartPole-v0'
GAMMA =0.99
MAX_STEPS = 200
NUM_EPISODES=500
import numpy as np
import matplotlib.pyplot as plt
import gym
import random
from collections import namedtuple

Transition = namedtuple('Transition',('state','action','next_state','reward'))



class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

BATCH_SIZE = 32
CAPACITY = 10000

class Net(nn.Module):

    def __init__(self, n_in, n_mid,n_out):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(n_in,n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self,x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output


class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)

        n_in,n_mid,n_out = num_states,32,num_actions
        self.main_q_network = Net(n_in,n_mid,n_out)
        self.target_q_network = Net(n_in, n_mid, n_out)
        print(self.main_q_network)

        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)

    def replay(self):

        if len(self.memory) < BATCH_SIZE:
            return

        self.batch,self.state_batch,self.action_batch,self.reward_batch,self.non_final_next_states = self.make_minibatch()

        self.expected_state_action_values = self.get_expected_state_action_values()

        self.update_main_q_network()



    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        return action

    def make_minibatch(self):
        transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 각 변수를 미니배치에 맞는 형태로 변형
        # transitions는 각 단계 별로 (state, action, state_next, reward) 형태로 BATCH_SIZE개수만큼 저장됨
        # 기존 (state, action, state_next,reward)*BATCH_SIZE형태로 되어있다. 이를 미니배치로 바꾸려면 다음과같은 형태로 되어야한다.
        # (state*BATCH_SIZE,action*BATCH_SIZE, state_next*BATCH_SIZE,reward*BATCH_SIZE)로 바꾼다.
        batch = Transition(*zip(*transitions))

        # 2.3각 변수 요소를 미니배치에 맞게 변형하고, 신경망으로 다룰 수 있게 Variable로 만든다.
        # state를 예로 들면, [torch.FlotTensor of size 1*4] 형태의 요소가 BATCH_SIZE 개수만큼 있는 형태이다.
        # cate Concatenates를 의미한다.
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        return batch, state_batch, action_batch,reward_batch,non_final_next_states
    def get_expected_state_action_values(self):
        self.main_q_network.eval()
        self.target_q_network.eval()

        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))  # true false로 계산하겟지

        next_state_values = torch.zeros(BATCH_SIZE)

        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).max(1)[1].detach()

        a_m_non_final_next_states = a_m[non_final_mask].view(-1,1).type(torch.int64)

        next_state_values[non_final_mask] = self.target_q_network(self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        self.main_q_network.train()

        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()


class Environment:

    def __init__(self):
        self.env = gym.make(ENV)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.agent = Agent(num_states, num_actions)

    def run(self):

        episode_10_list = np.zeros(10)

        complete_episodes = 0
        episode_final = False
        frames = []

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()

            state = observation
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            for step in range(MAX_STEPS):

                #                 if episode_final is True:

                #                     frames.append(self.env.render(mode='rgb_array'))
                if episode_final:
                    self.env.render()

                action = self.agent.get_action(state, episode)

                observation_next, _, done, _ = self.env.step(action.item())

                if done:
                    state_next = None

                    episode_10_list = np.hstack((episode_10_list[1:], step + 1))

                    if step < 195:
                        reward = torch.FloatTensor([-1.0])
                        complete_episodes = 0

                    else:
                        reward = torch.FloatTensor([1.0])

                        complete_episodes = complete_episodes + 1
                else:
                    reward = torch.FloatTensor([0.0])
                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)

                self.agent.memorize(state, action, state_next, reward)

                self.agent.update_q_function()

                state = state_next

                if done:
                    print("%d Episode: Finished after %d steps : 최근 10 에피소드의 평균 단계 수 = %.1f" % (
                    episode, step + 1, episode_10_list.mean()))
                    if(episode %2 ==0):
                        self.agent.update_target_q_function()
                    break

                if episode_final is True:
                    # display_frames_as_gif(frames)
                    break

                if complete_episodes >= 10:
                    print('10 에피소드 연속 성공')
                    episode_final = True

cartpole_env = Environment()
cartpole_env.run()

