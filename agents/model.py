import torch
import torch.nn as nn
import copy
import numpy as np
from .utils import *

class Net(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
#            nn.Linear(128, 64),
#            nn.ReLU(),
#            nn.Linear(64, n_actions),
        )
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advan = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )
    
    def forward(self, image, state):
        h_1 = self.layer1(image)
        h_2 = self.layer2(state)
        h_1 = torch.squeeze(h_1)
        #print(h_1.shape)
        if len(h_1.shape) == 1:
            h_3 = torch.cat((h_1, h_2))
        else:
            h_3 = torch.cat((h_1, h_2), 1)
        #print("AAA", h_1.shape, h_2.shape, h_3.shape)
        h_3 = self.layer3(h_3)
        
        value = self.value(h_3)
        advan = self.advan(h_3)
        q = value + (advan - advan.mean())
        return q

class SumTree():
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [0 for i in range(2 * capacity - 1)]
        self.data = [0 for i in range(capacity)]
        self.counter = 0
    
    def add(self, p, data):
        tree_ind = self.counter % self.capacity + self.capacity - 1
        self.update(tree_ind, p)
        self.data[self.counter % self.capacity] = data
        self.counter += 1
        
    def get(self, p):
        cur = 0
        ret = 0
        while True:
            left = 2 * cur + 1
            right = 2 * cur + 2
            if left >= min(self.counter, self.capacity):
                ret = cur
                break
            else:
                if p <= self.tree[left]:
                    cur = left
                else:
                    p -= self.tree[left]
                    cur = right
        data_ind = ret - self.capacity + 1
        return ret, self.data[data_ind]
        
    def back(self, ind, dt):
        parent = (ind - 1) // 2
        self.tree[parent] += dt
        if parent != 0:
            self.back(parent, dt)
        
    def update(self, ind, p):
        #print(ind)
        dt = p - self.tree[ind]
        self.tree[ind] = p
        self.back(ind, dt)

class Memory():
    def __init__(self, capacity=5000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.e = 0.01
        self.a = 0.6
        self.MP = 0.9
    
    def store(self, data):
        p = (self.MP + self.e) ** self.a
        self.tree.add(p, data)
        
    def sample(self, batch_size):
        memory = []
        index = []
        sep = min(self.tree.counter, self.capacity) / batch_size
        for i in range(batch_size):
            l = np.random.uniform(sep * i, sep * (i + 1))
            
            ind, data = self.tree.get(l)
            index.append(ind)
            memory.append(data)
        return index, memory
        
    def update(self, ind, loss):
        self.MP = max(self.MP, max(abs(loss)))
        for i in range(len(ind)):
            p = (abs(loss[i]) + self.e) ** self.a
            self.tree.update(ind[i], p)
    

class DQN():
    def __init__(self, model_path="dualing", n_actions=5, final_e=0.2, batch_size=128, capacity=5000, c="cuda:4"):
        self.model_path = model_path
        self.n_actions = n_actions
        self.e = 1
        self.final_e = final_e
        self.decay = 1e-3
        self.batch_size = batch_size
        self.mem = Memory(capacity)
        self.counter = 0
        self.capacity = capacity
        self.c = c
        self.step = 0
        self.save_step = 1000
        self.change_step = 100
        self.n_actions = n_actions
        self.gamma = 0.95
        self.eval_net = Net(n_actions).to(self.c)
        self.target_net = Net(n_actions).to(self.c)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=1e-4)
        self.loss_func = nn.MSELoss()
    
    def choose_action(self, image, state, valid_actions):
        if self.e > self.final_e and self.step >= self.capacity:
            self.e -= self.decay
        if np.random.uniform() < self.e or self.step < self.capacity:
            action_num = np.random.randint(0, self.n_actions)
        else:
            action_num = self.eval_net(image.to(self.c), state.to(self.c))
            action_num = torch.argmax(action_num)
        return action_num
    
    def store_memory(self, prev_image, prev_features, action_num, reward, image, features):
        self.mem.store([prev_image, prev_features, action_num, reward, image, features])
    
    def learn(self):
        if self.step % self.change_step == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.step % self.save_step == 0:
            torch.save(self, f"{self.model_path}-{self.step}.pt")
            print(f"Save at {self.model_path}-{self.step}.pt")
        self.step += 1
        #print(self.mem.tree.counter, self.mem.tree.capacity)
#        for i in range(len(self.mem.tree.data)):
#            if isinstance(self.mem.tree.data[i], int):
#                print("WWW")
        sample_index, data = self.mem.sample(self.batch_size)
#        print(data[0][0].shape, data[0][1], data[0][2], data[0][3], data[0][4].shape, data[0][5])
#        for i in data:
#            if isinstance(i, int):
#                print(i)
#            print(type(i))
        prev_image = torch.stack([m[0] for m in data]).to(self.c)
        prev_features = torch.stack([m[1] for m in data]).to(self.c)
        action_num = torch.tensor([[m[2]] for m in data]).to(self.c)
        reward = torch.FloatTensor([[m[3]] for m in data]).to(self.c)
        image = torch.stack([m[4] for m in data]).to(self.c)
        features = torch.stack([m[5] for m in data]).to(self.c)
        #print(prev_image.shape, prev_features.shape, image.shape, features.shape)
        
        q_eval = self.eval_net(prev_image, prev_features).gather(1, action_num)
        q_nxt = self.target_net(image, features).detach()
        q_target = reward + self.gamma * q_nxt.max(1)[0].view(self.batch_size, 1)
        td_error = q_target - q_eval
        self.mem.update(sample_index, td_error)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        h_1 = self.layer1(x)
        #print(h_1.shape)
        h_1 = torch.squeeze(h_1)
        #print(h_1.shape)
        h_2 = self.layer2(h_1)
        #print(h_2.shape)
        return h_2


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, image, state):
        #print(image.shape)
        h_1 = self.layer1(image)
        h_2 = self.layer2(state)
        h_1 = torch.squeeze(h_1)
        #print(h_1.shape)
        if len(h_1.shape) == 1:
            h_3 = torch.cat((h_1, h_2))
        else:
            h_3 = torch.cat((h_1, h_2), 1)
        #print("AAA", h_1.shape, h_2.shape, h_3.shape)
        h_3 = self.layer3(h_3)
        return h_3

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
    
        self.layer3 = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, image, state, action):
        #print("cri", action.shape)
        h_1 = self.layer1(image)
        h_2 = self.layer2(state)
        h_3 = self.layer3(action)
        h_4 = self.layer4(torch.cat([h_1.squeeze(), h_2, h_3], 1))
        return h_4


class DDPG():
    def __init__(self, model_path="DDPG", batch_size=128, capacity=5000, c="cuda:4"):
        self.model_path = model_path
        self.var = 10
        self.decay = 0.9997
        self.final_var = 0.01
        self.batch_size = batch_size
        self.mem = [0 for i in range(capacity)]
        self.counter = 0
        self.capacity = capacity
        self.c = c
        self.step = 0
        self.save_step = 1000
        self.T = 0.01
        self.gamma = 0.95
        self.actor = Actor().to(self.c)
        self.actor_target = copy.deepcopy(self.actor)
        self.act_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic = Critic().to(self.c)
        self.critic_target = copy.deepcopy(self.critic)
        self.cri_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.loss_func = nn.MSELoss()
    
    def choose_action(self, image, state, valid_actions):
        action = self.actor(image.to(self.c), state.to(self.c))
        action = np.random.normal(action.detach().to("cpu"), self.var)
        return action
    
    def store_memory(self, prev_image, prev_features, action_num, reward, image, features):
        self.mem[self.counter % self.capacity] = [prev_image, prev_features, action_num, reward, image, features]
        self.counter += 1
    
    def learn(self):
        #soft update
        if self.step % self.save_step == 0:
            torch.save(self, f"{self.model_path}-{self.step}.pt")
            print(f"Save at {self.model_path}-{self.step}.pt")
        self.var *= self.decay
        self.var = max(self.var, self.final_var)
        self.step += 1
        sample_index = np.random.choice(self.capacity, self.batch_size)
        data = [self.mem[i] for i in sample_index]
        prev_image = torch.stack([m[0] for m in data]).to(self.c)
        prev_features = torch.stack([m[1] for m in data]).to(self.c)
        action_num = torch.FloatTensor([[m[2]] for m in data]).to(self.c)
        reward = torch.FloatTensor([[m[3]] for m in data]).to(self.c)
        image = torch.stack([m[4] for m in data]).to(self.c)
        features = torch.stack([m[5] for m in data]).to(self.c)
        #print(prev_image.shape, prev_features.shape, image.shape, features.shape)
        
        #actor loss
        action_prev = self.actor(prev_image, prev_features)
        q = self.critic(prev_image, prev_features, action_prev)
        loss_a = -torch.mean(q)
        self.act_optimizer.zero_grad()
        loss_a.backward()
        self.act_optimizer.step()
        
        action = self.actor(image, features)
        q_nxt = self.critic(image, features, action)
        q_target = reward + self.gamma * q_nxt
        q_eval = self.critic(image, features, action_num)
        td_error = self.loss_func(q_eval, q_target)
        self.cri_optimizer.zero_grad()
        td_error.backward()
        self.cri_optimizer.step()
        
        for target, eval in zip(self.actor_target.parameters(), self.actor.parameters()):
            target.data.copy_(target.data * (1.0 - self.T) + eval.data * self.T)
        for target, eval in zip(self.critic_target.parameters(), self.critic.parameters()):
            target.data.copy_(target.data * (1.0 - self.T) + eval.data * self.T)
        
