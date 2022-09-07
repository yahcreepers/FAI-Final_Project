from game.players import BasePokerPlayer
from agents.model import DQN, ConvNet
#from .utils import *
import os
from game.engine.card import Card
import numpy as np
import torch
import random as rand
import copy
from collections import deque
import sys
sys.path.insert(0, '')

round_map = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
suits = list(Card.SUIT_MAP.keys())
ranks = list(Card.RANK_MAP.keys())

def gen_cards_im(cards):
    a = torch.zeros(4, 13)
    for i, card in enumerate(cards):
        s = suits.index(card.suit)
        r = ranks.index(card.rank)
        a[s][r] = 1
    return torch.nn.functional.pad(a, (2, 2, 6, 7))

class Player(BasePokerPlayer):
    def __init__(self, do_train=False, model_path="", n_actions=5, batch_size=128, capacity=5000, device="cuda:0", check_path=None):
        self.do_train = do_train
        self.model_path = model_path
        self.check_path = check_path
        self.device = device
        self.step = 0
        self.update_step = 5000
        self.remain_round = 0
        #print(os.getcwd())
        if self.do_train:
            if check_path != None:
                #print(f"........load model from {check_path}........")
                self.model = torch.load(self.check_path, map_location=torch.device('cpu'))
                self.model.eval_net = self.model.eval_net.to(self.device)
                self.model.target_net = self.model.target_net.to(self.device)
                self.model.c = self.device
                self.model.e = -1
#                self.model.optimizer = torch.optim.Adam(self.model.eval_net.parameters(), lr=1e-4)
#                self.model.step = 0
#                self.update_step = -1
            else:
                self.model = DQN(self.model_path, n_actions, 0.2, batch_size, capacity, c=device)
        else:
            self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
            self.model.eval_net = self.model.eval_net.to(self.device)
            self.model.target_net = self.model.target_net.to(self.device)
            tor
            self.model.c = self.device
            self.model.e = -1
        
        self.watch = ConvNet().to("cpu")
        #print(os.path)
        self.watch.load_state_dict(torch.load("./agents/model-999.pt"))
        self.last_image = None
        self.last_features = None
        self.last_action = None
        self.last_amount = None
        self.cache = []
        self.C = 0
        self.F = 0
        
        self.CHE = 0
        
        #self.model = torch.load(model_path, map_location=lambda storage, loc: storage)

    def declare_action(self, valid_actions, hole_card, round_state):
        #print(self.step)
        community_card = round_state["community_card"]
        hole_card = [Card.from_str(s) for s in hole_card]
        community_card = [Card.from_str(s) for s in community_card]
        hc = gen_cards_im(hole_card)
        cc = gen_cards_im(community_card)
        un = hc + cc
        img = torch.stack([hc, cc, un])
        with torch.no_grad():
            wr = self.watch(img)
        #wr = estimate_hole_card_win_rate(nb_simulation=5000, nb_player=2, hole_card=hole_card, community_card=community_card)
        #print(wr)
        features = [round_state['pot']['main']['amount'], round_state['small_blind_pos'], round_state['big_blind_pos'], round_state['dealer_btn'], round_state['next_player'], round_state['round_count'], wr]
        features.extend([s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid])
        features.extend([s['stack'] for s in round_state['seats'] if s['uuid'] != self.uuid])
        features.extend([0 if i != round_map[round_state['street']] else 1 for i in range(4)])
        features.append([s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0] - self.street_stack)
        features.append([s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0] - self.round_stack)
        features.append(round_state['pot']['main']['amount'] - [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0] + self.round_stack)
        features = torch.Tensor(features)
        
        if [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0] - self.start_stack > self.thrs and not self.do_train:
            #print(self.thrs, [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0], self.b)
            return valid_actions[0]["action"], valid_actions[0]["amount"]
        
        if self.do_train == False:
            with torch.no_grad():
                action_num = self.model.choose_action(img, features, valid_actions)
        else:
            action_num = self.model.choose_action(img, features, valid_actions)
            
        if action_num == 0:
            action = valid_actions[0]["action"]
            amount = valid_actions[0]["amount"]
        if action_num == 1:
            action = valid_actions[1]["action"]
            amount = valid_actions[1]["amount"]
        if action_num == 2:
            action = valid_actions[2]["action"]
            amount = valid_actions[2]["amount"]["min"]
        if action_num == 3:
            action = valid_actions[2]["action"]
            amount = valid_actions[2]["amount"]["max"]
        if action_num == 4:
            action = valid_actions[2]["action"]
            amount = valid_actions[2]["amount"]["max"] // 2
            if amount < valid_actions[2]["amount"]["min"]:
                amount = (valid_actions[2]["amount"]["max"] + valid_actions[2]["amount"]["min"]) // 2
        if amount < 0:
            action, amount = valid_actions[1]['action'], valid_actions[1]['amount']
        
        if self.do_train:
            if self.last_image != None:
                #self.model.store_memory(self.last_image, self.last_features, self.last_action, 0, img, features)
                self.F += 1
                self.cache.append([self.last_image, self.last_features, self.last_action, self.last_amount, img, features])
            self.last_image = copy.deepcopy(img)
            self.last_features = copy.deepcopy(features)
            self.last_action = copy.deepcopy(action_num)
            self.last_amount = copy.deepcopy(amount if amount > 0 else 0)
            #self.cache.append([copy.deepcopy(img), copy.deepcopy(features), copy.deepcopy(action_num)])
            
            if self.model.mem.tree.counter >= self.model.mem.tree.capacity:
                if self.CHE == 0:
                    print(f"-----START_LEARNING-------{self.step, self.model.mem.tree.counter, self.model.mem.tree.capacity, self.C, self.F}")
                    self.CHE = 1
                self.model.learn()
            
            self.step += 1
        
        return action, amount

    def receive_game_start_message(self, game_info):
        self.start_stack = game_info["rule"]["initial_stack"]
        self.remain_round = game_info["rule"]['max_round']
        self.blind = game_info["rule"]["small_blind_amount"]
        self.first = (game_info["seats"][0]["uuid"] == self.uuid)
        self.thrs = self.blind * self.remain_round // 2 + self.remain_round // 2 * self.blind * 2 + self.remain_round % 2 * self.blind * (-self.first + 2)
        self.b = self.blind * (-self.first + 2)
        

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_stack = [s['stack'] for s in seats if s['uuid'] == self.uuid][0]

    def receive_street_start_message(self, street, round_state):
        self.street_stack = [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0]

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
#        print("WWW")
        self.thrs -= self.b
        if self.b == self.blind * 2:
            self.b = self.blind
        elif self.b == self.blind:
            self.b = self.blind * 2
        if self.do_train:
#            if self.last_image != None:
#                self.F += 1
#                reward = [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0] - self.round_stack
##                print(reward)
##                print(winners)
##                print(hand_info)
#                reward = (2 * (reward >= 0) - 1) * np.log(1 + abs(reward))
#                #print(reward)
#                self.model.store_memory(self.last_image, self.last_features, self.last_action, reward, self.last_image, self.last_features)
#            self.last_image = None
#            self.last_features = None
#            self.last_action = None
            self.C += len(self.cache)
            reward = [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0] - self.round_stack
            reward /= self.blind * 2
            r = 0
            for i in range(len(self.cache)):
                if i == len(self.cache) - 1:
                    r = reward
                else:
                    r = self.cache[i][3] * (2 * (reward > 0) - 1) / (self.blind * 2)
                self.model.store_memory(self.cache[i][0], self.cache[i][1], self.cache[i][2], r, self.cache[i][4], self.cache[i][5])
            #print(self.last_image == self.cache[len(self.cache) - 1][3], self.last_features == self.cache[len(self.cache) - 1][4])
            #print(self.cache)
            self.cache = []
            self.last_image = None
            self.last_features = None
            self.last_action = None
            self.last_amount = None
#            if len(self.cache) <= self.cache_len:
#                self.model.store_memory(self.cache[0][0], self.cache[0][1], reward * self.model.gamma ** (len(self.cache) - 1), self.cache[-1][0], self.cache[-1][1])
#            else:
#                for i in range(len(self.cache) - self.cache_len + 1):
#                    if i == len(self.cache) - self.cache_len:
#                        self.model.store_memory(self.cache[i][0], self.cache[i][1], reward * (self.model.gamma ** (self.cache_len - 1)), self.cache[i + self.cache_len - 1][0], self.cache[i + self.cache_len - 1][1])
#                    else:
#                        self.model.store_memory(self.cache[i][0], self.cache[i][1], 0, self.cache[i + self.cache_len - 1][0], self.cache[i + self.cache_len - 1][1])


def setup_ai(do_train=False, model_path="./agents/DQN-PER-10000.pt", n_actions=5, batch_size=128, capacity=5000, device="cpu", check_path=""):
    return Player(do_train, model_path, n_actions, batch_size, capacity, device, check_path)
