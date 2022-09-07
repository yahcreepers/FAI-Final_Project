from game.players import BasePokerPlayer
from agents.model import DQN, ConvNet, DDPG
from .utils import *
import numpy as np
import torch
import random as rand
import copy

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

class DDPGPlayer(BasePokerPlayer):
    def __init__(self, do_train=True, model_path="./DDPG", batch_size=128, capacity=5000, device="cuda:4"):
        self.do_train = do_train
        self.model_path = model_path
        self.cache = []
        if self.do_train:
            self.model = DDPG(self.model_path, batch_size, capacity, c=device)
        else:
            self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
            self.model.actor = self.model.actor.to(device)
            self.model.act_target = self.model.act_target.to(device)
            self.model.critic = self.model.critic.to(device)
            self.model.cri_target = self.model.cri_target.to(device)
            self.model.c = device
        
        self.watch = ConvNet().to("cpu")
        self.watch.load_state_dict(torch.load("./embedding/model-999.pt"))
        self.step = 0
        self.update_step = 5000
        self.last_image = None
        self.last_features = None
        self.last_action = None
        
        self.CHE = 0
        
        #self.model = torch.load(model_path, map_location=lambda storage, loc: storage)

    def declare_action(self, valid_actions, hole_card, round_state):
        #print(self.step)
        community_card = round_state["community_card"]
        hole_card = gen_cards(hole_card)
        community_card = gen_cards(community_card)
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
        
        
        money = float(self.model.choose_action(img, features, valid_actions))
        
        if valid_actions[2]["amount"]["max"] == -1:
            if money < -10:
                action = valid_actions[0]["action"]
                amount = valid_actions[0]["amount"]
            else:
                action, amount = valid_actions[1]['action'], valid_actions[1]['amount']
        
        elif money < -10:
            action = valid_actions[0]["action"]
            amount = valid_actions[0]["amount"]
        elif money <= valid_actions[1]["amount"]:
            action = valid_actions[1]["action"]
            amount = valid_actions[1]["amount"]
        else:
            action = valid_actions[2]["action"]
            amount = min(max(money, valid_actions[2]["amount"]["min"]), valid_actions[2]["amount"]["max"])
        
        if self.do_train:
            if self.last_image != None:
                self.cache.append([self.last_image, self.last_features, self.last_action, img, features])
            self.last_image = copy.deepcopy(img)
            self.last_features = copy.deepcopy(features)
            self.last_action = copy.deepcopy(amount) if money > 0 else copy.deepcopy(money)
            #print(action, amount, money)
            
            if self.step > self.update_step:
                if self.CHE == 0:
                    print("-----START_LEARNING-------")
                    self.CHE = 1
                self.model.learn()
            
            self.step += 1
        
        return action, amount

    def receive_game_start_message(self, game_info):
        self.start_stack = game_info["rule"]["initial_stack"]

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_stack = [s['stack'] for s in seats if s['uuid'] == self.uuid][0]

    def receive_street_start_message(self, street, round_state):
        self.street_stack = [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0]

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
#        print("WWW")
        if self.do_train:
            if self.last_image != None:
                reward = [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0] - self.round_stack
#                print(reward)
#                print(winners)
#                print(hand_info)
                reward = (2 * (reward >= 0) - 1) * np.log(1 + abs(reward))
                #print(reward)
                for C in self.cache:
                    self.model.store_memory(C[0], C[1], C[2], reward, C[3], C[4])
                self.model.store_memory(self.last_image, self.last_features, self.last_action, reward, self.last_image, self.last_features)
            self.cache = []
            self.last_image = None
            self.last_features = None
            self.last_action = None


def setup_ai():
    return DDPGPlayer()

