#from game.engine.card import Card
#from game.engine.deck import Deck
#import numpy as np
#import pickle
#import random
#from agents.model import DQN
#import torch
#import random as rand
#
#model = DQN()
#img = torch.zeros(3, 17, 17)
#f = torch.zeros(13)
#for i in range(2005):
#    model.store_memory(img, f, 0, 0, img, f)
#model.learn()
#model.learn()

import json
from game.game import setup_config, start_poker
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from agents.console_player import setup_ai as console_ai
from agents.player import setup_ai as player_ai
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai

## Play in interactive mode if uncomment
#config.register_player(name="me", algorithm=console_ai())
P = player_ai()
players = [baseline0_ai(), baseline1_ai(), baseline2_ai(), baseline3_ai()]
win = [0, 0, 0, 0]
lose = [0, 0, 0, 0]
x = ["baseline0", "baseline1", "baseline2", "baseline3"]
for i, player in enumerate(players):
    for j in tqdm(range(20)):
        player = players[i]
        config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
        config.register_player(name="p1", algorithm=player)
        config.register_player(name="p2", algorithm=P)
        game_result = start_poker(config, verbose=1)
        if game_result["players"][0]["stack"] > game_result["players"][1]["stack"]:
            lose[i] += 1
        else:
            win[i] += 1
        print(game_result)
    print(win[i], lose[i], win[i]/(win[i] + lose[i]))

plt.bar(x, [win[i]/(win[i] + lose[i]) for i in range(4)])
plt.savefig("Test_Win_rate.png")
