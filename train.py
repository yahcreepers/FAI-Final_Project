import json
from game.game import setup_config, start_poker
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from agents.console_player import setup_ai as console_ai
from agents.player import setup_ai as player_ai
from agents.agent import setup_ai as DDPG_ai
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai

## Play in interactive mode if uncomment
#config.register_player(name="me", algorithm=console_ai())
#P = DDPG_ai()
P = player_ai(do_train=True, model_path="DQN-PER-REWARD", device="cuda:0", check_path=None)
#players = [baseline0_ai(), baseline1_ai(), baseline2_ai(), baseline3_ai(), player_ai(do_train=False, model_path="./dualing-31500.pt", device="cuda:7")]
#players = [baseline0_ai(), baseline1_ai(), baseline2_ai(), baseline3_ai()]
players = [random_ai()]
#players = [random_ai(), baseline0_ai(), baseline1_ai(), baseline2_ai(), baseline3_ai()]
win = [0 for i in range(len(players))]
lose = [0 for i in range(len(players))]
win_t = [0 for i in range(len(players))]
lose_t = [0 for i in range(len(players))]
win_s = [[] for i in range(len(players))]
win_T = []
x = []
for i in tqdm(range(20000)):
    index = np.random.randint(0, len(players))
    player = players[index]
    config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
    config.register_player(name="p1", algorithm=player)
    config.register_player(name="p2", algorithm=P)
    game_result = start_poker(config, verbose=1)
    if game_result["players"][0]["stack"] > game_result["players"][1]["stack"]:
        lose[index] += 1
        lose_t[index] += 1
        #print("Loss", index)
    else:
        win[index] += 1
        win_t[index] += 1
        #print("Win", index)
    if i % 100 == 0 and i != 0:
        plt.clf()
        x.append(i)
        for j in range(len(players)):
            win_s[j].append(win_t[j]/(win_t[j] + lose_t[j]))
            plt.plot(x, win_s[j], label=f"baseline-ai-{j}")
            print(f"Win Rate {j}", win_t[j]/(win_t[j] + lose_t[j]))
#        win_s[4].append(win_t[4]/(win_t[4] + lose_t[4]))
#        plt.plot(x, win_s[4], label=f"DQN")
#        print(f"Win Rate DQN", win_t[4]/(win_t[4] + lose_t[4]))
        win_T.append(sum(win_t)/(sum(win_t) + sum(lose_t)))
        plt.plot(x, win_T, label="Total")
        plt.ylabel("Win Rate")
        plt.xlabel("epoch")
        plt.legend(loc = 'upper right')
        print(f"Win Rate", sum(win_t)/(sum(win_t) + sum(lose_t)))
        win_t = [0 for i in range(len(players))]
        lose_t = [0 for i in range(len(players))]
        plt.savefig("Win_Rate-DQN-REWARD.png")

for i in range(4):
    print(f"Win Rate {i}", win[i]/(win[i] + lose[i]))
print(f"Win Rate", sum(win)/(sum(win) + sum(lose)))
