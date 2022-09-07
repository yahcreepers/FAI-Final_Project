from game.engine.card import Card
from game.engine.deck import Deck
from tqdm import tqdm
import pickle
from agents.utils import *
import numpy as np
import pickle
import random
from agents.model import DQN
import torch
import random as rand
import matplotlib.pyplot as plt


def gen_N_cards(n):
    index = np.random.choice([i for i in range(1, 53)], n)
    return [Card.from_id(id) for id in index]

See = False

if See:
    count = 10000
    x = []
    y = []
    w = []
    plt.clf()
    c = gen_N_cards(2)
    hc = c[:2]
    cc = None
    if len(c) > 2:
        cc = c[2:]
    for i in range(10):
        win_rate = []
        for j in range(100):
            wr = estimate_hole_card_win_rate(count + 10000 * i, 2, hc, cc)
            win_rate.append(wr)
        std = np.std(win_rate)
        x.append(count + 10000 * i)
        y.append(std)
        w.append(sum(win_rate)/100)
        if i % 2 == 0 and i != 0:
            print(f"Step: {i}")
            plt.clf()
            plt.plot(x, y)
            plt.savefig("BIG_monte-conv-std.png")
            plt.clf()
            plt.plot(x, w)
            print(x, y, w)
            plt.savefig("BIG_monte-conv-mean.png")
    plt.clf()
    plt.plot(x, y)
    plt.savefig("BIG_monte-conv-std.png")
    plt.clf()
    plt.plot(x, w)
    plt.savefig("BIG_monte-conv-mean.png")

else:
    data = []
    for i in tqdm(range(10000)):
        c = gen_N_cards(np.random.choice([2, 5, 6, 7]))
        hc = c[:2]
        cc = None
        if len(c) > 2:
            cc = c[2:]
        wr = estimate_hole_card_win_rate(50000, 2, hc, cc)
        data.append([c, wr])
        if i % 1000 == 0 and i != 0:
            print(i)
            with open('maping.pkl', 'wb') as f:
                pickle.dump(data, f)
    
    with open('maping.pkl', 'wb') as f:
        pickle.dump(data, f)
