from game.engine.card import Card
from game.engine.deck import Deck
import numpy as np
import pickle
from agents.utils import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import os
import random
from agents.model import ConvNet
import torch
import random as rand
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


def main(args):
    
    suits = list(Card.SUIT_MAP.keys())
    ranks = list(Card.RANK_MAP.keys())
    def gen_cards_im(cards):
        a = torch.zeros(4, 13)
        for i, card in enumerate(cards):
            s = suits.index(card.suit)
            r = ranks.index(card.rank)
            a[s][r] = 1
        return torch.nn.functional.pad(a, (2, 2, 6, 7))
    def gen_N_cards(n):
        index = np.random.choice([i for i in range(1, 53)], n)
        return [Card.from_id(id) for id in index]
    
    model = ConvNet().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.L1Loss().to(args.device)
#    data = []
#    for i in tqdm(range(10000)):
#        c = gen_N_cards(np.random.choice([2, 5, 6, 7]))
#        hc = c[:2]
#        cc = None
#        if len(c) > 2:
#            cc = c[2:]
#        wr = estimate_hole_card_win_rate(1000, 2, hc, cc)
#        data.append([c, wr])
#        if i % 1000 == 0 and i != 0:
#            print(i)
#
#    with open('small_maping.pkl', 'wb') as f:
#        pickle.dump(data, f)
    with open(args.train_file, "rb") as f:
        data = pickle.load(f)
    dataset = []
    for i in data:
        hc = i[0]
        cc = i[1]
        hc = [Card.from_id(id) for id in hc]
        cc = [Card.from_id(id) for id in cc]
        hc = gen_cards_im(hc)
        cc = gen_cards_im(cc)
        un = hc + cc
        dataset.append((torch.stack([hc, cc, un]), i[2]))
        
    with open(args.eval_file, "rb") as f:
        eval_data = pickle.load(f)
    eval_dataset = []
    for i in eval_data:
        hc = i[0][:2]
        cc = i[0][2:]
        hc = gen_cards_im(hc)
        cc = gen_cards_im(cc)
        un = hc + cc
        eval_dataset.append((torch.stack([hc, cc, un]), i[1]))
    
    train_loader = DataLoader(dataset[:99000], batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(dataset[99000:], batch_size=args.batch_size, shuffle=True)
    x = []
    y = []
    for i in tqdm(range(args.epoch)):
        for c, wr in train_loader:
            c = c.to(args.device)
            wr = wr.to(args.device)
            output = model(c)
            loss = loss_func(output, wr.unsqueeze(1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if i % args.eval_epoch == 0:
            torch.save(model.state_dict(), f"{args.output_dir}/model-{i}.pt")
            with torch.no_grad():
                total_loss = 0
                for c, wr in tqdm(eval_loader):
                    c = c.to(args.device)
                    wr = wr.to(args.device)
                    output = model(c)
                    loss = loss_func(output, wr.unsqueeze(1))
                    total_loss += loss.item()
                print(f"epoch-{i}", total_loss/len(eval_loader))
                x.append(i)
                y.append(total_loss/len(eval_loader))
                plt.plot(x, y)
                plt.savefig("Embedding-Loss.png")
    torch.save(model.state_dict(), f"{args.output_dir}/model-{i}.pt")
    with torch.no_grad():
        total_loss = 0
        for c, wr in tqdm(eval_loader):
            c = c.to(args.device)
            wr = wr.to(args.device)
            output = model(c)
            loss = loss_func(output, wr.unsqueeze(1))
            total_loss += loss.item()
        print(f"epoch-{i}", total_loss/len(eval_loader))
        x.append(i)
        y.append(total_loss/len(eval_loader))
        plt.plot(x, y)
        plt.savefig("Embedding-Loss.png")


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--do_train", action="store_true"
    )
    parser.add_argument(
        "--do_predict", action="store_true"
    )
    parser.add_argument(
        "--train_file", default='dataset.pkl', type=str
    )
    parser.add_argument(
        "--eval_file", default='maping.pkl', type=str
    )
    parser.add_argument(
        "--output_dir", default='embedding/', type=str, help="dir to save finetuned model"
    )
    parser.add_argument(
        "--epoch", default=1000, type=int, help="total number of epoch"
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="training batch size"
    )
    parser.add_argument(
        "--eval_epoch", default=50, type=int
    )
    parser.add_argument(
        "--device", default="cpu", type=str
    )
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
#    if not os.path.exists(args.output_dir + "/runs"):
#        os.mkdir(args.output_dir + "/runs")
    main(args)
