import numpy as np

from game.engine.card import Card
from game.engine.deck import Deck
from game.engine.hand_evaluator import HandEvaluator

def estimate_hole_card_win_rate(nb_simulation, nb_player, hole_card, community_card=None):
    if community_card == None:
        community_card = []
    win_count = sum([montecarlo_simulation(nb_player, hole_card, community_card) for i in range(nb_simulation)])
    return 1.0 * win_count / nb_simulation

def montecarlo_simulation(nb_player, hole_card, community_card):
    show_cards = [card.to_id() for card in hole_card + community_card]
    no_cards = [id for id in range(1, 53) if id not in show_cards]
    index = np.random.choice(no_cards, 5 - len(community_card) + 2 * (nb_player - 1))
    community_card += [Card.from_id(id) for id in index[:5 - len(community_card)]]
    opp_hole = [Card.from_id(id) for id in index[5 - len(community_card):]]
    opp_score = [HandEvaluator.eval_hand([opp_hole[2*i], opp_hole[2*i+1]], community_card) for i in range(0, len(opp_hole)//2)]
    my_score = HandEvaluator.eval_hand(hole_card, community_card)
    return 1 if my_score >= max(opp_score) else 0

