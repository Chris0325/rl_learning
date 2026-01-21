from utils.blackjack_util import *

V = np.zeros((2, 10, 10))
C = np.zeros_like(V)


def run_init():
    # initially each has two cards
    player, dealer = Player(V=V, C=C), Dealer()

    player.deal_card()
    dealer.deal_card()
    player.deal_card()
    return player, dealer, dealer.deal_card()


def sample_episode(status, usable_ace, show_card, cards, reward):
    return [((int(usable_ace), show_card-1, s-12), reward) for s in np.array(cards).cumsum() if s >= 12]


run(50_0000, run_init=run_init, sample_episode=sample_episode, state_value=lambda x: x)
