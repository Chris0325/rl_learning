from utils.blackjack_util import *

V = np.zeros((2, 10, 10, 2))
C = np.zeros_like(V)


class OptimalPlayer(Player):

    def __init__(self, V=None, C=None, cards=None):
        super().__init__(V, C, cards)
        if cards:
            self.usable_ace = np.random.choice(2)
            if self.usable_ace:
                self.cards = [self.cards[0]-11, 11]

    def should_sticks(self, show_card):
        action_value = self.V[int(self.usable_ace), show_card-1, sum(self.cards)-12]
        return np.random.choice(np.where(action_value == action_value.max())[0])


def run_init():
    player, dealer = OptimalPlayer(V=V, C=C, cards=[np.random.choice(range(12, 22))]), Dealer(cards=[np.random.choice(range(12, 22))])
    return player, dealer, np.random.choice(range(1, 11))


def sample_episode(status, usable_ace, show_card, cards, reward):
    to_ret = [[[int(usable_ace), show_card-1, s-12, 0], reward] for s in np.array(cards).cumsum() if s >= 12]
    to_ret[-1][0][-1] = (1 if status == Status.STICKS else 0)
    return to_ret


run(50_0000, run_init, sample_episode=sample_episode, state_value=lambda x: np.max(x, axis=-1))
