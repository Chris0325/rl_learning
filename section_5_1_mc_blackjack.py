from utils.util import *


class Status(Enum):
    STICKS = 0
    BUST = 1


class Player:

    def __init__(self):
        self.cards, self.usable_ace = [], False

    def deal_card(self):
        actual_card = card = np.random.choice(range(1, 11), p=[1/13]*9 + [4/13])
        if card == 1 and sum(self.cards) <= 10:
            card, self.usable_ace = 11, True
        self.cards.append(card)
        return actual_card

    def not_use_ace(self):
        for i, card in enumerate(self.cards):
            if card == 11:
                self.cards[i], self.usable_ace = 1, False

    def should_sticks(self):
        return sum(self.cards) >= 20
    
    def play(self):
        while True:
            if sum(self.cards) > 21:
                if self.usable_ace:
                    self.not_use_ace()
                    continue

                return Status.BUST

            if self.should_sticks():
                return Status.STICKS

            self.deal_card()


class Dealer(Player):

    def should_sticks(self):
        return sum(self.cards) >= 17


def sample_episode(usable_ace, show_card, cards, reward):
    return [((int(usable_ace), show_card-1, s-12), reward) for s in np.array(cards).cumsum() if s >= 12]


def update(V, C, episode):
    for s, r in episode:
        C[*s] += 1
        V[*s] += (r - V[*s]) / C[*s]


def run(n):
    V = np.zeros((2, 10, 10))
    C = np.zeros_like(V)

    for _ in tqdm(range(n)):
        player, dealer = Player(), Dealer()

        # initially each has two cards
        player.deal_card()
        dealer.deal_card()
        player.deal_card()
        show_card = dealer.deal_card()

        status = player.play()
        if status == Status.BUST:
            # exclude the bust card
            update(V=V, C=C, episode=sample_episode(usable_ace=player.usable_ace, show_card=show_card, cards=player.cards[:-1], reward=-1))

        elif sum(player.cards) == 21:
            # natural
            if sum(dealer.cards) == 21:
                update(V=V, C=C, episode=sample_episode(usable_ace=player.usable_ace, show_card=show_card, cards=player.cards, reward=0))
            else:
                update(V=V, C=C, episode=sample_episode(usable_ace=player.usable_ace, show_card=show_card, cards=player.cards, reward=1))

        else:
            status = dealer.play()
            if status == Status.BUST:
                update(V=V, C=C, episode=sample_episode(usable_ace=player.usable_ace, show_card=show_card, cards=player.cards, reward=1))
            else:
                player_hand_sum, dealer_hand_sum = sum(player.cards), sum(dealer.cards)
                if player_hand_sum > dealer_hand_sum:
                    update(V=V, C=C, episode=sample_episode(usable_ace=player.usable_ace, show_card=show_card, cards=player.cards, reward=1))
                elif player_hand_sum == dealer_hand_sum:
                    update(V=V, C=C, episode=sample_episode(usable_ace=player.usable_ace, show_card=show_card, cards=player.cards, reward=0))
                else:
                    update(V=V, C=C, episode=sample_episode(usable_ace=player.usable_ace, show_card=show_card, cards=player.cards, reward=-1))

    X, Y = np.meshgrid(range(1, 11), range(12, 22))
    fig = plt.figure(figsize=(12, 9))

    ax1 = fig.add_subplot(211, projection='3d')
    ax1.grid(False)
    surf1 = ax1.plot_surface(X, Y, V[1])
    ax1.set_title('Usable ace')
    ax1.set_xlim([1, 10])
    ax1.set_ylim([12, 21])
    ax1.set_zlim([-1, 1])

    ax2 = fig.add_subplot(212, projection='3d')
    ax2.grid(False)
    surf2 = ax2.plot_surface(X, Y, V[0])
    ax2.set_title('No usable ace')
    ax2.set_xlim([1, 10])
    ax2.set_ylim([12, 21])
    ax2.set_zlim([-1, 1])
    plt.show()


run(50_0000)
