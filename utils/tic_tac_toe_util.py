from utils.util import *


def numpy_to_tuple(s):
    return tuple([tuple(row.tolist()) for row in s])


class Player:

    def __init__(self, chess, a, greedy=.1):
        self.chess = chess
        self.greedy = greedy
        self.a = a
        self.need_explore = True
        self.value = defaultdict(lambda: 0.5)

        # backup usage
        self.last_ss = []

    def backup(self, s):
        # add another s when opponent wins
        if s != self.last_ss[-1][0]:
            self.last_ss.append((s, False))
    
        for (current_s, current_greedy), (prev_s, _) in zip(self.last_ss[1:], self.last_ss[:-1]):
            if not current_greedy:
                self.value[prev_s] += self.a * (self.value_of(current_s) - self.value_of(prev_s))
        
        # only keep the last s for possible ending backup
        self.last_ss = self.last_ss[-1:]

    def value_of(self, s):
        if s in self.value:
            return self.value[s]
        
        who = who_wins(s)
        if who == self.chess:
            self.value[s] = 1
        elif who in [-self.chess, 2]:
            self.value[s] = 0
        else:
            self.value[s] = .5
        return self.value[s]

    def step(self, s):
        # refresh last_ss
        self.last_ss = [(s, False)]

        s = np.array(s)
        new_ss = []
        for i in range(3):
            for j in range(3):
                if s[i, j] == 0:
                    new_s = s.copy()
                    new_s[i, j] = self.chess
                    new_ss.append(numpy_to_tuple(new_s))

        if self.need_explore and np.random.random() < self.greedy:
            new_s = random.choice(new_ss)
            self.last_ss.append((new_s, True))
        else:
            max_v = max(self.value_of(s) for s in new_ss)
            new_s = random.choice([s for s in new_ss if self.value_of(s) == max_v])
            self.last_ss.append((new_s, False))

        return new_s


def who_wins(s):
    for row in s:
        if abs(sum(row)) == 3:
            return sum(row) / 3

    for j in range(3):
        col = [s[i][j] for i in range(3)]
        if abs(sum(col)) == 3:
            return sum(col) / 3

    diag1 = [s[i][i] for i in range(3)]
    if abs(sum(diag1)) == 3:
        return sum(diag1) / 3

    diag2 = [s[i][2 - i] for i in range(3)]
    if abs(sum(diag2)) == 3:
        return sum(diag2) / 3
    
    if all(s[i][j] != 0 for i in range(3) for j in range(3)):
        return 2

    return 0


def self_play(runs=100):
    player_a = Player(1, 0.1)
    player_b = Player(-1, 0.1)
    for n in tqdm(range(runs), desc="self play"):
        s = numpy_to_tuple(np.zeros((3, 3), dtype=int))
        if n % 2 == 0:
            player1, player2 = player_a, player_b
        else:
            player1, player2 = player_b, player_a
        
        while True:
            s = player1.step(s)
            if who_wins(s) != 0:
                player1.backup(s)
                player2.backup(s)
                break
            else:
                player1.backup(s)

            s = player2.step(s)
            if who_wins(s) != 0:
                player1.backup(s)
                player2.backup(s)
                break
            else:
                player2.backup(s)
    
    return player_a
