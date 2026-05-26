from utils.tic_tac_toe_util import *

player = self_play(100000)
player.need_explore = False


def state_to_graph(s):
    graph = ""
    for row in s:
        for cell in row:
            if cell == 0:
                graph += "."
            elif cell == 1:
                graph += "X"
            else:
                graph += "O"
        graph += "\n"
    return graph


def game_result(who):
    if who == 1:
        return "you lose!"
    elif who == -1:
        return "you win!"
    else:
        return "tie!"



while True:
    s = np.zeros((3, 3), dtype=int)
    answer = input("who first? (you/me): ")
    if answer == "you":
        s = player.step(numpy_to_tuple(s))

    while True:
        print(state_to_graph(s))
        answer = input("input your move (row col): ")
        row, col = map(int, answer.split())
        s = np.array(s)
        if s[row-1, col-1] != 0:
            print("invalid move, try again")
            continue
        s[row-1, col-1] = -1
        s = numpy_to_tuple(s)
        who = who_wins(s)
        if who:
            print(state_to_graph(s))
            print(game_result(who))
            player.backup(s)
            break

        s = player.step(s)
        who = who_wins(s)
        player.backup(s)
        if who:
            print(state_to_graph(s))
            print(game_result(who))
            player.backup(s)
            break

    answer = input("play again? (y/n): ")
    if answer != "y":
        break
