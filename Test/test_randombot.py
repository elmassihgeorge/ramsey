import sys, os
file_dir = os.path.dirname('ramsey')
sys.path.append(file_dir)

import time
from Common.view import View
from Common.game_state import GameState
from Common.player import Player
from Agent.random_bot import RandomBot

def random_game():
    """
    Two random bot opponents attempt to color a K_5 graph
    without forming red or blue triangles
    """
    ORDER = 50
    CLIQUE_ORDERS = (40, 40)
    tic = time.time()
    game = GameState.new_game(ORDER, CLIQUE_ORDERS)
    bots = {
        Player.red: RandomBot(),
        Player.blue: RandomBot()
    }

    while not game.is_over():
        bot_move = bots[game.active].select_move(game)
        game = game.apply_move(bot_move)
    toc = time.time()
    print("time:", toc - tic)
    View(game.board).render()

if __name__ == "__main__":
    random_game()