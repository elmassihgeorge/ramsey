import sys, os
file_dir = os.path.dirname('ramsey')
sys.path.append(file_dir)

from Common.view import View
from Common.game_state import GameState
from Common.player import Player
from Agent.random_bot import RandomBot

def main():

    order = 25
    clique_orders = (4, 5)
    game = GameState.new_game(order, clique_orders)

    bots = {
        Player.red: RandomBot(),
        Player.blue: RandomBot()
    }

    while not game.is_over():
        bot_move = bots[game.current_player].select_move(game)
        print(bot_move)
        game = game.apply_move(bot_move)

    View(game.board).render()

if __name__ == "__main__":
    main()