import sys, os
file_dir = os.path.dirname('ramsey')
sys.path.append(file_dir)

from Agent.random_bot import RandomBot
from Common.view import View
from Common.game_state import GameState
from Common.player import Player
from Encoder.two_color_encoder import TwoColorEncoder
from Encoder.k3_encoder import K3Encoder
from Encoder.k4_encoder import K4Encoder

def test_encoder():
    """
    Two random bot opponents attempt to color a K_5 graph
    without forming red or blue triangles
    """
    
    game = GameState.new_game(9, (5, 5))
    bots = {
        Player.red: RandomBot(),
        Player.blue: RandomBot()
    }

    while not game.is_over():
        bot_move = bots[game.active].select_move(game)
        game = game.apply_move(bot_move)

    encoder = K4Encoder(9)
    print(encoder.encode(game))
    View(game.board).render()

if __name__ == "__main__":
    test_encoder()