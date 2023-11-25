import networkx as nx
from typing import List, Tuple
from ramsey.Common.board import Board
from ramsey.Common.move import Move
from ramsey.Common.player import Player

"""
A GameState which holds the board and metadata surrounding the game
"""
class GameState:
    def __init__(self, board : Board,
                clique_orders : Tuple[int],
                player : Player = Player.red,
                previous : "GameState" = None,
                move: Move = None):
        self.board = board
        self.current_player = player
        self.previous_state = previous
        self.last_move = move
        self.clique_orders = clique_orders
        self.s = self.clique_orders[0]
        self.t = self.clique_orders[1]
        self.colors = set(nx.get_edge_attributes(self.board.graph, "color").keys())

    """
    Applies colors to the graph and returns the successor state
    """
    def apply_move(self, move : Move) -> "GameState":
        if move.is_play:
            next_board = self.board.copy()
            next_board.color_edge(move.edge, move.color)
        else:
            next_board = self.board
        return GameState(board = next_board, 
                        clique_orders = self.clique_orders,
                        player = self.current_player.other,
                        previous = self, 
                        move = move)
    
    """
    Creates a new game with a specific complete graph of order k and clique orders s, t
    """
    @classmethod
    def new_game(cls, order, clique_orders):
        return GameState(Board(order), clique_orders)
    
    """
    A game is over when there is a red clique or order >= s, or a blue clique of order >= t
    """
    def is_over(self):
        if self.last_move is None:
            return False
        if len(self.board.black_edges()) == 0 or self.last_move.is_resign:
            return True
        return self.board.get_monochromatic_clique_number("red") >= self.s or self.board.get_monochromatic_clique_number("blue") >= self.t


