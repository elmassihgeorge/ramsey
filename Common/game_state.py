import networkx as nx
from typing import List, Tuple
from Common.board import Board
from Common.move import Move
from Common.player import Player

"""
A GameState which holds the board and metadata surrounding the game
"""
class GameState:
    def __init__(self, board : Board,
                clique_orders : Tuple[int, int],
                player : Player = Player.red,
                last_state : "GameState" = None,
                last_move: Move = None):
        self.board = board
        self.active = player
        self.last_state = last_state
        self.last_move = last_move
        self.clique_orders = clique_orders
        self.s = self.clique_orders[0]
        self.t = self.clique_orders[1]

    @classmethod
    def new_game(cls, order : int, clique_orders : Tuple[int, int]) -> "GameState":
        """
        Creates a new game with a specific complete graph of order k and clique orders s, t
        """
        return GameState(board = Board(order),
                         clique_orders=clique_orders)
    
    def apply_move(self, move : Move) -> "GameState":
        """
        Applies colors to the graph and returns the successor state
        """
        if move.is_play:
            next_board = self.board.copy()
            next_board.color_edge(move.edge, move.color)
        else:
            next_board = self.board
        return GameState(board = next_board, 
                        clique_orders = self.clique_orders,
                        player = self.active.other,
                        last_state = self, 
                        last_move = move)
    
    def is_over(self) -> bool:
        """
        A game is over when there is a red clique of order >= s, or a blue clique of order >= t
        """
        if self.last_move is None:
            return False
        if len(self.board.black_edges()) == 0 or self.last_move.is_resign:
            return True
        return self.board.get_monochromatic_clique_number("red") >= self.s or self.board.get_monochromatic_clique_number("blue") >= self.t


