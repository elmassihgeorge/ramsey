from Common.board import Board
from typing import List, Tuple
import networkx as nx

class State:
    def __init__(self, board : Board, clique_orders : Tuple[int]):
        self.board = board
        # Idea: Given the main "board", extract subgraphs of different colors
        colors = set(nx.get_edge_attributes(self.board.graph, "color").keys())