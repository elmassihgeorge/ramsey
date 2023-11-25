import sys, os
file_dir = os.path.dirname('ramsey')
sys.path.append(file_dir)

from matplotlib import pyplot as plt
import networkx as nx
from Common.view import View
from Common.move import Move
from Common.player import Player
from Common.board import Board

def test_coloring():
    board = Board(5)
    board.color_edge((0, 1), 'blue')
    board.color_edge((1, 3), 'red')
    View(board).render()

def test_color_on_already_colored():
    board = Board(5)
    board.color_edge((0, 1), 'blue')
    try:
        board.color_edge((0, 1), "red")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    test_coloring()
    test_color_on_already_colored()
    