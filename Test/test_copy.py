import sys, os
file_dir = os.path.dirname('ramsey')
sys.path.append(file_dir)

import time
import networkx as nx
from matplotlib import pyplot as plt
from copy import deepcopy
from ramsey.Common.view import View
from ramsey.Common.board import Board

def main():
    board = Board(20)
    board.color_edge((0, 1), 'blue')
    deepcopy_board(board)
    nxcopy_board(board)

def deepcopy_board(board):
    tic = time.time()
    deepcopy_board = deepcopy(board)
    toc = time.time()
    print("deepcopy time:", toc - tic)
    print("deepcopy board:", deepcopy_board.graph[0][1])

def nxcopy_board(board):
    tic = time.time()
    nxcopy_board = board.copy()
    toc = time.time()
    print("nxcopy time:", toc - tic)
    print("nxcopy board:", nxcopy_board.graph[0][1])
    View(nxcopy_board).render()

if __name__ == "__main__":
    main()