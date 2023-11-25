from matplotlib import pyplot as plt
import networkx as nx

"""
View wrapper for boards that display them with PyPlot
"""
class View():
    def __init__(self, board):
        self.board = board
        self.colors = [board.graph[u][v]['color'] for u,v in board.graph.edges()]

    def render(self):
        nx.draw(self.board.graph, edge_color = self.colors)
        plt.show()