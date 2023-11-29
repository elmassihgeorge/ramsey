from matplotlib import pyplot as plt
import networkx as nx

"""
GUI for boards that display them with PyPlot
"""
class View():
    def __init__(self, board):
        self.board = board
        self.colors = [board.graph[u][v]['color'] for u,v in board.graph.edges()]

    def render(self):
        """
        Draws the current grpah and plots in a pyplot window
        """
        nx.draw(self.board.graph, edge_color = self.colors)
        plt.show()