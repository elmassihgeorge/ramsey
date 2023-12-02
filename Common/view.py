from matplotlib import pyplot as plt
import networkx as nx

"""
GUI for boards that display them with PyPlot
"""
class View():
    def __init__(self, board):
        self.board = board

    def render(self):
        """
        Draws the current grpah and plots in a pyplot window
        """
        colors = [self.board.graph[u][v]['color'] for u,v in self.board.graph.edges()]
        nx.draw(self.board.graph, edge_color = colors)
        plt.show()