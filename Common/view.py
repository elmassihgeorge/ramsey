from matplotlib import pyplot as plt
import networkx as nx

class View():
    """
    GUI for boards that display them with PyPlot
    """
    def __init__(self, board):
        self.board = board

    def render(self):
        """
        Draws the current grpah and plots in a pyplot window
        """
        colors = [self.board.graph[u][v]['color'] for u,v in self.board.graph.edges()]
        nx.draw(self.board.graph, edge_color = colors)
        plt.show()