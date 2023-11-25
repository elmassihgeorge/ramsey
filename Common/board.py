import networkx as nx

class Board:
    def __init__(self, order : int):
        self.graph = nx.complete_graph(order)
        nx.set_edge_attributes(self.graph, 'black', name = 'color')

    def color_edge(self, edge, color):
        if self.get_edge_color(edge) is not "black":
            raise Exception("Cannot color an edge that isn't black")
        u, v = edge
        self.graph[u][v]['color'] = color

    def get_edge_color(self, edge):
        u, v = edge
        return self.graph[u][v]['color']

    