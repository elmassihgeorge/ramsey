import networkx as nx

"""
Represents that graph that the players take turn "coloring"
"""
class Board:
    def __init__(self, order : int = 0, graph = None):
        assert order or graph
        self.order = order if order else graph.order()
        self.graph = graph if graph else nx.complete_graph(order)
        if order:
            nx.set_edge_attributes(self.graph, 'black', name = 'color')

    """
    Assign a color to a black edge
    """
    def color_edge(self, edge, color):
        if self.get_edge_color(edge) is not "black":
            raise Exception("Cannot color an edge that isn't black")
        u, v = edge
        self.graph.add_edge(u, v, color=color)

    """
    Retrieve the color of an edge
    """
    def get_edge_color(self, edge):
        u, v = edge
        return self.graph[u][v]['color']
    
    """
    Retrieves the size of the largest clique of such color
    """
    def get_monochromatic_clique_number(self, color="black"):
        edges = [(u, v, c) for u, v, c in self.graph.edges.data('color') if c == color]
        subgraph = nx.Graph()
        for (u, v, c) in edges:
            subgraph.add_edge(u, v, color=c)
        return nx.graph_clique_number(subgraph)

    """
    Return edges that have not been colored
    """
    def black_edges(self):
        return [(u, v) for u, v, c in self.graph.edges.data('color') if c == 'black']
    
    """
    Creates a copy of this board using networkx.copy
    """
    def copy(self) -> "Board":
        return Board(graph=self.graph.copy())

    def __repr__(self):
        return str([(u, v, c) for u, v, c in self.graph.edges.data('color')])
        