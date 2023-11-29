from typing import List, Tuple
import networkx as nx

"""
Represents that graph that the players take turn "coloring"
"""
class Board:
    def __init__(self, order : int = 5, graph : nx.Graph = None):
        assert order >= 0 or graph
        self.order = order if order else graph.order()
        self.graph = graph if graph else nx.complete_graph(order)
        if graph is None:
            nx.set_edge_attributes(self.graph, 'black', name = 'color')

    def color_edge(self, edge : Tuple[int, int], color : str):
        """
        Assign a color to a black edge
        """
        if self.get_edge_color(edge) == "black":
            u, v = edge
            self.graph.add_edge(u, v, color=color)
        else:
            raise Exception("Cannot color an edge that isn't black")

    def get_edge_color(self, edge : Tuple[int, int]) -> str:
        """
        Retrieve the color of an edge
        """
        u, v = edge
        return self.graph[u][v]['color']
    
    def black_edges(self) -> List[Tuple[int, int]]:
        """
        Return edges that have not been colored
        """
        return [(u, v) for u, v, c in self.graph.edges.data('color') if c == 'black']
    
    def copy(self) -> "Board":
        """
        Creates a copy of this board using networkx.copy
        """
        return Board(graph=self.graph.copy())
    
    def get_monochromatic_clique_number(self, color="black") -> int:
        """
        Retrieves the size of the largest clique of such color
        """
        edges = [(u, v, c) for u, v, c in self.graph.edges.data('color') if c == color]
        monochromatic_subgraph = nx.Graph()
        for (u, v, c) in edges:
            monochromatic_subgraph.add_edge(u, v, color=c)
        return nx.graph_clique_number(monochromatic_subgraph)

        