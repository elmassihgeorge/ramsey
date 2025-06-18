from typing import List, Tuple
import networkx as nx

class Board:
    """
    Represents that graph that the players take turn "coloring"
    """
    def __init__(self, order : int = 5, graph : nx.Graph = None):
        assert order >= 0 or graph
        self.order = order if order else graph.order()
        self.graph = graph if graph else nx.complete_graph(order)
        if graph is None:
            nx.set_edge_attributes(self.graph, 'black', name = 'color')

    def color_edge(self, edge: Tuple[int, int], color: str):
        if color not in ["red", "blue"]:
            raise ValueError(f"Invalid color: {color}. Must be 'red' or 'blue'.")
        
        current_color = self.get_edge_color(edge)
        if current_color == "black":
            u, v = edge
            self.graph[u][v]['color'] = color
        else:
            raise Exception(f"Cannot color edge {edge} which is already {current_color}")


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
        return Board(order=self.order, graph=self.graph.copy())
    
    def get_monochromatic_clique_number(self, color="black") -> int:
        """
        Retrieves the size of the largest clique of such color
        """
        monochromatic_subgraph = self.get_monochromatic_subgraph(color=color)
        
        # Handle empty graph
        if len(monochromatic_subgraph.edges()) == 0:
            return 1 if len(monochromatic_subgraph.nodes()) > 0 else 0
        
        # Find all maximal cliques
        try:
            # This works in all NetworkX versions
            all_cliques = nx.find_cliques(monochromatic_subgraph)
            max_size = 0
            for clique in all_cliques:
                if len(clique) > max_size:
                    max_size = len(clique)
            return max_size
        except:
            # Fallback
            return 0
    
    def get_monochromatic_subgraph(self, color="black") -> nx.Graph:
        """
        Return a subgraph with the same vertices as self.board
        but only include edges of specified color
        """
        monochromatic_subgraph = nx.Graph()
        monochromatic_subgraph.add_nodes_from(range(self.order))
        edges = [(u, v, c) for u, v, c in self.graph.edges.data('color') if c == color]
        for (u, v, c) in edges:
            monochromatic_subgraph.add_edge(u, v, color=c)
        return monochromatic_subgraph
    
    def get_monochromatic_clique_subgraph(self, color="black", clique_size=3):
        """
        Like get_monochromatic_subgraph, but requires edges are part of a clique of specified order
        """
        monochromatic_subgraph = self.get_monochromatic_subgraph(color)
        all_cliques = nx.find_cliques(monochromatic_subgraph)
        filtered_cliques = [clique for clique in all_cliques if len(clique) == clique_size]
        for u, v in self.graph.edges():
            if not any(map((lambda clique : u in clique and v in clique), filtered_cliques)):
                try:
                    monochromatic_subgraph.remove_edge(u, v)
                except Exception as e:
                    continue
        return monochromatic_subgraph
        