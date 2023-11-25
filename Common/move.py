"""
A move that a player may make to color the board.
A player may pass their turn, resign, or play by coloring an edge.
"""
class Move:
    def __init__(self, edge=None, color=None, is_pass=False, is_resign=False):
        assert (edge is not None) ^ is_pass ^ is_resign
        self.edge = edge
        self.color = color
        self.is_play = (self.edge is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign

    @classmethod
    def play(cld, edge, color):
        return Move(edge=edge, color=color)
    
    @classmethod
    def pass_turn(cls):
        return Move(is_pass=True)
    
    @classmethod
    def resign(cls):
        return Move(is_resign=True)
    
    def __repr__(self):
        if self.is_play:
            return "[color {} {}]".format(self.edge, self.color)
        elif self.is_pass:
            return "[pass]"
        elif self.is_resign:
            return "[resign]"