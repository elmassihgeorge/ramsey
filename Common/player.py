from enum import Enum

"""
A data representation for a player
"""
class Player(Enum):
    red = 1
    blue = 2

    @property
    def other(self):
        return Player.red if self == Player.blue else Player.blue
