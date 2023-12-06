from enum import Enum

class Player(Enum):
    """
    Data representation for a player
    """
    red = 1
    blue = 2

    @property
    def other(self):
        return Player.red if self == Player.blue else Player.blue
