from enum import Enum


class Move:
    PASS = "pass"
    RESIGN = "resign"

    def __init__(self, edge):
        self.edge = edge
        